#!/usr/bin/env python3
"""
Volatility forecasting for the forecaster (--volatility-only path).

Lightweight by design: imports ONLY numpy / pandas / requests so it can run
without TensorFlow and stays standalone-testable. Asset-agnostic — the caller
supplies the return series, an explicit annualization factor, and an
implied-volatility series, so the same code serves crypto and equities.

Implied-vol sources (fetched by the caller):
  - Crypto: Deribit DVOL (keyless public REST, fetched by fetch_dvol below).
  - Equity: a CBOE VIX-family index or option-chain ATM IV (fetched via yfinance
    in the caller and passed in as iv_df).
"""

import numpy as np
import pandas as pd
import requests

DERIBIT_DVOL_URL = "https://www.deribit.com/api/v2/public/get_volatility_index_data"


def fetch_dvol(currency='BTC', resolution='3600', days=60):
    """Fetch the Deribit DVOL implied-volatility index (crypto "VIX"), keyless.

    Returns a DataFrame [ts, open, high, low, close] (close = annualized vol in %),
    or None on network/HTTP failure so the caller can degrade gracefully.
    """
    end_ms = int(pd.Timestamp.utcnow().timestamp() * 1000)
    start_ms = end_ms - days * 24 * 60 * 60 * 1000
    params = {
        'currency': currency,
        'start_timestamp': start_ms,
        'end_timestamp': end_ms,
        'resolution': resolution,
    }
    try:
        resp = requests.get(DERIBIT_DVOL_URL, params=params, timeout=15)
        resp.raise_for_status()
        payload = resp.json()
    except (requests.RequestException, ValueError) as exc:
        print(f"  [volatility] DVOL fetch failed: {exc}")
        return None

    data = payload.get('result', {}).get('data', [])
    if not data:
        print("  [volatility] DVOL fetch returned no data")
        return None

    df = pd.DataFrame(data, columns=['ts', 'open', 'high', 'low', 'close'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
    df = df.sort_values('ts').reset_index(drop=True)
    return df


def realized_vol(log_ret, window, ann):
    """Rolling realized volatility, annualized via the `ann` factor."""
    return log_ret.rolling(window).std() * ann


def ewma_vol(log_ret, ann, lam=0.94):
    """RiskMetrics EWMA volatility, annualized.

    Returns (series, latest). lam=0.94 is the standard RiskMetrics decay.
    """
    r = pd.Series(log_ret).dropna().values
    if len(r) == 0:
        return pd.Series(dtype=float), float('nan')
    var = np.empty(len(r))
    var[0] = r[0] ** 2
    for i in range(1, len(r)):
        var[i] = lam * var[i - 1] + (1 - lam) * r[i - 1] ** 2
    vol = np.sqrt(var) * ann
    return pd.Series(vol), float(vol[-1])


def _garch11_negloglik(params, r):
    omega, alpha, beta = params
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
        return 1e10
    n = len(r)
    sigma2 = np.empty(n)
    sigma2[0] = np.var(r)
    for t in range(1, n):
        sigma2[t] = omega + alpha * r[t - 1] ** 2 + beta * sigma2[t - 1]
    sigma2 = np.maximum(sigma2, 1e-12)
    return 0.5 * np.sum(np.log(sigma2) + r ** 2 / sigma2)


def garch11_forecast(log_ret, horizon, ann):
    """Fit GARCH(1,1) by ML (scipy) and forecast annualized vol over `horizon` steps.

    Returns the annualized forecast (from the mean variance over the horizon),
    or None if scipy/optimization is unavailable or the fit is degenerate.
    """
    try:
        from scipy.optimize import minimize
    except ImportError:
        return None

    r = pd.Series(log_ret).dropna().values
    if len(r) < 50:
        return None
    r = r - r.mean()
    var0 = np.var(r)

    x0 = [var0 * 0.05, 0.05, 0.90]
    bounds = [(1e-12, None), (0.0, 1.0), (0.0, 1.0)]
    try:
        res = minimize(_garch11_negloglik, x0, args=(r,), method='L-BFGS-B', bounds=bounds)
    except Exception:
        return None
    if not res.success:
        return None

    omega, alpha, beta = res.x
    if alpha + beta >= 1:
        return None

    n = len(r)
    sigma2 = np.empty(n)
    sigma2[0] = var0
    for t in range(1, n):
        sigma2[t] = omega + alpha * r[t - 1] ** 2 + beta * sigma2[t - 1]

    s2 = sigma2[-1]
    fc = np.empty(horizon)
    for h in range(horizon):
        s2 = omega + (alpha + beta) * s2
        fc[h] = s2
    mean_var = fc.mean()
    return float(np.sqrt(mean_var) * ann)


def regime_label(current_iv, iv_history):
    """Label the current implied-vol regime by its percentile in recent history."""
    hist = pd.Series(iv_history).dropna()
    if len(hist) < 5:
        return 'UNKNOWN', float('nan')
    pct = float((hist < current_iv).mean() * 100)
    if pct < 25:
        label = 'LOW'
    elif pct < 50:
        label = 'NORMAL'
    elif pct < 75:
        label = 'ELEVATED'
    else:
        label = 'HIGH'
    return label, pct


def forecast_volatility(log_ret, horizon_steps, ann, rv_window=24,
                        iv_df=None, iv_label='IV'):
    """Build the volatility forecast bundle for the report.

    Combines a rolling RV snapshot, an EWMA estimate, a GARCH(1,1) horizon
    forecast (falling back to EWMA), and the implied-vol context. `ann` is the
    annualization factor for the return frequency; `iv_df` has a 'close' column
    of implied vol in annualized %.

    Returns a dict of annualized-percent values plus the IV-RV spread and regime.
    """
    log_ret = pd.Series(log_ret).dropna()

    rv_series = realized_vol(log_ret, rv_window, ann) * 100
    current_rv = float(rv_series.dropna().iloc[-1]) if rv_series.notna().any() else float('nan')

    _, ewma_latest = ewma_vol(log_ret, ann, lam=0.94)
    current_ewma = ewma_latest * 100

    garch_fc = garch11_forecast(log_ret, horizon=max(1, horizon_steps), ann=ann)
    if garch_fc is not None:
        forecast_rv = garch_fc * 100
        forecast_method = 'GARCH(1,1)'
    else:
        forecast_rv = current_ewma
        forecast_method = 'EWMA (GARCH unavailable)'

    out = {
        'current_rv': current_rv,
        'current_ewma': current_ewma,
        'forecast_rv': forecast_rv,
        'forecast_method': forecast_method,
        'current_iv': None,
        'iv_label': iv_label,
        'iv_rv_spread': None,
        'regime': None,
        'regime_pct': None,
        'iv_available': False,
    }

    if iv_df is not None and not iv_df.empty:
        current_iv = float(iv_df['close'].iloc[-1])
        label, pct = regime_label(current_iv, iv_df['close'])
        out.update({
            'current_iv': current_iv,
            'iv_rv_spread': current_iv - forecast_rv,
            'regime': label,
            'regime_pct': pct,
            'iv_available': True,
        })

    return out


def interpret_spread(spread):
    """One-line interpretation of the IV-RV variance-risk premium."""
    if spread is None:
        return "Implied vol unavailable; no implied-vs-realized comparison."
    if spread > 5:
        return ("Implied >> realized: positive variance-risk premium "
                "(options richly priced; vol-sellers favored).")
    if spread < -5:
        return ("Realized >> implied: negative variance-risk premium "
                "(options cheap relative to recent moves; vol-buyers favored).")
    return "Implied ~ realized: variance-risk premium near neutral."


def write_report(path, info, generated_utc, asset_name, horizon_steps,
                 horizon_unit, current_price=None):
    """Write the volatility report to `path` (text), mirroring the price report style."""
    lines = []
    lines.append("=" * 70)
    lines.append(f"{asset_name} VOLATILITY FORECAST REPORT - {info['iv_label']} + EWMA/GARCH")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Generated: {generated_utc}")
    if current_price is not None:
        lines.append(f"Current Price: ${current_price:.2f}")
    lines.append(f"Horizon: {horizon_steps} {horizon_unit}")
    lines.append("")

    lines.append(f"IMPLIED VOLATILITY ({info['iv_label']}):")
    if info['iv_available']:
        lines.append(f"  Current implied vol (annualized): {info['current_iv']:.2f}%")
        if info['regime'] != 'UNKNOWN':
            lines.append(f"  Vol regime:                       {info['regime']} "
                         f"({info['regime_pct']:.0f}th pct of recent history)")
        else:
            lines.append("  Vol regime:                       UNKNOWN (no IV history)")
    else:
        lines.append("  Implied-vol fetch FAILED - degraded to realized-vol-only report.")
    lines.append("")

    lines.append("REALIZED VOLATILITY (annualized):")
    lines.append(f"  Current rolling RV:           {info['current_rv']:.2f}%")
    lines.append(f"  Current EWMA (RiskMetrics):   {info['current_ewma']:.2f}%")
    lines.append(f"  Horizon forecast [{info['forecast_method']}]: {info['forecast_rv']:.2f}%")
    lines.append("")

    lines.append("IV - RV SPREAD (variance-risk premium):")
    if info['iv_rv_spread'] is not None:
        lines.append(f"  Implied - forecast RV = {info['iv_rv_spread']:+.2f} vol points")
    lines.append(f"  {interpret_spread(info['iv_rv_spread'])}")
    lines.append("")

    with open(path, 'w') as f:
        f.write("\n".join(lines) + "\n")
    return "\n".join(lines)
