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

Forward-vol models: GARCH(1,1) (garch11_forecast) and HAR-RV (har_rv_forecast,
Corsi 2009) share the same signature and None-fallback contract; gk_variance /
bipower_variance supply range-based / jump-robust per-bar realized measures
that can feed HAR via its `rv` argument. Run `python3 volatility.py` for a
no-network smoke test of the pure functions.
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


def gk_variance(open_, high, low, close):
    """Per-bar Garman-Klass variance estimate from OHLC.

        v = 0.5 * ln(H/L)^2 - (2*ln2 - 1) * ln(C/O)^2

    Range-based estimators are several times more efficient vol proxies than
    squared close-to-close returns, and 24/7 crypto has no overnight gap, so
    this is a near-free upgrade to the realized-variance input of HAR-RV.

    Bars that fail an OHLC sanity check (non-positive prices, H < L,
    H < max(O, C), L > min(O, C), or non-finite values) come back as NaN; the
    result is floored at 0 (the GK formula can go slightly negative on
    doji-like bars). Returns a float ndarray aligned with the inputs.
    """
    o = np.asarray(open_, dtype=float).ravel()
    h = np.asarray(high, dtype=float).ravel()
    l = np.asarray(low, dtype=float).ravel()
    c = np.asarray(close, dtype=float).ravel()
    valid = (np.isfinite(o) & np.isfinite(h) & np.isfinite(l) & np.isfinite(c)
             & (o > 0) & (h > 0) & (l > 0) & (c > 0)
             & (h >= l) & (h >= np.maximum(o, c)) & (l <= np.minimum(o, c)))
    out = np.full(o.shape, np.nan)
    hl = np.log(h[valid] / l[valid])
    co = np.log(c[valid] / o[valid])
    out[valid] = np.maximum(0.5 * hl ** 2 - (2.0 * np.log(2.0) - 1.0) * co ** 2, 0.0)
    return out


def bipower_variance(log_ret):
    """Jump-robust per-bar variance proxy: (pi/2) * |r_t| * |r_{t-1}|.

    The bipower product converges to integrated variance WITHOUT the jump
    contribution, so feeding it to HAR-RV de-emphasizes liquidation-cascade
    bars relative to squared returns. First element (no prior return) is NaN.
    """
    r = np.asarray(log_ret, dtype=float).ravel()
    out = np.full(r.shape, np.nan)
    if r.size >= 2:
        out[1:] = (np.pi / 2.0) * np.abs(r[1:]) * np.abs(r[:-1])
    return out


def har_rv_forecast(log_ret, horizon, ann, rv=None, bars_per_day=24):
    """HAR-RV (Corsi 2009) realized-variance forecast, annualized vol output.

    Same contract as garch11_forecast: takes the hourly log-return series, a
    `horizon` in bars, the bar->annual factor `ann`, and returns the
    annualized vol forecast (sqrt of the mean forecast variance over the
    horizon) or None when the fit is unavailable or degenerate — so callers
    can fall back to GARCH.

    Daily realized variance is the sum of per-bar contributions over
    consecutive `bars_per_day`-bar blocks aligned to END at the latest
    observation (a block == one calendar day for a gap-free 24/7 asset). The
    per-bar contribution defaults to the squared return; pass `rv` (e.g.
    gk_variance or bipower_variance output, aligned with log_ret) to use a
    range-based / jump-robust measure instead. Days with NaN bars are rescaled
    from their valid bars (>= 75% required, else the previous day's RV is
    carried; >10% such days aborts to None).

    The forecast regresses RV_{d+1} on [1, RV_d, mean(RV_{d-4..d}),
    mean(RV_{d-21..d})] by OLS and iterates the recursion over
    ceil(horizon / bars_per_day) days, feeding forecasts back into the lags.
    """
    if rv is not None:
        x = np.asarray(rv, dtype=float).ravel()
    else:
        x = pd.Series(log_ret, dtype=float).values ** 2
    n_days = x.size // bars_per_day
    if n_days < 45:                       # 22-day lag + a usable fit window
        return None
    x = x[x.size - n_days * bars_per_day:]            # blocks end at the latest bar
    blocks = x.reshape(n_days, bars_per_day)

    n_valid = np.isfinite(blocks).sum(axis=1)
    with np.errstate(invalid='ignore'):
        rv_d = np.nansum(blocks, axis=1) * (bars_per_day / np.maximum(n_valid, 1))
    bad = n_valid < int(0.75 * bars_per_day)
    if bad.mean() > 0.10 or bad[0]:
        return None
    for i in np.flatnonzero(bad):
        rv_d[i] = rv_d[i - 1]                          # carry previous day's RV

    d_lag = rv_d[21:-1]
    w_lag = np.array([rv_d[i - 4:i + 1].mean() for i in range(21, n_days - 1)])
    m_lag = np.array([rv_d[i - 21:i + 1].mean() for i in range(21, n_days - 1)])
    y = rv_d[22:]
    X = np.column_stack([np.ones_like(d_lag), d_lag, w_lag, m_lag])
    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(beta)):
        return None

    hist = list(rv_d)
    n_fc = max(1, int(np.ceil(horizon / bars_per_day)))
    fcs = []
    for _ in range(n_fc):
        feats = np.array([1.0, hist[-1], np.mean(hist[-5:]), np.mean(hist[-22:])])
        f = float(beta @ feats)
        if not np.isfinite(f) or f <= 0.0:
            return None
        fcs.append(f)
        hist.append(f)
    hourly_var = np.mean(fcs) / bars_per_day
    return float(np.sqrt(hourly_var) * ann)


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


if __name__ == '__main__':
    # No-network smoke test of the pure realized-measure / HAR functions.
    rng = np.random.default_rng(7)
    ANN = np.sqrt(24 * 365)

    # (a) HAR-RV on iid returns recovers the true annualized vol.
    sigma = 0.01
    r = rng.normal(0.0, sigma, size=120 * 24)
    fc = har_rv_forecast(r, horizon=24, ann=ANN)
    truth = sigma * ANN
    print(f"HAR-RV iid:    forecast={fc:.4f}  truth={truth:.4f}  "
          f"ratio={fc/truth:.3f}")
    assert fc is not None and 0.8 < fc / truth < 1.2, fc

    # (b) After a vol-regime shift, HAR tracks the RECENT level.
    r2 = np.concatenate([rng.normal(0, 0.01, 90 * 24), rng.normal(0, 0.03, 40 * 24)])
    fc2 = har_rv_forecast(r2, horizon=24, ann=ANN)
    print(f"HAR-RV shift:  forecast={fc2:.4f}  recent truth={0.03*ANN:.4f}")
    assert fc2 is not None and fc2 > 0.02 * ANN, fc2

    # (c) None-fallback contract on short input.
    assert har_rv_forecast(r[:30 * 24], horizon=24, ann=ANN) is None

    # (d) gk_variance: exact value when C == O, NaN on insane bars, floor at 0.
    n = 10
    o = np.full(n, 100.0)
    c = o.copy()
    hgh = o * np.exp(0.02)
    low = o * np.exp(-0.02)
    gk = gk_variance(o, hgh, low, c)
    expected = 0.5 * (0.04) ** 2
    assert np.allclose(gk, expected), gk[:3]
    bad = gk_variance([100, -5, 100], [99, 110, 110], [98, 90, 90], [98.5, 100, 100])
    assert np.isnan(bad[0]) and np.isnan(bad[1]) and np.isfinite(bad[2])
    print(f"gk_variance:   {gk[0]:.6f} == 0.5*ln(H/L)^2 = {expected:.6f}; "
          f"invalid bars -> NaN OK")

    # (e) bipower converges to sigma^2 on iid normals (E = (pi/2)(E|r|)^2).
    rb = rng.normal(0.0, 0.01, size=50_000)
    bp = bipower_variance(rb)
    ratio = np.nanmean(bp) / 0.01 ** 2
    print(f"bipower:       mean/sigma^2 = {ratio:.3f} (expect ~1)")
    assert 0.9 < ratio < 1.1, ratio
    assert np.isnan(bp[0])

    # (f) HAR fed by a GK-style per-bar rv series (constant -> exact recovery).
    rv_const = np.full(120 * 24, 1e-4)
    fc3 = har_rv_forecast(None, horizon=24, ann=ANN, rv=rv_const)
    print(f"HAR-RV on rv:  forecast={fc3:.4f}  truth={0.01*ANN:.4f}")
    assert fc3 is not None and abs(fc3 / (0.01 * ANN) - 1) < 0.01, fc3

    print("\nvolatility smoke test: all assertions passed")
