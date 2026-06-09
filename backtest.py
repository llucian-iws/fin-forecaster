#!/usr/bin/env python3
"""Walk-forward backtest for the BTC forecasting stack.

Evaluates the forecast on truly out-of-sample folds (expanding window,
h-step-ahead; training targets are fully realized at/before the fold's "now",
so nothing leaks across the line). Folds are stepped by the horizon, so the
evaluated folds are non-overlapping.

Metrics
  - directional hit-rate          (vs a 50% coin-flip baseline)
  - MAE / MAPE of the point price  (vs a random-walk persistence baseline)
  - 90% interval coverage for three shock-vol variants that share the SAME
    point forecast, so the comparison isolates band width:
        realized   - trailing vol_24h            (the pre-integration band)
        garch      - GARCH(1,1) forward forecast  (the model half)
        garch+dvol - GARCH forward blended w/ Deribit DVOL (the full integration)

Engines (pluggable via --engine)
  lite    - GradientBoosting, fast, many folds (default)
  cnnlstm - the full Conv1D/LSTM stack, slow, few folds (run separately)
"""

import warnings
warnings.filterwarnings('ignore')

import argparse
import os
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import GradientBoostingRegressor

import volatility
import forecast_post as fp
import metrics as mx
from exogenous import fetch_funding, funding_features

OUTPUT_DIR = Path(os.getenv('OUTPUT_DIR', '/app/results'))
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

Z90 = 1.6448536269514722          # two-sided 90% (0.05 per tail)
ANN_HOURLY = np.sqrt(24 * 365)    # hourly -> annualized vol factor
N_MC = 4000                       # MC draws per fold for CRPS / quantile bands
BASE_FEATURES = ['log_ret', 'vol', 'rsi', 'ema_stack']
FUND_FEATURES = BASE_FEATURES + ['funding_z', 'funding_cum_24h']
FEATURES = BASE_FEATURES          # cnn engine uses the base TA features


def build_features(df):
    """Core feature set (matches btc_forecast_lite.py). No look-ahead: every
    column at row i uses only data up to row i."""
    df = df.copy()
    df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['vol'] = df['log_ret'].rolling(24).std()
    df['rsi'] = 100 - (100 / (1 + (df['log_ret'].rolling(14).mean() /
                                   df['log_ret'].rolling(14).std())))
    for span in [7, 21, 50]:
        df[f'ema{span}'] = df['Close'].ewm(span=span).mean()
    df['ema_stack'] = ((df['Close'] > df['ema7']).astype(int) +
                       (df['Close'] > df['ema21']).astype(int) +
                       (df['Close'] > df['ema50']).astype(int))
    return df.dropna()


def lite_rate(feat_train, y_train, feat_now):
    """Fit GB on the past and predict the mean hourly log-return at the fold."""
    scaler = RobustScaler().fit(feat_train)
    model = GradientBoostingRegressor(
        n_estimators=50, learning_rate=0.05, max_depth=4, random_state=42)
    model.fit(scaler.transform(feat_train), y_train)
    return float(model.predict(scaler.transform(feat_now.reshape(1, -1)))[0])


def cnn_rate(feat_all, tgt_all, t, h, window, epochs):
    """Train a compact Conv1D/LSTM and predict the rate at fold time t.

    Training sequences use only targets realized by t (rows < t-h+1); the scaler
    is fit on that training region only. Inference uses the window ENDING at t
    (features up to t are known at the fold), matching the lite engine."""
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    tf.random.set_seed(42)
    np.random.seed(42)
    train_end = t - h + 1                          # exclusive; targets realized by t
    scaler = RobustScaler().fit(feat_all[:train_end])
    fs = scaler.transform(feat_all[:t + 1])        # features up to fold time t

    # Supervised sequences: rows [i-window : i] -> mean-hourly-return target at i.
    X, Y = [], []
    for i in range(window, train_end):
        X.append(fs[i - window:i])
        Y.append(tgt_all[i])
    X, Y = np.asarray(X), np.asarray(Y)

    model = keras.Sequential([
        layers.Input((window, feat_all.shape[1])),
        layers.Conv1D(32, 3, activation='relu', padding='causal'),
        layers.Conv1D(32, 3, activation='relu', padding='causal'),
        layers.LSTM(32),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(1),
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, Y, epochs=epochs, batch_size=64, verbose=0)

    infer_seq = fs[t - window + 1:t + 1][None, :, :]
    return float(model.predict(infer_seq, verbose=0)[0, 0])


def load_dvol():
    """Fetch a long Deribit DVOL history, indexed by tz-aware UTC timestamp.
    Returns a Series of annualized-% implied vol, or None.

    Uses 12h resolution: Deribit caps a response at ~1000 rows, so hourly only
    spans ~41 days. The DVOL index barely moves intraday and the backtest only
    needs the latest value <= each fold, so 12h gives ~500 days of coverage."""
    try:
        d = volatility.fetch_dvol(currency='BTC', resolution='43200', days=500)
        if d is None or d.empty:
            return None
        # fetch_dvol already returns ts as tz-aware UTC datetimes.
        return pd.Series(d['close'].values, index=d['ts']).sort_index()
    except Exception as exc:
        print(f"  [dvol] history unavailable: {exc}")
        return None


def main():
    ap = argparse.ArgumentParser(description='Walk-forward backtest')
    ap.add_argument('--engine', choices=['lite', 'cnnlstm'], default='lite')
    ap.add_argument('--horizon', type=int, default=24, help='forecast horizon (hours)')
    ap.add_argument('--step', type=int, default=None,
                    help='hours between folds (default = horizon, i.e. non-overlapping daily)')
    ap.add_argument('--max-folds', type=int, default=150)
    ap.add_argument('--min-train', type=int, default=2000, help='min rows before first fold')
    ap.add_argument('--window', type=int, default=168, help='cnnlstm sequence length')
    ap.add_argument('--epochs', type=int, default=3, help='cnnlstm epochs/fold')
    args = ap.parse_args()
    h = args.horizon
    step = args.step or h

    print("=" * 80)
    print(f"WALK-FORWARD BACKTEST  | engine={args.engine}  horizon={h}h")
    print("=" * 80)

    print("\n[1/3] Fetching BTC data + DVOL history...")
    end = datetime.datetime.now()
    start = end - datetime.timedelta(days=700)
    raw = yf.download('BTC-USD', start=start.strftime('%Y-%m-%d'),
                      end=end.strftime('%Y-%m-%d'), interval='1h', progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(-1)
    raw = raw[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    df = build_features(raw)
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    n = len(df)
    print(f"  {n} feature rows ({df.index[0].date()} -> {df.index[-1].date()})")

    dvol = load_dvol()
    if dvol is not None:
        print(f"  DVOL history: {len(dvol)} pts ({dvol.index[0].date()} -> {dvol.index[-1].date()})")
    else:
        print("  DVOL history: unavailable (garch+dvol variant falls back to garch)")

    # --- Exogenous: funding-rate features (look-ahead-free, ffilled to hourly)
    print("  Fetching Binance funding-rate history...")
    fund_df = fetch_funding('BTCUSDT', days=700)
    ff = funding_features(fund_df, df.index).fillna(0.0)
    for c in ('funding_z', 'funding_cum_24h', 'funding_rate'):
        df[c] = ff[c].values
    funding_ok = fund_df is not None and len(fund_df) > 0
    print("  Funding: " + (f"OK, {len(fund_df)} prints "
          f"({fund_df['ts'].iloc[0].date()} -> {fund_df['ts'].iloc[-1].date()})"
          if funding_ok else "unavailable -> zero-filled (no exogenous signal)"))

    close = df['Close'].values
    logc = np.log(close)
    log_ret = df['log_ret'].values
    realized_h = df['vol'].values                 # trailing vol_24h (hourly std)
    base_mat = df[BASE_FEATURES].values
    fund_mat = df[FUND_FEATURES].values
    # Target: mean hourly log-return over the next h hours.
    tgt = np.full(n, np.nan)
    tgt[:n - h] = (logc[h:] - logc[:n - h]) / h

    # Fold "now" indices t: predict close[t+h] from info up to t.
    t_end = n - 1 - h
    t_start = max(args.min_train, t_end - (args.max_folds - 1) * step)
    folds = list(range(t_start, t_end + 1, step))
    ann_factor = (24 * 365) / step                # folds per year, for Sharpe
    print(f"\n[2/3] Running {len(folds)} folds "
          f"({df.index[folds[0]].date()} -> {df.index[folds[-1]].date()}, step {step}h)...")

    rng = np.random.default_rng(12345)
    rows = []
    dvol_hits = 0
    # Strictly-prior accumulators for OUT-OF-SAMPLE post-processing (no leak).
    hist_rate, hist_realized, hist_err = [], [], []   # base rate vs realized tgt
    pt_err = {'base': [], 'fund': []}                 # point abs-errors -> ensemble weights
    lite = args.engine == 'lite'
    for fi, t in enumerate(folds):
        last = t - h + 1                              # train on rows whose target is realized by t
        if last < args.min_train:
            continue
        ytr = tgt[:last]

        # --- model variants ---------------------------------------------
        if lite:
            rate_base = lite_rate(base_mat[:last], ytr, base_mat[t])
            rate_fund = lite_rate(fund_mat[:last], ytr, fund_mat[t])
            # vol-standardized target (#6): regress tgt/vol, rescale by current vol
            vstd = ytr / np.where(realized_h[:last] > 0, realized_h[:last], np.nan)
            m2 = ~np.isnan(vstd)
            rate_vstd = (lite_rate(base_mat[:last][m2], vstd[m2], base_mat[t]) * realized_h[t]
                         if int(m2.sum()) >= args.min_train else rate_base)
        else:
            rate_base = cnn_rate(base_mat, tgt, t, h, args.window, args.epochs)
            rate_fund = rate_vstd = rate_base        # cnn engine stays on base features

        # ensemble of base+funding, weighted by trailing OOS point-MAE (#7)
        mae_b = float(np.mean(pt_err['base'])) if pt_err['base'] else None
        mae_f = float(np.mean(pt_err['fund'])) if pt_err['fund'] else None
        rate_ens = fp.ensemble_combine({'base': rate_base, 'fund': rate_fund},
                                       {'base': mae_b, 'fund': mae_f})

        # shrink drift toward random walk + subtract trailing bias, both OOS (#1)
        alpha = fp.fit_shrinkage(hist_rate, hist_realized)
        bias_r = fp.rolling_bias_correction(hist_err, window=30)
        rate_adj = fp.apply_shrinkage(rate_base, alpha) - bias_r

        cur, actual = close[t], close[t + h]
        rates = {'base': rate_base, 'fund': rate_fund, 'volstd': rate_vstd,
                 'ens': rate_ens, 'adj': rate_adj, 'persist': 0.0}
        rec = {'t': df.index[t], 'cur': cur, 'actual': actual,
               'alpha': alpha, 'bias_r': bias_r, 'vol_h': realized_h[t] * np.sqrt(h)}
        for k, r in rates.items():
            rec[f'pt_{k}'] = cur * np.exp(r * h)

        # --- shock-vol variants -> calibrated distribution around the adj point
        sig_real = realized_h[t]
        gf = volatility.garch11_forecast(pd.Series(log_ret[:t + 1]), horizon=h, ann=ANN_HOURLY)
        sig_garch = (gf / ANN_HOURLY) if gf is not None else sig_real
        dv = None
        if dvol is not None:
            past = dvol[dvol.index <= df.index[t]]
            if len(past):
                dv = float(past.iloc[-1])
        if dv is not None:
            dvol_hits += 1
            sig_dvol = float(np.mean([sig_garch, (dv / 100.0) / ANN_HOURLY]))
        else:
            sig_dvol = sig_garch

        p_adj = rec['pt_adj']
        for tag, sig in (('real', sig_real), ('garch', sig_garch), ('dvol', sig_dvol)):
            sd = max(float(sig), 1e-9) * np.sqrt(h)
            draws = p_adj * np.exp(rng.normal(0.0, sd, N_MC))
            band = fp.empirical_quantile_band(draws, (0.05, 0.25, 0.5, 0.75, 0.95))
            rec[f'cov_{tag}'] = mx.interval_coverage(band[0.05], band[0.95], actual)
            rec[f'crps_{tag}'] = mx.crps_ensemble(draws, actual)
            rec[f'pin_{tag}'] = mx.pinball_loss(band, actual)
            rec[f'wid_{tag}'] = (band[0.95] - band[0.05]) / p_adj
        rows.append(rec)

        # advance strictly-prior accumulators AFTER use
        hist_rate.append(rate_base); hist_realized.append(tgt[t]); hist_err.append(rate_base - tgt[t])
        pt_err['base'].append(abs(rec['pt_base'] - actual))
        pt_err['fund'].append(abs(rec['pt_fund'] - actual))
        if (fi + 1) % 25 == 0:
            print(f"    {fi + 1}/{len(folds)} folds...")

    R = pd.DataFrame(rows)
    print(f"  Completed {len(R)} folds.")

    # ---- Metrics --------------------------------------------------------
    cur = R['cur'].values
    actual = R['actual'].values
    rw_mae = float(np.abs(cur - actual).mean())
    rw_mape = float(np.abs(cur / actual - 1).mean())

    def pstats(tag):
        p = R[f'pt_{tag}'].values
        hit = float((np.sign(p - cur) == np.sign(actual - cur)).mean())
        return hit, float(np.abs(p - actual).mean()), float(np.abs(p / actual - 1).mean())

    drift_adj = np.log(R['pt_adj'].values / cur)          # predicted total log-ret over horizon
    realized_ret = np.log(actual / cur)
    strat = mx.strategy_eval(drift_adj, realized_ret, R['vol_h'].values,
                             fee_bps=5.0, ann_factor=ann_factor)

    models = [('base', 'base (TA only)    '), ('adj', 'shrunk+debiased   ')]
    if lite:
        models = [('base', 'base (TA only)    '), ('fund', '+ funding         '),
                  ('volstd', 'vol-standardized  '), ('ens', 'ensemble(base+fnd)'),
                  ('adj', 'shrunk+debiased   ')]

    print("\n[3/3] Results")
    L = []
    L.append("=" * 74)
    L.append(f"WALK-FORWARD BACKTEST  (engine={args.engine}, horizon={h}h)")
    L.append("=" * 74)
    L.append(f"Generated:  {datetime.datetime.utcnow():%Y-%m-%d %H:%M:%S} UTC")
    L.append(f"Folds:      {len(R)} non-overlapping "
             f"({R['t'].iloc[0]:%Y-%m-%d} -> {R['t'].iloc[-1]:%Y-%m-%d}, step {step}h)")
    L.append(f"Funding:    {'on' if funding_ok else 'unavailable (zero-filled)'}")
    L.append("")
    L.append("POINT FORECAST            hit%      MAE         MAPE")
    for tag, label in models:
        hit, mae, mape = pstats(tag)
        L.append(f"  {label}    {hit*100:5.1f}   ${mae:>9,.2f}   {mape*100:5.2f}%")
    L.append(f"  persistence (RW)           --   ${rw_mae:>9,.2f}   {rw_mape*100:5.2f}%")
    adj_mae = pstats('adj')[1]
    L.append(f"  -> shrunk+debiased beats persistence MAE: "
             f"{'YES' if adj_mae < rw_mae else 'NO'}  (final alpha~{R['alpha'].iloc[-1]:.2f})")
    L.append("")
    L.append("DISTRIBUTION around shrunk point   coverage    CRPS      pinball   width")
    best_crps, best_tag = None, None
    for tag, label in (('real', 'realized vol_24h'), ('garch', 'GARCH forward  '),
                       ('dvol', 'GARCH+DVOL     ')):
        cov = float(R[f'cov_{tag}'].mean())
        crps = float(R[f'crps_{tag}'].mean())
        pin = float(R[f'pin_{tag}'].mean())
        wid = float(R[f'wid_{tag}'].mean())
        L.append(f"  {label}    {cov:6.3f}   ${crps:>8,.2f}   ${pin:>7,.2f}   +/-{wid/2*100:4.1f}%")
        if best_crps is None or crps < best_crps:
            best_crps, best_tag = crps, label.strip()
    L.append(f"  -> best CRPS (calibration+sharpness): {best_tag}")
    if dvol is not None:
        L.append(f"  (DVOL present for {dvol_hits}/{len(R)} folds)")
    L.append("")
    L.append("ECONOMIC  (vol-targeted long/short on shrunk drift, 5bps fee)")
    L.append(f"  Sharpe (annualized):   {strat['sharpe']:+.2f}")
    L.append(f"  Total return (log):    {strat['total_return']*100:+.1f}%")
    L.append(f"  Max drawdown:          {strat['max_drawdown']*100:.1f}%")
    L.append(f"  Trade hit-rate:        {strat['hit_rate']*100:.1f}%   "
             f"avg|pos| {strat['avg_abs_position']:.2f}")
    L.append("=" * 74)
    report = "\n".join(L)
    print("\n" + report)

    (OUTPUT_DIR / 'backtest_report.txt').write_text(report + "\n")
    R.to_csv(OUTPUT_DIR / 'backtest_folds.csv', index=False)
    print(f"\n  Saved: {OUTPUT_DIR/'backtest_report.txt'}")
    print(f"  Saved: {OUTPUT_DIR/'backtest_folds.csv'}")


if __name__ == '__main__':
    main()
