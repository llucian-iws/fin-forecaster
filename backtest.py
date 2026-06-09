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

OUTPUT_DIR = Path(os.getenv('OUTPUT_DIR', '/app/results'))
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

Z90 = 1.6448536269514722          # two-sided 90% (0.05 per tail)
ANN_HOURLY = np.sqrt(24 * 365)    # hourly -> annualized vol factor
FEATURES = ['log_ret', 'vol', 'rsi', 'ema_stack']


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

    close = df['Close'].values
    logc = np.log(close)
    log_ret = df['log_ret'].values
    realized_h = df['vol'].values          # trailing vol_24h (hourly std)
    feat = df[FEATURES].values
    # Target: mean hourly log-return over the next h hours.
    tgt = np.full(n, np.nan)
    tgt[:n - h] = (logc[h:] - logc[:n - h]) / h

    # Fold "now" indices t: predict close[t+h] from info up to t.
    t_end = n - 1 - h
    t_start = max(args.min_train, t_end - (args.max_folds - 1) * step)
    folds = list(range(t_start, t_end + 1, step))
    print(f"\n[2/3] Running {len(folds)} folds "
          f"({df.index[folds[0]].date()} -> {df.index[folds[-1]].date()}, step {step}h)...")

    rows = []
    dvol_hits = 0
    for fi, t in enumerate(folds):
        # Train only on rows whose full h-ahead target is realized by t.
        last = t - h + 1                       # exclusive end for training
        ft, yt = feat[:last], tgt[:last]
        mask = ~np.isnan(yt)
        ft, yt = ft[mask], yt[mask]
        if len(yt) < args.min_train:
            continue

        if args.engine == 'lite':
            rate = lite_rate(ft, yt, feat[t])
        else:
            rate = cnn_rate(feat, tgt, t, h, args.window, args.epochs)

        point = close[t] * np.exp(rate * h)
        actual = close[t + h]

        # Three shock-vol variants (hourly std), shared point forecast.
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
            sig_dvol = np.mean([sig_garch, (dv / 100.0) / ANN_HOURLY])
        else:
            sig_dvol = sig_garch

        rec = {'t': df.index[t], 'cur': close[t], 'point': point, 'actual': actual,
               'rate': rate}
        for tag, sig in [('real', sig_real), ('garch', sig_garch), ('dvol', sig_dvol)]:
            band = Z90 * sig * np.sqrt(h)
            lo, hi = point * np.exp(-band), point * np.exp(band)
            rec[f'cov_{tag}'] = int(lo <= actual <= hi)
            rec[f'wid_{tag}'] = (hi - lo) / point
        rows.append(rec)
        if (fi + 1) % 25 == 0:
            print(f"    {fi + 1}/{len(folds)} folds...")

    R = pd.DataFrame(rows)
    print(f"  Completed {len(R)} folds.")

    # ---- Metrics --------------------------------------------------------
    cur, point, actual = R['cur'].values, R['point'].values, R['actual'].values
    dir_correct = np.sign(point - cur) == np.sign(actual - cur)
    hit_rate = dir_correct.mean()
    mae, mape = np.abs(point - actual).mean(), np.abs(point / actual - 1).mean()
    bias = (point / actual - 1).mean()
    rw_mae, rw_mape = np.abs(cur - actual).mean(), np.abs(cur / actual - 1).mean()

    print("\n[3/3] Results")
    lines = []
    lines.append("=" * 70)
    lines.append(f"WALK-FORWARD BACKTEST  (engine={args.engine}, horizon={h}h)")
    lines.append("=" * 70)
    lines.append(f"Generated:  {datetime.datetime.utcnow():%Y-%m-%d %H:%M:%S} UTC")
    lines.append(f"Folds:      {len(R)} non-overlapping "
                 f"({R['t'].iloc[0]:%Y-%m-%d} -> {R['t'].iloc[-1]:%Y-%m-%d})")
    lines.append("")
    lines.append("POINT FORECAST")
    lines.append(f"  Directional hit-rate:  {hit_rate*100:5.1f}%   (coin-flip 50.0%)")
    lines.append(f"  MAE:                   ${mae:>10,.2f}   (persistence ${rw_mae:,.2f})")
    lines.append(f"  MAPE:                  {mape*100:5.2f}%   (persistence {rw_mape*100:.2f}%)")
    lines.append(f"  Bias (mean % error):   {bias*100:+5.2f}%")
    beats = "YES" if mae < rw_mae else "NO"
    lines.append(f"  Beats persistence MAE: {beats}")
    lines.append("")
    lines.append("INTERVAL COVERAGE @ 90%  (nominal 0.90; closer = better calibrated)")
    for tag, label in [('real', 'realized vol_24h '), ('garch', 'GARCH forward    '),
                       ('dvol', 'GARCH+DVOL       ')]:
        cov = R[f'cov_{tag}'].mean()
        wid = R[f'wid_{tag}'].mean()
        lines.append(f"  {label}: {cov:0.3f}   mean width +/-{wid/2*100:4.1f}%")
    if dvol is not None:
        lines.append(f"  (DVOL available for {dvol_hits}/{len(R)} folds; "
                     f"rest fall back to GARCH)")
    lines.append("=" * 70)
    report = "\n".join(lines)
    print("\n" + report)

    (OUTPUT_DIR / 'backtest_report.txt').write_text(report + "\n")
    R.to_csv(OUTPUT_DIR / 'backtest_folds.csv', index=False)
    print(f"\n  Saved: {OUTPUT_DIR/'backtest_report.txt'}")
    print(f"  Saved: {OUTPUT_DIR/'backtest_folds.csv'}")


if __name__ == '__main__':
    main()
