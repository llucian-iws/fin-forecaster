#!/usr/bin/env python3
"""
Bitcoin Price Forecaster - FULL QUANTITATIVE STACK
Predicts BTC price for next Sunday 12:00 AM UTC using CNN-LSTM + HMM + Monte Carlo
Adapted from SNOW.ipynb with sophisticated deep learning methods
"""

import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import norm
from hmmlearn.hmm import GaussianHMM
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import datetime
import pytz
from pathlib import Path

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 80)
print("BITCOIN PRICE FORECASTER | Full Quantitative Stack (CNN-LSTM + HMM)")
print("=" * 80)

LOOKBACK_HOURS = 168
HORIZON_HOURS = 24
MODEL_EPOCHS = 5
VERBOSE = 1

# =====================================================================
# SECTION 1: Data Fetching & Feature Engineering
# =====================================================================
print("\n[1/6] Fetching Bitcoin data...")

end_date = datetime.datetime.now()
start_date = end_date - datetime.timedelta(days=700)
date_str_start = start_date.strftime('%Y-%m-%d')
date_str_end = end_date.strftime('%Y-%m-%d')

df = yf.download('BTC-USD', start=date_str_start, end=date_str_end, interval='1h', progress=False)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.droplevel(-1)
df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy().dropna()

current_price_val = float(df['Close'].values[-1])
print(f"  Downloaded {len(df)} hourly candles")
print(f"  Date range: {df.index[0].date()} to {df.index[-1].date()}")
print(f"  Current price: ${current_price_val:.2f}")

# ── Technical Indicators ────────────────────────────────────────────
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - 100 / (1 + rs)

def compute_macd(series, fast=12, slow=26, sig=9):
    ema_fast = series.ewm(span=fast).mean()
    ema_slow = series.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=sig).mean()
    return macd - signal

def compute_bollinger_pct(series, period=20):
    ma = series.rolling(period).mean()
    std = series.rolling(period).std()
    return (series - (ma - 2*std)) / (4*std + 1e-9)

def compute_atr(df_in, period=14):
    hi, lo, cl = df_in['High'], df_in['Low'], df_in['Close']
    tr = pd.concat([
        hi - lo,
        (hi - cl.shift()).abs(),
        (lo - cl.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

df['RSI'] = compute_rsi(df['Close'])
df['MACD'] = compute_macd(df['Close'])
df['BollPct'] = compute_bollinger_pct(df['Close'])
df['ATR'] = compute_atr(df)
df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))
df['vol_12h'] = df['log_ret'].rolling(12).std()
df['vol_24h'] = df['log_ret'].rolling(24).std()

for span in [7, 21, 50, 200]:
    df[f'EMA{span}'] = df['Close'].ewm(span=span, adjust=False).mean()

df['ema_stack'] = (
    (df['Close'] > df['EMA7']).astype(int) +
    (df['Close'] > df['EMA21']).astype(int) +
    (df['Close'] > df['EMA50']).astype(int) +
    (df['Close'] > df['EMA200']).astype(int)
)

df['vol_norm'] = np.log1p(df['Volume']) / np.log1p(df['Volume']).rolling(24).mean()

df.dropna(inplace=True)
print(f"  After feature engineering: {len(df)} rows")
print(f"  Features: Close, RSI, MACD, Bollinger, ATR, vol, EMAs, ema_stack, vol_norm")

# =====================================================================
# SECTION 2: HMM Regime Detection
# =====================================================================
print("\n[2/6] Training Hidden Markov Model (regime detection)...")

X_hmm = df[['log_ret', 'vol_24h']].dropna().values
hmm_model = GaussianHMM(n_components=3, covariance_type='spherical', n_iter=100, random_state=42, min_covar=1e-4)
try:
    hmm_model.fit(X_hmm)
    hmm_ok = True
except (ValueError, np.linalg.LinAlgError):
    print("  HMM fitting failed, using fallback regime detection")
    hmm_ok = False

if hmm_ok:
    states = hmm_model.predict(X_hmm)
    hmm_idx = df[['log_ret', 'vol_24h']].dropna().index
    state_series = pd.Series(states, index=hmm_idx)
else:
    hmm_idx = df[['log_ret', 'vol_24h']].dropna().index
    returns = df.loc[hmm_idx, 'log_ret']
    states = (returns > returns.rolling(50).mean()).astype(int) + (returns > 0).astype(int)
    state_series = pd.Series(states, index=hmm_idx)

state_returns = {}
for s in range(3):
    mask = state_series == s
    mean_r = df.loc[mask, 'log_ret'].mean()
    state_returns[s] = mean_r

sorted_states = sorted(state_returns, key=lambda x: state_returns[x])
state_map = {sorted_states[0]: 'BEAR', sorted_states[1]: 'CHOP', sorted_states[2]: 'BULL'}

df['hmm_state_raw'] = state_series.reindex(df.index)
df['hmm_regime'] = df['hmm_state_raw'].map(state_map)
df['hmm_regime'] = df['hmm_regime'].ffill()

current_regime = df['hmm_regime'].iloc[-1]
print(f"  Current regime: {current_regime}")
print(f"  State mean returns: {[(state_map[s], f'{state_returns[s]*100:.4f}%/hour') for s in range(3)]}")

# =====================================================================
# SECTION 3: CNN-LSTM Model Training (Sophisticated Deep Learning)
# =====================================================================
print("\n[3/6] Building & training CNN-LSTM model (deep learning)...")

FEATURES = [
    'log_ret', 'vol_24h', 'vol_12h',
    'RSI', 'MACD', 'BollPct', 'ATR',
    'ema_stack', 'vol_norm'
]

df['hmm_num'] = df['hmm_regime'].map({'BEAR': 0, 'CHOP': 1, 'BULL': 2})
FEATURES.append('hmm_num')

df_model = df[FEATURES + ['Close']].dropna().copy()

# Scaling
scaler = RobustScaler()
X_scaled = scaler.fit_transform(df_model[FEATURES].values)
prices = df_model['Close'].values

# Build sequences for CNN-LSTM
X_seq, y_seq = [], []
for i in range(LOOKBACK_HOURS, len(X_scaled) - HORIZON_HOURS):
    X_seq.append(X_scaled[i-LOOKBACK_HOURS:i])
    fwd = np.log(prices[i+1:i+HORIZON_HOURS+1] / prices[i:i+HORIZON_HOURS]).mean()
    y_seq.append(fwd)

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

# Train/val/test split
N = len(X_seq)
train_end = int(N * 0.80)
val_end = int(N * 0.90)

X_train = X_seq[:train_end]
y_train = y_seq[:train_end]
X_val = X_seq[train_end:val_end]
y_val = y_seq[train_end:val_end]
X_test = X_seq[val_end:]
y_test = y_seq[val_end:]

print(f"  Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

# Build CNN-LSTM with MC Dropout
def build_cnn_lstm(seq_len, n_feat, dropout_rate=0.25):
    inp = keras.Input(shape=(seq_len, n_feat))
    x = layers.Conv1D(64, 3, padding='causal', activation='relu')(inp)
    x = layers.Conv1D(64, 3, padding='causal', activation='relu', dilation_rate=2)(x)
    x = layers.Conv1D(32, 3, padding='causal', activation='relu', dilation_rate=4)(x)
    x = layers.Dropout(dropout_rate)(x, training=True)
    x = layers.LSTM(64, return_sequences=True)(x)
    x = layers.LSTM(32)(x)
    x = layers.Dropout(dropout_rate)(x, training=True)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)
    out = layers.Dense(1)(x)
    return keras.Model(inp, out)

tf.random.set_seed(42)
np.random.seed(42)

model = build_cnn_lstm(LOOKBACK_HOURS, len(FEATURES))
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='huber')

es_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_cb = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5, verbose=0)

hist = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=MODEL_EPOCHS,
    batch_size=32,
    callbacks=[es_cb, lr_cb],
    verbose=VERBOSE
)

test_loss = model.evaluate(X_test, y_test, verbose=0)
print(f"  Test Loss (scaled): {test_loss:.5f}")

# =====================================================================
# SECTION 4: Monte Carlo Dropout Inference
# =====================================================================
print("\n[4/6] Running Monte Carlo dropout inference (200 passes)...")

N_MC = 200

def mc_predict(model, X, n_samples=N_MC):
    preds = np.stack([model(X, training=True).numpy().flatten() for _ in range(n_samples)], axis=0)
    mean = preds.mean(axis=0)
    std = preds.std(axis=0)
    return mean.reshape(1, -1) if mean.ndim == 1 else mean, std.reshape(1, -1) if std.ndim == 1 else std

print("  Calibrating conformal prediction bands...")
v_mean, v_std = mc_predict(model, X_val, n_samples=50)
cal_err = np.abs(v_mean.flatten() - y_val)
alpha = 0.10
q_hat = np.percentile(cal_err, 90)

print(f"  Conformal quantile (90% coverage): ±{q_hat*100:.3f}% log-ret")

print("  Running live inference...")
X_live = X_scaled[-LOOKBACK_HOURS:].reshape(1, LOOKBACK_HOURS, -1)
live_mean, live_std = mc_predict(model, X_live, n_samples=N_MC)

def log_rets_to_prices(start_price, log_rets):
    cum = np.concatenate([[0.0], np.cumsum(log_rets)])
    return float(start_price) * np.exp(cum[1:])

current_price = float(df['Close'].values[-1])
mean_path = log_rets_to_prices(current_price, live_mean[0])
lower_path = log_rets_to_prices(current_price, live_mean[0] - q_hat)
upper_path = log_rets_to_prices(current_price, live_mean[0] + q_hat)
mc_lower = log_rets_to_prices(current_price, live_mean[0] - 2*live_std[0])
mc_upper = log_rets_to_prices(current_price, live_mean[0] + 2*live_std[0])

# =====================================================================
# SECTION 5: Find Next Sunday 12:00 AM UTC
# =====================================================================
print("\n[5/6] Calculating forecast for next Sunday 12:00 AM UTC...")

utc_now = datetime.datetime.now(pytz.UTC)
current_weekday = utc_now.weekday()

if current_weekday == 6:
    hours_to_sunday = 24
else:
    days_until_sunday = (6 - current_weekday) % 7
    if days_until_sunday == 0:
        days_until_sunday = 7
    hours_to_sunday = days_until_sunday * 24 - utc_now.hour + 0

next_sunday = utc_now + datetime.timedelta(hours=hours_to_sunday)
next_sunday = next_sunday.replace(hour=0, minute=0, second=0, microsecond=0)

if hours_to_sunday > HORIZON_HOURS:
    forecast_hour_idx = HORIZON_HOURS - 1
else:
    forecast_hour_idx = int(hours_to_sunday)

print(f"  Current time (UTC): {utc_now.strftime('%Y-%m-%d %H:%M:%S UTC')}")
print(f"  Next Sunday 12:00 AM UTC: {next_sunday.strftime('%Y-%m-%d %H:%M:%S UTC')}")
print(f"  Hours until target: {hours_to_sunday}")

sunday_mean = mean_path[forecast_hour_idx]
sunday_lower = lower_path[forecast_hour_idx]
sunday_upper = upper_path[forecast_hour_idx]
sunday_mc_lower = mc_lower[forecast_hour_idx]
sunday_mc_upper = mc_upper[forecast_hour_idx]

sunday_pct_change = (sunday_mean / current_price - 1) * 100

print(f"\n  ┌─────────────────────────────────────────────────────────┐")
print(f"  │ BITCOIN FORECAST: Next Sunday 12:00 AM UTC            │")
print(f"  ├─────────────────────────────────────────────────────────┤")
print(f"  │ Current price:        ${current_price:.2f}                    │")
print(f"  │ Mean forecast:        ${sunday_mean:.2f} ({sunday_pct_change:+.2f}%)        │")
print(f"  │ 90% Conf. interval:   ${sunday_lower:.2f} - ${sunday_upper:.2f}           │")
print(f"  │ MC Dropout bounds:    ${sunday_mc_lower:.2f} - ${sunday_mc_upper:.2f}            │")
print(f"  │ Regime:               {current_regime}                         │")
print(f"  └─────────────────────────────────────────────────────────┘")

# =====================================================================
# SECTION 6: Event Scenario Monte Carlo
# =====================================================================
print("\n[6/6] Running scenario analysis (10,000 paths)...")

N_PATHS = 10000
pre_mean_hourly = float(np.mean(live_mean[0][:int(hours_to_sunday/6)+1]))
pre_vol = float(df['vol_24h'].iloc[-1])

SCENARIOS = {
    'BULL: ETF approval / macro catalyst': {
        'prob': 0.35,
        'shock': +0.04,
        'color': '#00cc66',
        'symbol': '▲'
    },
    'BASE: No major catalyst': {
        'prob': 0.40,
        'shock': 0.00,
        'color': '#ffaa00',
        'symbol': '●'
    },
    'BEAR: Regulatory headwinds': {
        'prob': 0.25,
        'shock': -0.06,
        'color': '#ff4444',
        'symbol': '▼'
    }
}

scenario_paths = {}
scenario_stats = {}

for name, s in SCENARIOS.items():
    paths = np.zeros((N_PATHS, int(hours_to_sunday)))
    for p_idx in range(N_PATHS):
        price = current_price
        for h in range(int(hours_to_sunday)):
            ret = pre_mean_hourly + np.random.normal(0, pre_vol)
            ret += s['shock'] / hours_to_sunday
            price = price * np.exp(ret)
            paths[p_idx, h] = price

    final_prices = paths[:, -1]
    scenario_paths[name] = paths
    scenario_stats[name] = {
        'mean': final_prices.mean(),
        'median': np.median(final_prices),
        'p10': np.percentile(final_prices, 10),
        'p25': np.percentile(final_prices, 25),
        'p75': np.percentile(final_prices, 75),
        'p90': np.percentile(final_prices, 90),
        'prob_up_5pct': np.mean(final_prices > current_price * 1.05),
        'prob_up_10pct': np.mean(final_prices > current_price * 1.10),
        'prob_down_5pct': np.mean(final_prices < current_price * 0.95),
        'prob': s['prob']
    }

comp_final = np.zeros(N_PATHS)
for name, s in SCENARIOS.items():
    comp_final += s['prob'] * scenario_paths[name][:, -1]

print(f"\n  Scenario Analysis Results:")
for name, stats in scenario_stats.items():
    print(f"\n  {name}")
    print(f"    Probability: {stats['prob']*100:.0f}%")
    print(f"    Mean: ${stats['mean']:.2f} | Median: ${stats['median']:.2f}")
    print(f"    Range (P10-P90): ${stats['p10']:.2f} - ${stats['p90']:.2f}")
    print(f"    P(+5%): {stats['prob_up_5pct']*100:.1f}% | P(+10%): {stats['prob_up_10pct']*100:.1f}%")
    print(f"    P(-5%): {stats['prob_down_5pct']*100:.1f}%")

print(f"\n  ╭─ PROBABILITY-WEIGHTED COMPOSITE ─╮")
print(f"  │ Mean: ${np.mean(comp_final):.2f}")
print(f"  │ P(+5%): {np.mean(comp_final > current_price*1.05)*100:.1f}%")
print(f"  │ P(+10%): {np.mean(comp_final > current_price*1.10)*100:.1f}%")
print(f"  │ P(-5%): {np.mean(comp_final < current_price*0.95)*100:.1f}%")
print(f"  ╰───────────────────────────────────╯")

# =====================================================================
# Visualization
# =====================================================================
print("\n[SAVING] Generating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.patch.set_facecolor('#0d1117')

for ax in axes.flat:
    ax.set_facecolor('#1a1a2e')
    ax.tick_params(colors='#aaa')

# Panel 1: Recent price + forecast
ax = axes[0, 0]
recent_df = df.tail(168)
ax.plot(recent_df.index, recent_df['Close'], color='#ffdd44', lw=2, label='BTC (last 7d)', zorder=5)
forecast_hours = np.arange(int(hours_to_sunday))
forecast_times = [df.index[-1] + datetime.timedelta(hours=h) for h in forecast_hours]
ax.plot(forecast_times, mean_path[:int(hours_to_sunday)], color='cyan', lw=2, ls='--', label='CNN-LSTM Mean', zorder=6)
ax.fill_between(forecast_times, lower_path[:int(hours_to_sunday)], upper_path[:int(hours_to_sunday)],
                 alpha=0.15, color='cyan', label='90% CI', zorder=4)
ax.axvline(next_sunday, color='lime', lw=2, ls=':', alpha=0.8, label='Target: Sunday 12 AM UTC')
ax.set_ylabel('Price (USD)', color='white')
ax.set_title('Bitcoin Price + CNN-LSTM Forecast', color='white', fontsize=11)
ax.legend(loc='best', facecolor='#0a0a0a', labelcolor='white', fontsize=8)
ax.grid(True, alpha=0.2)

# Panel 2: Training history
ax = axes[0, 1]
ax.plot(hist.history['loss'], label='Train Loss', color='#ff7777', lw=1.5)
ax.plot(hist.history['val_loss'], label='Val Loss', color='#77ff77', lw=1.5)
ax.set_ylabel('Loss', color='white')
ax.set_xlabel('Epoch', color='white')
ax.set_title('CNN-LSTM Training Loss (Huber)', color='white', fontsize=11)
ax.legend(facecolor='#0a0a0a', labelcolor='white', fontsize=8)
ax.grid(True, alpha=0.2)

# Panel 3: Scenario fan chart
ax = axes[1, 0]
hours_array = np.arange(1, int(hours_to_sunday) + 1)
for name, s in SCENARIOS.items():
    paths = scenario_paths[name]
    p50 = np.percentile(paths, 50, axis=0)
    p10 = np.percentile(paths, 10, axis=0)
    p90 = np.percentile(paths, 90, axis=0)
    ax.fill_between(hours_array, p10, p90, alpha=0.1, color=s['color'])
    ax.plot(hours_array, p50, color=s['color'], lw=2, label=f"{name} ({s['prob']*100:.0f}%)")

ax.axhline(current_price, color='white', lw=1, ls=':', alpha=0.5)
ax.set_ylabel('Price (USD)', color='white')
ax.set_xlabel('Hours until Sunday 12 AM UTC', color='white')
ax.set_title('Event Scenario Analysis (10k paths)', color='white', fontsize=11)
ax.legend(facecolor='#0a0a0a', labelcolor='white', fontsize=7.5)
ax.grid(True, alpha=0.2)

# Panel 4: Final distribution
ax = axes[1, 1]
for name, s in SCENARIOS.items():
    final_prices = scenario_paths[name][:, -1]
    ax.hist(final_prices, bins=60, alpha=0.35, color=s['color'], label=name, density=True)

ax.axvline(current_price, color='white', lw=1.5, ls='--', label='Current')
ax.axvline(sunday_mean, color='cyan', lw=2, ls='--', label='CNN-LSTM Mean')
ax.set_xlabel('BTC Price (USD)', color='white')
ax.set_ylabel('Density', color='white')
ax.set_title('Sunday 12 AM UTC: Price Distribution', color='white', fontsize=11)
ax.legend(facecolor='#0a0a0a', labelcolor='white', fontsize=8)
ax.grid(True, alpha=0.2, axis='y')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'btc_forecast.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
print(f"  Saved: {OUTPUT_DIR}/btc_forecast.png")
plt.close()

# =====================================================================
# Summary Report
# =====================================================================
report_file = OUTPUT_DIR / 'forecast_report.txt'
with open(report_file, 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("BITCOIN PRICE FORECAST REPORT - CNN-LSTM + HMM + Monte Carlo\n")
    f.write("=" * 70 + "\n\n")

    f.write(f"Generated: {utc_now.strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
    f.write(f"Current Price: ${current_price:.2f}\n")
    f.write(f"Current Regime: {current_regime}\n\n")

    f.write(f"TARGET: Sunday 12:00 AM UTC ({next_sunday.strftime('%Y-%m-%d %H:%M:%S')})\n")
    f.write(f"Hours until target: {hours_to_sunday}\n\n")

    f.write("FORECAST (CNN-LSTM + MC Dropout):\n")
    f.write(f"  Mean price:         ${sunday_mean:.2f} ({sunday_pct_change:+.2f}%)\n")
    f.write(f"  90% CI:             ${sunday_lower:.2f} - ${sunday_upper:.2f}\n")
    f.write(f"  MC range (2σ):      ${sunday_mc_lower:.2f} - ${sunday_mc_upper:.2f}\n\n")

    f.write("SCENARIO ANALYSIS (10,000 paths each):\n")
    for name, stats in scenario_stats.items():
        f.write(f"\n  {name}\n")
        f.write(f"    Probability: {stats['prob']*100:.0f}%\n")
        f.write(f"    Mean: ${stats['mean']:.2f} | Median: ${stats['median']:.2f}\n")
        f.write(f"    P10-P90: ${stats['p10']:.2f} - ${stats['p90']:.2f}\n")
        f.write(f"    P(+5%): {stats['prob_up_5pct']*100:.1f}% | P(+10%): {stats['prob_up_10pct']*100:.1f}%\n")
        f.write(f"    P(-5%): {stats['prob_down_5pct']*100:.1f}%\n")

    f.write(f"\n\nPROBABILITY-WEIGHTED COMPOSITE:\n")
    f.write(f"  Mean: ${np.mean(comp_final):.2f}\n")
    f.write(f"  P(+5%): {np.mean(comp_final > current_price*1.05)*100:.1f}%\n")
    f.write(f"  P(+10%): {np.mean(comp_final > current_price*1.10)*100:.1f}%\n")
    f.write(f"  P(-5%): {np.mean(comp_final < current_price*0.95)*100:.1f}%\n")

print(f"  Saved: {report_file}")

print("\n" + "=" * 80)
print("✅ COMPLETE! All outputs saved to:", OUTPUT_DIR)
print("=" * 80)
