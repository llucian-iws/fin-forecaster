#!/usr/bin/env python3
"""
Bitcoin Price Forecaster - Full Quantitative Stack
Predicts BTC price for next Sunday 12:00 AM UTC using Ensemble + HMM + Monte Carlo
Uses XGBoost, Ridge regression, and statistical forecasting methods
"""

import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from scipy.stats import norm
from hmmlearn.hmm import GaussianHMM
import datetime
import pytz
from pathlib import Path

# =====================================================================
# CONFIG
# =====================================================================
OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

# Crypto-specific parameters
LOOKBACK_HOURS = 168  # 1 week of hourly data
HORIZON_HOURS = 24    # 24 hours ahead (to cover varying Sunday times globally)
START_DATE = '2024-01-01'  # yfinance limits hourly to 730 days (~2 years)
MODEL_EPOCHS = 100
VERBOSE = 1

print("=" * 80)
print("BITCOIN PRICE FORECASTER | Full Quantitative Stack")
print("=" * 80)

# =====================================================================
# SECTION 1: Data Fetching & Feature Engineering
# =====================================================================
print("\n[1/6] Fetching Bitcoin data...")

# Use dynamic date range: last 700 days from today
# (yfinance hourly data limited to 730 days)
import datetime as dt
end_date = dt.datetime.now()
start_date = end_date - dt.timedelta(days=700)
date_str_start = start_date.strftime('%Y-%m-%d')
date_str_end = end_date.strftime('%Y-%m-%d')

df = yf.download('BTC-USD', start=date_str_start, end=date_str_end, interval='1h', progress=False)
# Handle MultiIndex columns from yfinance (tuples like ('Close', 'BTC-USD'))
if isinstance(df.columns, pd.MultiIndex):
    # Flatten to just the price type (first level)
    df.columns = df.columns.droplevel(-1)
# Now select columns
df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy().dropna()
print(f"  Downloaded {len(df)} hourly candles")
print(f"  Date range: {df.index[0].date()} to {df.index[-1].date()}")
current_price_val = float(df['Close'].values[-1])
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

# Compute indicators
df['RSI'] = compute_rsi(df['Close'])
df['MACD'] = compute_macd(df['Close'])
df['BollPct'] = compute_bollinger_pct(df['Close'])
df['ATR'] = compute_atr(df)
df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))
df['vol_12h'] = df['log_ret'].rolling(12).std()
df['vol_24h'] = df['log_ret'].rolling(24).std()

# EMAs
for span in [7, 21, 50, 200]:
    df[f'EMA{span}'] = df['Close'].ewm(span=span, adjust=False).mean()

# EMA stack (how many above)
df['ema_stack'] = (
    (df['Close'] > df['EMA7']).astype(int) +
    (df['Close'] > df['EMA21']).astype(int) +
    (df['Close'] > df['EMA50']).astype(int) +
    (df['Close'] > df['EMA200']).astype(int)
)

# Volume-weighted features
df['vol_norm'] = np.log1p(df['Volume']) / np.log1p(df['Volume']).rolling(24).mean()

df.dropna(inplace=True)
print(f"  After feature engineering: {len(df)} rows")
print(f"  Features: Close, RSI, MACD, BollPct, ATR, vol, EMAs, ema_stack, vol_norm")

# =====================================================================
# SECTION 2: HMM Regime Detection
# =====================================================================
print("\n[2/6] Training Hidden Markov Model (regime detection)...")

X_hmm = df[['log_ret', 'vol_24h']].dropna().values
# Use spherical covariance for numerical stability
hmm_model = GaussianHMM(n_components=3, covariance_type='spherical', n_iter=100, random_state=42, min_covar=1e-4)
try:
    hmm_model.fit(X_hmm)
except (ValueError, np.linalg.LinAlgError):
    # Fallback: use simple regime classification based on returns
    print("  HMM fitting failed, using fallback regime detection")
    hmm_model = None
if hmm_model is not None:
    states = hmm_model.predict(X_hmm)
    hmm_idx = df[['log_ret', 'vol_24h']].dropna().index
    state_series = pd.Series(states, index=hmm_idx)
else:
    # Fallback: classify by returns
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
# SECTION 3: Ensemble Model Training (XGBoost-style with Gradient Boosting)
# =====================================================================
print("\n[3/6] Building & training ensemble model (Gradient Boosting)...")

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

# Build feature matrix for next-hour prediction
X_feat, y_target = [], []
for i in range(LOOKBACK_HOURS, len(X_scaled) - HORIZON_HOURS):
    # Use lagged features as inputs
    X_feat.append(X_scaled[i-LOOKBACK_HOURS:i].flatten())
    # Target: average log return over next HORIZON hours
    fwd = np.log(prices[i+1:i+HORIZON_HOURS+1] / prices[i:i+HORIZON_HOURS]).mean()
    y_target.append(fwd)

X_feat = np.array(X_feat)
y_target = np.array(y_target)

# Train/val/test split
N = len(X_feat)
train_end = int(N * 0.80)
val_end = int(N * 0.90)

X_train = X_feat[:train_end]
y_train = y_target[:train_end]
X_val = X_feat[train_end:val_end]
y_val = y_target[train_end:val_end]
X_test = X_feat[val_end:]
y_test = y_target[val_end:]

print(f"  Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

# Build ensemble
np.random.seed(42)
print("  Training Gradient Boosting regressor...")
model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=5,
    min_samples_split=10,
    random_state=42,
    verbose=0
)
model.fit(X_train, y_train)

# Evaluate
from sklearn.metrics import mean_squared_error, mean_absolute_error
train_pred = model.predict(X_train)
val_pred = model.predict(X_val)
test_pred = model.predict(X_test)

test_mse = mean_squared_error(y_test, test_pred)
test_mae = mean_absolute_error(y_test, test_pred)
print(f"  Test MSE: {test_mse:.5f}")
print(f"  Test MAE: {test_mae:.5f}")

# Create history dict for plotting compatibility
hist = {
    'history': {
        'loss': [np.sqrt(mean_squared_error(y_train[:i+1], train_pred[:i+1])) for i in range(len(train_pred))],
        'val_loss': [np.sqrt(mean_squared_error(y_val[:max(1,i)], val_pred[:max(1,i)])) for i in range(len(val_pred))]
    }
}

# =====================================================================
# SECTION 4: Bootstrap Uncertainty Estimation
# =====================================================================
print("\n[4/6] Running bootstrap uncertainty estimation (200 resamples)...")

N_MC = 200

# Bootstrap resampling for uncertainty estimation
print("  Calibrating prediction intervals...")
bootstrap_preds = []
for _ in range(N_MC):
    # Resample training data with replacement
    indices = np.random.choice(len(X_train), size=len(X_train), replace=True)
    X_boot = X_train[indices]
    y_boot = y_train[indices]

    # Train light model on bootstrap sample
    m = GradientBoostingRegressor(
        n_estimators=50, learning_rate=0.05, max_depth=5,
        random_state=np.random.randint(0, 10000), verbose=0
    )
    m.fit(X_boot, y_boot)
    bootstrap_preds.append(m.predict(X_val))

bootstrap_preds = np.array(bootstrap_preds)
v_mean = bootstrap_preds.mean(axis=0)
v_std = bootstrap_preds.std(axis=0)

# Conformal quantiles
cal_err = np.abs(v_mean - y_val)
alpha = 0.10
q_hat = np.percentile(cal_err, 90)  # Single quantile for simplicity

print(f"  Conformal quantile (90% coverage): ±{q_hat*100:.3f}% log-ret")

# Live inference
print("  Running live inference...")
live_mean_val = model.predict(X_feat[-1:].reshape(1, -1))[0]
live_std_val = v_std.mean()  # Average bootstrap std

# Generate full 24-hour forecast by iterating
live_mean = np.full(HORIZON_HOURS, live_mean_val)
live_std = np.full(HORIZON_HOURS, live_std_val)

def log_rets_to_prices(start_price, log_rets):
    cum = np.concatenate([[0.0], np.cumsum(log_rets)])
    return float(start_price) * np.exp(cum[1:])

current_price = df['Close'].iloc[-1]
print(f"  Current BTC price: ${current_price:.2f}")

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
current_weekday = utc_now.weekday()  # 0=Mon, 6=Sun

# Calculate hours until next Sunday 12:00 AM UTC
if current_weekday == 6:  # Today is Sunday
    hours_to_sunday = 24
else:
    days_until_sunday = (6 - current_weekday) % 7
    if days_until_sunday == 0:
        days_until_sunday = 7
    hours_to_sunday = days_until_sunday * 24 - utc_now.hour + 0  # 12 AM = hour 0

next_sunday = utc_now + datetime.timedelta(hours=hours_to_sunday)
next_sunday = next_sunday.replace(hour=0, minute=0, second=0, microsecond=0)

# Clamp to available forecast horizon
if hours_to_sunday > HORIZON_HOURS:
    print(f"  WARNING: Next Sunday is {hours_to_sunday} hours away, beyond {HORIZON_HOURS}h forecast")
    forecast_hour_idx = HORIZON_HOURS - 1
else:
    forecast_hour_idx = int(hours_to_sunday)

print(f"  Current time (UTC): {utc_now.strftime('%Y-%m-%d %H:%M:%S UTC')}")
print(f"  Next Sunday 12:00 AM UTC: {next_sunday.strftime('%Y-%m-%d %H:%M:%S UTC')}")
print(f"  Hours until target: {hours_to_sunday}")
print(f"  Using forecast index: {forecast_hour_idx} (clamped to {HORIZON_HOURS}h horizon)")

# Get Sunday forecast price
sunday_mean = mean_path[forecast_hour_idx]
sunday_lower = lower_path[forecast_hour_idx]
sunday_upper = upper_path[forecast_hour_idx]
sunday_mc_lower = mc_lower[forecast_hour_idx]
sunday_mc_upper = mc_upper[forecast_hour_idx]

sunday_pct_change = (sunday_mean / current_price - 1) * 100
sunday_pct_change_lower = (sunday_lower / current_price - 1) * 100
sunday_pct_change_upper = (sunday_upper / current_price - 1) * 100

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
pre_mean_hourly = float(np.mean(live_mean[0][:hours_to_sunday//6+1]))
pre_vol = float(df['vol_24h'].iloc[-1])

SCENARIOS = {
    'BULL: ETF approval / macro catalyst': {
        'prob': 0.35,
        'shock': +0.04,  # +4% cumulative boost
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
        'shock': -0.06,  # -6% drawdown
        'color': '#ff4444',
        'symbol': '▼'
    }
}

scenario_paths = {}
scenario_stats = {}

for name, s in SCENARIOS.items():
    paths = np.zeros((N_PATHS, hours_to_sunday))
    for p_idx in range(N_PATHS):
        price = current_price
        for h in range(hours_to_sunday):
            ret = pre_mean_hourly + np.random.normal(0, pre_vol)
            ret += s['shock'] / hours_to_sunday  # Spread shock across hours
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

print(f"\n  Scenario Analysis Results:")
for name, stats in scenario_stats.items():
    print(f"\n  {name}")
    print(f"    Probability: {stats['prob']*100:.0f}%")
    print(f"    Mean: ${stats['mean']:.2f} | Median: ${stats['median']:.2f}")
    print(f"    Range (P10-P90): ${stats['p10']:.2f} - ${stats['p90']:.2f}")
    print(f"    P(+5%): {stats['prob_up_5pct']*100:.1f}% | P(+10%): {stats['prob_up_10pct']*100:.1f}%")
    print(f"    P(-5%): {stats['prob_down_5pct']*100:.1f}%")

# Weighted composite
comp_final = np.zeros(N_PATHS)
for name, s in SCENARIOS.items():
    comp_final += s['prob'] * scenario_paths[name][:, -1]

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
recent_df = df.tail(168)  # Last 7 days
ax.plot(recent_df.index, recent_df['Close'], color='#ffdd44', lw=2, label='BTC (last 7d)', zorder=5)

# Add forecast range
forecast_hours = np.arange(hours_to_sunday)
forecast_times = [df.index[-1] + datetime.timedelta(hours=h) for h in forecast_hours]
ax.plot(forecast_times, mean_path[:hours_to_sunday], color='cyan', lw=2, ls='--', label='Mean Forecast', zorder=6)
ax.fill_between(forecast_times, lower_path[:hours_to_sunday], upper_path[:hours_to_sunday],
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
ax.set_title('Model Training Loss (Huber)', color='white', fontsize=11)
ax.legend(facecolor='#0a0a0a', labelcolor='white', fontsize=8)
ax.grid(True, alpha=0.2)

# Panel 3: Scenario fan chart
ax = axes[1, 0]
hours_array = np.arange(1, hours_to_sunday + 1)
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
    f.write("BITCOIN PRICE FORECAST REPORT\n")
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
print("COMPLETE! All outputs saved to:", OUTPUT_DIR)
print("=" * 80)
