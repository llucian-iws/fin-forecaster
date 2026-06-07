#!/usr/bin/env python3
"""
Bitcoin Price Forecaster - Lite Version
Fast execution with core forecasting features
"""

import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import GradientBoostingRegressor
from pathlib import Path
import datetime
import pytz
import sys
import os
import argparse

OUTPUT_DIR = Path("/app/results")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Parse command-line arguments or environment variables for target date/time
parser = argparse.ArgumentParser(description='Bitcoin Price Forecaster - Lite')
parser.add_argument('--target-date', type=str, default=os.getenv('TARGET_DATE'),
                    help='Target date in format YYYY-MM-DD or "next-<day>" (e.g., next-wednesday)')
parser.add_argument('--target-hour', type=int, default=int(os.getenv('TARGET_HOUR', 0)),
                    help='Target hour in UTC (0-23, default 0 for midnight)')
args = parser.parse_args()

print("=" * 80)
print("BITCOIN PRICE FORECASTER - LITE VERSION")
print("=" * 80)

def calculate_target_datetime(target_date_arg, target_hour):
    utc_now = datetime.datetime.now(pytz.UTC)
    weekdays = {
        'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
        'friday': 4, 'saturday': 5, 'sunday': 6
    }

    if target_date_arg:
        if target_date_arg.lower().startswith('next-'):
            day_name = target_date_arg.lower().replace('next-', '')
            if day_name not in weekdays:
                print(f"Unknown day: {day_name}. Using next-wednesday.")
                day_name = 'wednesday'
            target_weekday = weekdays[day_name]
            current_weekday = utc_now.weekday()
            days_until = (target_weekday - current_weekday) % 7
            if days_until == 0:
                days_until = 7
            target = utc_now + datetime.timedelta(days=days_until)
        else:
            try:
                target = datetime.datetime.strptime(target_date_arg, '%Y-%m-%d')
                target = pytz.UTC.localize(target)
            except ValueError:
                print(f"Invalid date format: {target_date_arg}. Using next-wednesday.")
                return calculate_target_datetime('next-wednesday', target_hour)
    else:
        target_weekday = 2
        current_weekday = utc_now.weekday()
        days_until = (target_weekday - current_weekday) % 7
        if days_until == 0:
            days_until = 7
        target = utc_now + datetime.timedelta(days=days_until)

    target = target.replace(hour=target_hour, minute=0, second=0, microsecond=0)
    hours_to_target = (target - utc_now).total_seconds() / 3600.0
    return target, hours_to_target

target_dt, hours_to_target = calculate_target_datetime(args.target_date, args.target_hour)

# =====================================================================
# 1. Fetch Data
# =====================================================================
print("\n[1/5] Fetching BTC data...")
end_date = datetime.datetime.now()
start_date = end_date - datetime.timedelta(days=700)
date_str_start = start_date.strftime('%Y-%m-%d')
date_str_end = end_date.strftime('%Y-%m-%d')

df = yf.download('BTC-USD', start=date_str_start, end=date_str_end, interval='1h', progress=False)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.droplevel(-1)
df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy().dropna()

current_price = float(df['Close'].values[-1])
print(f"  Downloaded {len(df)} candles")
print(f"  Current BTC price: ${current_price:.2f}")

# =====================================================================
# 2. Feature Engineering
# =====================================================================
print("\n[2/5] Engineering features...")
df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))
df['vol'] = df['log_ret'].rolling(24).std()
df['rsi'] = 100 - (100 / (1 + (df['log_ret'].rolling(14).mean() / df['log_ret'].rolling(14).std())))
for span in [7, 21, 50]:
    df[f'ema{span}'] = df['Close'].ewm(span=span).mean()
df['ema_stack'] = ((df['Close'] > df['ema7']).astype(int) +
                    (df['Close'] > df['ema21']).astype(int) +
                    (df['Close'] > df['ema50']).astype(int))
df = df.dropna()
print(f"  Features engineered: {len(df)} rows")

# =====================================================================
# 3. Train Model
# =====================================================================
print("\n[3/5] Training ensemble model...")
FEATURES = ['log_ret', 'vol', 'rsi', 'ema_stack']
scaler = RobustScaler()
X = scaler.fit_transform(df[FEATURES].values[:-1])
y = df['log_ret'].values[1:]

split = int(len(X) * 0.85)
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

model = GradientBoostingRegressor(
    n_estimators=50,
    learning_rate=0.05,
    max_depth=4,
    random_state=42,
    verbose=0
)
model.fit(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"  Model R² score: {test_score:.4f}")

# =====================================================================
# 4. Make Prediction
# =====================================================================
print(f"\n[4/5] Generating forecast for {target_dt.strftime('%A %H:%M UTC')}...")
utc_now = datetime.datetime.now(pytz.UTC)

# Predict
X_latest = scaler.transform(df[FEATURES].values[-1:])
predictions = []
for _ in range(100):
    # Bootstrap prediction
    pred = model.predict(X_latest + np.random.normal(0, 0.01, X_latest.shape))[0]
    predictions.append(pred)

predictions = np.array(predictions)
forecast_return = predictions.mean()          # predicted per-hour log return

# Uncertainty is driven by realized return volatility (24h rolling std of
# hourly log returns), scaled to the target horizon as a random walk. The GBM
# bootstrap std is ~0 (trees are flat to tiny input noise), so it cannot
# represent price uncertainty on its own.
sigma_h = float(df['vol'].iloc[-1])
horizon_ret = forecast_return * hours_to_target
horizon_std = sigma_h * np.sqrt(hours_to_target)
forecast_std = horizon_std

# Convert to price at the target horizon
forecast_price = current_price * np.exp(horizon_ret)
price_lower = current_price * np.exp(horizon_ret - 2*horizon_std)
price_upper = current_price * np.exp(horizon_ret + 2*horizon_std)

# Market regime
regime = "BULL" if forecast_return > 0 else "BEAR"

print(f"  Target: {target_dt.strftime('%A %H:%M UTC (%Y-%m-%d)')} ({hours_to_target:.1f}h away)")
print(f"  Regime: {regime}")

# =====================================================================
# 5. Scenario Analysis
# =====================================================================
print("\n[5/5] Running scenario analysis (5k paths)...")
N_PATHS = 5000
scenarios = {
    'BULL (+4%)': {'shock': 0.04, 'prob': 0.35, 'color': '#00cc66'},
    'BASE (0%)': {'shock': 0.00, 'prob': 0.40, 'color': '#ffaa00'},
    'BEAR (-6%)': {'shock': -0.06, 'prob': 0.25, 'color': '#ff4444'}
}

H = int(hours_to_target)
scenario_results = {}
for name, config in scenarios.items():
    # Vectorized GBM: each path's final price = current * exp(sum of H hourly
    # log-returns). Identical draws/distribution to the per-step loop, just no
    # Python inner loop.
    drift = forecast_return + config['shock'] / hours_to_target
    rets = drift + np.random.normal(0, sigma_h, size=(N_PATHS, H))
    paths = current_price * np.exp(rets.sum(axis=1))
    scenario_results[name] = {
        'mean': paths.mean(),
        'median': np.median(paths),
        'p10': np.percentile(paths, 10),
        'p90': np.percentile(paths, 90),
        'prob_up_5': np.mean(paths > current_price * 1.05),
        'prob_up_10': np.mean(paths > current_price * 1.10),
        'prob_down_5': np.mean(paths < current_price * 0.95),
        'paths': paths
    }

# Weighted composite
weighted_paths = np.zeros(N_PATHS)
for name, config in scenarios.items():
    weighted_paths += config['prob'] * scenario_results[name]['paths']

# =====================================================================
# Generate Report
# =====================================================================
report = f"""================================================================================
BITCOIN PRICE FORECAST REPORT
================================================================================

Generated: {utc_now.strftime('%Y-%m-%d %H:%M:%S UTC')}
Current Price: ${current_price:.2f}
Target: {target_dt.strftime('%A %H:%M UTC (%Y-%m-%d)')}
Hours until target: {hours_to_target:.1f}

================================================================================
FORECAST (CNN-LSTM + MC Bootstrap)
================================================================================

Mean price:         ${forecast_price:.2f} ({(forecast_price/current_price - 1)*100:+.2f}%)
95% CI:             ${price_lower:.2f} - ${price_upper:.2f}
Forecast std:       ${forecast_std * current_price:.2f}
Market Regime:      {regime}

================================================================================
SCENARIO ANALYSIS (5,000 paths each)
================================================================================

"""

composite_mean = weighted_paths.mean()

for name, results in scenario_results.items():
    prob = scenarios[name]['prob']
    report += f"\n{name} (prob={prob*100:.0f}%)\n"
    report += f"  Mean: ${results['mean']:.2f} | Median: ${results['median']:.2f}\n"
    report += f"  P10-P90: ${results['p10']:.2f} - ${results['p90']:.2f}\n"
    report += f"  P(+5%): {results['prob_up_5']*100:.1f}% | P(+10%): {results['prob_up_10']*100:.1f}%\n"
    report += f"  P(-5%): {results['prob_down_5']*100:.1f}%\n"

report += f"""
================================================================================
PROBABILITY-WEIGHTED COMPOSITE
================================================================================

Mean: ${composite_mean:.2f} ({(composite_mean/current_price - 1)*100:+.2f}%)
P(+5%): {np.mean(weighted_paths > current_price*1.05)*100:.1f}%
P(+10%): {np.mean(weighted_paths > current_price*1.10)*100:.1f}%
P(-5%): {np.mean(weighted_paths < current_price*0.95)*100:.1f}%

================================================================================
"""

# Save report
with open(OUTPUT_DIR / 'forecast_report.txt', 'w') as f:
    f.write(report)

# =====================================================================
# Generate Visualization
# =====================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.patch.set_facecolor('#0d1117')

for ax in axes.flat:
    ax.set_facecolor('#1a1a2e')
    ax.tick_params(colors='#aaa')

# Panel 1: Recent price + forecast
ax = axes[0, 0]
recent = df['Close'].tail(168).values
ax.plot(recent, color='#ffdd44', lw=2, label='BTC (last 7d)', zorder=5)
ax.axhline(forecast_price, color='cyan', lw=2, ls='--', label=f'Forecast: ${forecast_price:.0f}')
ax.fill_between(range(len(recent)), price_lower, price_upper, alpha=0.2, color='cyan', label='95% CI')
ax.set_ylabel('Price (USD)', color='white')
ax.set_title('BTC Price + Forecast', color='white', fontsize=11)
ax.legend(facecolor='#0a0a0a', labelcolor='white', fontsize=9)
ax.grid(True, alpha=0.2)

# Panel 2: Feature importance
ax = axes[0, 1]
importances = model.feature_importances_
ax.barh(FEATURES, importances, color='#4488ff')
ax.set_xlabel('Importance', color='white')
ax.set_title('Feature Importance', color='white', fontsize=11)
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')
ax.set_facecolor('#1a1a2e')

# Panel 3: Scenario distributions
ax = axes[1, 0]
for name, results in scenario_results.items():
    color = scenarios[name]['color']
    ax.hist(results['paths'], bins=50, alpha=0.4, color=color, label=name, density=True)
ax.axvline(current_price, color='white', lw=2, ls='--', label='Current')
ax.set_xlabel('Price (USD)', color='white')
ax.set_ylabel('Density', color='white')
ax.set_title('Scenario Price Distribution', color='white', fontsize=11)
ax.legend(facecolor='#0a0a0a', labelcolor='white', fontsize=9)
ax.tick_params(colors='#aaa')

# Panel 4: Summary stats
ax = axes[1, 1]
ax.axis('off')
summary_text = f"""
FORECAST SUMMARY

Current: ${current_price:.2f}
Forecast: ${forecast_price:.2f}
Change: {(forecast_price/current_price - 1)*100:+.2f}%

Composite Mean: ${composite_mean:.2f}
P(+5%): {np.mean(weighted_paths > current_price*1.05)*100:.1f}%
P(+10%): {np.mean(weighted_paths > current_price*1.10)*100:.1f}%
P(-5%): {np.mean(weighted_paths < current_price*0.95)*100:.1f}%

Target: {target_dt.strftime('%A %H:%M UTC')}
Hours: {hours_to_target:.1f}
Regime: {regime}
"""
ax.text(0.1, 0.5, summary_text, color='white', fontsize=11, family='monospace',
        verticalalignment='center', bbox=dict(boxstyle='round', facecolor='#0a0a0a', alpha=0.8))

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'btc_forecast.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
print(f"  Saved: {OUTPUT_DIR / 'btc_forecast.png'}")

print("\n" + "=" * 80)
print("✅ FORECAST COMPLETE")
print("=" * 80)
print(report)
