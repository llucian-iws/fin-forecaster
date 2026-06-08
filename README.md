# fin-forecaster

A quantitative forecasting stack for **price** and **volatility**, covering both
**crypto** and **stocks**. Two complementary engines:

- **Price forecast** — a CNN-LSTM + HMM + Monte Carlo stack that predicts the
  price at a configurable target date/time, with conformal + MC-dropout
  uncertainty bands and 10,000-path event scenarios.
- **Volatility forecast** (`--volatility-only`) — a lightweight implied-vs-realized
  volatility model (Deribit DVOL / CBOE VIX-family, EWMA, GARCH(1,1)) that
  reports the variance-risk premium. No TensorFlow needed.

## Quick start (Docker)

The full CNN-LSTM model needs TensorFlow 2.13+ (Python 3.11), so it runs in
Docker. Results are written to `./results/` via a volume mount.

```bash
docker build -t fin-forecaster:latest .

# Price forecast (default target: next Wednesday 00:00 UTC)
docker run --rm -v "$(pwd)/results:/app/results" fin-forecaster:latest \
  python btc_forecast.py

# Pick a target and average over several retrains
docker run --rm -v "$(pwd)/results:/app/results" fin-forecaster:latest \
  python btc_forecast.py --target-date next-wednesday --target-hour 12 --runs 3
```

The local `btc_forecast_lite.py` (scikit-learn Gradient Boosting) is a fast
fallback that runs without TensorFlow if you can't use Docker.

## Forecast modes

### 1. Price forecast (default)

CNN-LSTM predicts the mean hourly log-return over a 24h horizon; that rate is
compounded to the target. Outputs a point forecast, a 90% conformal band,
MC-dropout bounds, and probability-weighted BULL/BASE/BEAR scenarios.

| Flag | Env | Default | Meaning |
|------|-----|---------|---------|
| `--target-date` | `TARGET_DATE` | `next-wednesday` | `YYYY-MM-DD` or `next-<weekday>` |
| `--target-hour` | `TARGET_HOUR` | `0` | Target hour in UTC (0–23) |
| `--runs N` | `RUNS` | `1` | Retrain N times (distinct seeds) and average the forecast |
| — | `MODEL_EPOCHS` | `5` | Training epochs per run |

`--runs` reduces the run-to-run variance from random init + MC sampling by
averaging the stochastic outputs (hourly return, MC-dropout std, conformal
quantile). Data fetch and HMM run once; only train/infer loops.

### 2. Volatility forecast (`--volatility-only`)

Skips the CNN-LSTM/price path entirely (no TensorFlow) and forecasts volatility:
implied vol vs. realized vol, plus the IV−RV **variance-risk premium** and a
regime label. Works for crypto and stocks.

| Flag | Env | Default | Meaning |
|------|-----|---------|---------|
| `--volatility-only` | `VOLATILITY_ONLY` | off | Enable the volatility path |
| `--asset` | `ASSET` | `crypto` | `crypto` or `stock` |
| `--ticker` | `TICKER` | `BTC-USD` / `SPY` | Symbol to forecast |

```bash
# Crypto: Deribit DVOL + hourly realized vol
docker run --rm -v "$(pwd)/results:/app/results" fin-forecaster:latest \
  python btc_forecast.py --volatility-only --ticker BTC-USD

# Index ETF: CBOE VIX-family index + daily realized vol
docker run --rm -v "$(pwd)/results:/app/results" fin-forecaster:latest \
  python btc_forecast.py --volatility-only --asset stock --ticker SPY

# Single name: ~30-day ATM IV from the option chain (US market hours only)
docker run --rm -v "$(pwd)/results:/app/results" fin-forecaster:latest \
  python btc_forecast.py --volatility-only --asset stock --ticker SNOW
```

**Implied-vol sources** (all keyless):

| Asset | Implied vol | Realized vol annualization |
|-------|-------------|----------------------------|
| Crypto | Deribit DVOL (`get_volatility_index_data`) | hourly, √(24·365) |
| Index ETF/index | CBOE VIX family via yfinance — `SPY→^VIX`, `QQQ→^VXN`, `IWM→^RVX`, `DIA→^VXD` | daily, √252 |
| Single name | ~30-day ATM IV from the yfinance option chain | daily, √252 |

The report shows current implied vol, rolling + EWMA realized vol, a GARCH(1,1)
horizon forecast (falls back to EWMA if the fit is degenerate), and the
`implied − forecast realized` spread with an interpretation.

> **Single-name IV is gated to US market hours.** Yahoo's per-contract implied
> vol is stale/near-zero after the close, so outside 13:30–20:00 UTC (weekdays)
> a single name degrades to a realized-vol-only report. Index ETFs use the VIX
> family and work any time.

## Output

Written to `results/` (mounted from the host):

- `forecast_report.txt` — the report for whichever mode ran.
- `btc_forecast.png` — 4-panel chart (price path + forecast, training loss,
  scenario fan chart, target-time distribution). Price mode only.

Sample reports live in `samples/`.

## Architecture (price mode)

```
BTC-USD hourly (yfinance)
        │
        ▼
Feature engineering ── log_ret, vol_12h/24h, RSI, MACD, Bollinger, ATR,
        │              EMA stack, vol_norm, HMM regime
        ├───────────────► HMM regime detection (3-state BEAR/CHOP/BULL)
        ▼
CNN-LSTM (Conv1D ×3 → LSTM ×2 → Dense), MC-dropout
        │   × --runs (averaged)
        ▼
Conformal bands (90%) + MC-dropout bounds
        ▼
10,000-path BULL/BASE/BEAR scenario Monte Carlo  (shock vol = forward model: GARCH(1,1) ⊕ DVOL, falls back to realized vol_24h)
        ▼
Report + 4-panel chart
```

## Key files

| File | Purpose |
|------|---------|
| `btc_forecast.py` | Full stack: price (CNN-LSTM) + the `--volatility-only` wiring |
| `volatility.py` | Asset-agnostic vol math: DVOL fetch, RV, EWMA, GARCH(1,1), report |
| `btc_forecast_lite.py` | Gradient Boosting fallback (no TensorFlow) |
| `Dockerfile` / `docker-compose.yml` | Python 3.11 + TensorFlow runtime |

## Requirements

Python 3.11 (via Docker for the price path). Dependencies in `requirements.txt`:
numpy, pandas, scikit-learn, scipy, matplotlib, hmmlearn, yfinance, pytz,
tensorflow, xgboost, requests.

## Known limitations

- The CNN-LSTM has a **24h training horizon**; longer targets are reached by
  compounding the predicted hourly rate (an approximation, not a 60h+ model).
- **No accuracy backtest yet** — the tool reports forecasts but does not yet
  measure directional hit-rate, MAE, or interval coverage against realized
  outcomes. Treat outputs as indicative.
- `OUTPUT_DIR` is `/app/results` (Docker). Running the scripts outside the
  container requires that path to be writable.

## License

MIT
