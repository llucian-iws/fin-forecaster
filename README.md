# Bitcoin Price Forecaster

A comprehensive quantitative analysis stack for predicting Bitcoin price on Sunday at 12:00 AM UTC using ensemble machine learning, Hidden Markov Models, and Monte Carlo simulation.

## Features

- **Data Source**: Real-time BTC-USD data from Yahoo Finance (hourly candles)
- **Technical Indicators**: RSI, MACD, Bollinger Bands, ATR, EMAs, Volume normalization
- **Regime Detection**: Hidden Markov Model (3-state: BEAR/CHOP/BULL)
- **Forecasting**: Gradient Boosting Regressor with bootstrap uncertainty
- **Uncertainty Quantification**: Conformal prediction bands + Monte Carlo bootstrap
- **Scenario Analysis**: Bull/Base/Bear event scenarios with 10,000 simulated paths
- **Output**: Predictions, confidence intervals, scenario probabilities, visualizations

## Quick Start

### Using Docker (Recommended)

```bash
# Build and run in Docker
make docker-run

# Or use docker-compose directly
docker-compose up --build
```

### Local Installation

```bash
# Install dependencies
make install
# or
pip install -r requirements.txt

# Run forecast
make run
# or
python3 btc_forecast.py
```

## Output Files

The script generates results in the `results/` directory:

- **btc_forecast.png** - 4-panel visualization:
  - Panel 1: BTC price with CNN-LSTM forecast and 90% confidence interval
  - Panel 2: Model training loss (Gradient Boosting)
  - Panel 3: Scenario analysis fan chart (10,000 paths per scenario)
  - Panel 4: Sunday 12 AM UTC price distribution by scenario

- **forecast_report.txt** - Detailed text report with:
  - Current price and regime
  - Sunday forecast with confidence intervals
  - Scenario probabilities (Bull/Base/Bear)
  - Probability of various price moves (+5%, +10%, -5%)

## Model Architecture

### 1. Data Processing
- Fetch 24/7 hourly BTC-USD data (365+ days history)
- Engineer 10 technical features:
  - Volatility (12h, 24h rolling)
  - Momentum (RSI, MACD)
  - Bands (Bollinger %B, ATR)
  - Trend (EMA stack: 7/21/50/200)
  - Volume normalization
  - HMM regime classification

### 2. Regime Detection (HMM)
- 3-state Gaussian HMM on returns + volatility
- Identifies market regimes:
  - **BULL**: High positive returns, lower volatility
  - **CHOP**: Neutral regime, moderate vol
  - **BEAR**: Negative returns, higher volatility

### 3. Ensemble Model
- **Gradient Boosting Regressor** (scikit-learn):
  - 100 estimators, learning rate 0.05
  - Depth 5, predicts average hourly log-return over 24h horizon
  - Trained on 80% of data with 10% validation holdout

### 4. Uncertainty Quantification
- **Bootstrap Resampling**: 200 bootstrap samples with fresh models
- **Conformal Prediction Bands**: 90% coverage guarantees on validation set
- **Monte Carlo Paths**: 10,000 scenarios (Bull/Base/Bear events)

### 5. Scenario Analysis
Three market scenarios with event catalysts:

| Scenario | Probability | Shock | Event |
|----------|------------|-------|-------|
| BULL | 35% | +4% | ETF approval, macro catalyst |
| BASE | 40% | 0% | No major catalyst |
| BEAR | 25% | -6% | Regulatory headwinds |

Each scenario runs 10,000 paths to Sunday 12 AM UTC with:
- Pre-event drift from model forecast
- Event shock applied at random time
- Post-event follow-through volatility

## Forecast Target

**When**: Next Sunday 12:00 AM UTC

The script calculates:
- Current time in UTC
- Hours until next Sunday 12:00 AM UTC
- Forecast index (clamped to 24-hour horizon)
- Price prediction with uncertainty bands

## Example Output

```
┌─────────────────────────────────────────────────────────┐
│ BITCOIN FORECAST: Next Sunday 12:00 AM UTC            │
├─────────────────────────────────────────────────────────┤
│ Current price:        $67,500.00                        │
│ Mean forecast:        $68,200.00 (+1.04%)               │
│ 90% Conf. interval:   $66,800.00 - $69,600.00           │
│ MC Dropout bounds:    $66,200.00 - $70,100.00           │
│ Regime:               BULL                              │
└─────────────────────────────────────────────────────────┘

Scenario Analysis Results:

BULL: ETF approval / macro catalyst (prob=35%)
  Mean: $69,100.00 | Median: $69,050.00
  P(+5%): 32.1% | P(+10%): 8.5%
  P(-5%): 2.3%

BASE: No major catalyst (prob=40%)
  Mean: $68,200.00 | Median: $68,150.00
  P(+5%): 15.2% | P(+10%): 2.1%
  P(-5%): 8.7%

BEAR: Regulatory headwinds (prob=25%)
  Mean: $63,900.00 | Median: $63,850.00
  P(+5%): 0.5% | P(+10%): 0.0%
  P(-5%): 62.3%

PROBABILITY-WEIGHTED COMPOSITE:
 Mean: $67,850.00
 P(+5%): 16.2%
 P(+10%): 3.8%
 P(-5%): 21.4%
```

## Architecture Diagram

```
┌──────────────────────┐
│   BTC-USD Data       │  ← yfinance (hourly)
│   (365+ days)        │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Feature Engineering │
│  - Technical         │  → 10 engineered features
│  - Volatility        │
│  - EMAs              │
└──────────┬───────────┘
           │
      ┌────┴─────────────────────┐
      ▼                          ▼
┌──────────────┐        ┌──────────────┐
│ HMM Regime   │        │ Ensemble     │
│ Detection    │        │ Model        │
│ (3-state)    │        │ (GB Regr.)   │
└──────────┬───┘        └──────────┬───┘
           │                       │
           ▼                       ▼
        ┌──────────────────────────┴─────────────────┐
        │                                            │
        ▼                                            ▼
┌────────────────────┐              ┌────────────────────┐
│  Conformal         │              │  Bootstrap         │
│  Prediction Bands  │              │  Uncertainty       │
│  (90% coverage)    │              │  (200 resamples)   │
└────────────────────┘              └────────────────────┘
        │
        ▼
┌────────────────────────────────────────┐
│  Monte Carlo Scenario Simulation       │
│  - BULL scenario: 10,000 paths         │
│  - BASE scenario: 10,000 paths         │
│  - BEAR scenario: 10,000 paths         │
│  Target: Next Sunday 12 AM UTC         │
└────────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────────┐
│  Output Report & Visualization         │
│  - Forecast price + intervals          │
│  - Scenario probabilities              │
│  - Price distribution                  │
│  - 4-panel chart                       │
└────────────────────────────────────────┘
```

## Requirements

- Python 3.11+
- Dependencies: numpy, pandas, scikit-learn, scipy, matplotlib, hmmlearn, yfinance, pytz
- Docker (for containerized execution)

## Customization

Edit `btc_forecast.py` to:

- **Change prediction target**: Modify UTC calculation in Section 5
- **Adjust lookback window**: Change `LOOKBACK_HOURS` (default: 168 = 1 week)
- **Modify scenarios**: Update `SCENARIOS` dict in Section 6
- **Adjust model parameters**: Edit `GradientBoostingRegressor` kwargs in Section 3

## Technical Details

### Why Ensemble over Deep Learning?

This implementation uses scikit-learn's Gradient Boosting instead of TensorFlow/Keras because:

1. **Compatibility**: Works with Python 3.14+ (TensorFlow doesn't)
2. **Speed**: Trains in minutes vs. hours for deep models
3. **Interpretability**: Feature importance, decision paths
4. **Data efficiency**: Works well with limited historical data
5. **Stability**: Less prone to overfitting than RNNs/LSTMs

### Confidence Intervals

Predictions include three uncertainty layers:

1. **Model Uncertainty**: Bootstrap resampling of training data
2. **Conformal Bands**: 90% guaranteed coverage via validation quantiles
3. **Scenario Range**: 10th to 90th percentile across 10k simulation paths

## References

- Angeletos, G. M., & Lian, C. (2021). Confidence and the business cycle.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). Elements of Statistical Learning.
- Hamilton, J. D. (1989). A new approach to the economic analysis of nonstationary time series.

## License

MIT

## Author

Quantitative Analysis Stack | Bitcoin Price Forecaster
