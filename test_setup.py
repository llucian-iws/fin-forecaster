#!/usr/bin/env python3
"""Quick test to verify setup"""
import sys
print("Testing imports...")
try:
    import numpy as np
    print("✓ numpy")
    import pandas as pd
    print("✓ pandas")
    import yfinance as yf
    print("✓ yfinance")
    from sklearn.ensemble import GradientBoostingRegressor
    print("✓ scikit-learn")
    from hmmlearn.hmm import GaussianHMM
    print("✓ hmmlearn")
    import matplotlib.pyplot as plt
    print("✓ matplotlib")
    import pytz
    print("✓ pytz")

    print("\nFetching BTC data...")
    df = yf.download('BTC-USD', start='2024-01-01', interval='1h', progress=False)
    print(f"✓ Downloaded {len(df)} hourly candles")
    print(f"  Price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")
    print(f"  Latest: ${df['Close'].iloc[-1]:.2f}")

    print("\nSetup verified! Ready to run btc_forecast.py")
    sys.exit(0)
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
