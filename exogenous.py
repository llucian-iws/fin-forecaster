#!/usr/bin/env python3
"""
Exogenous market-microstructure features for the BTC forecaster.

These are FREE, KEYLESS public REST signals from Binance USDT-M Futures with
genuine short-horizon (24h) predictive content for BTC. The single
highest-value signal is the perpetual **funding rate** (a direct, observable
read on leveraged-positioning crowding); open interest and a futures-basis
proxy round out the microstructure picture.

Lightweight by design — imports ONLY numpy / pandas / requests, mirroring
`volatility.py`. No TensorFlow, no API keys, no network at import time. Every
fetch degrades gracefully: on any network / HTTP / parse failure it prints a
"  [exogenous] ..." note and returns None so the caller can carry on.

Sources (all keyless public REST, confirmed June 2026):
  - Funding rate history:  GET https://fapi.binance.com/fapi/v1/fundingRate
      8h cadence (00:00 / 08:00 / 16:00 UTC). History goes back to the perp's
      listing (BTCUSDT ~Sept 2019). Page size capped at 1000 rows/request, so
      fetch_funding() loops startTime forward to cover `days`.
      Response: [{symbol, fundingTime (ms), fundingRate (str), markPrice (str)}].
  - Open interest history:  GET https://fapi.binance.com/futures/data/openInterestHist
      ***Only the latest ~1 month (~30 days) of history is available*** — this is
      a hard Binance-side cap, not a paging limit. Older OI cannot be backfilled
      from this endpoint at any key tier. limit max 500.
      Response: [{symbol, sumOpenInterest (str), sumOpenInterestValue (str),
                  CMCCirculatingSupply (str), timestamp (ms)}].
  - Futures basis proxy:  GET https://fapi.binance.com/fapi/v1/premiumIndex
      A point-in-time snapshot of mark vs index price (no history). basis =
      mark/index - 1; positive basis ~ contango / bullish positioning.
      Response (single symbol): {symbol, markPrice (str), indexPrice (str),
                  estimatedSettlePrice, lastFundingRate, interestRate,
                  nextFundingTime, time}.

The headline feature builder is `funding_features`, which aligns the 8h
funding series onto an arbitrary hourly index with merge_asof(direction=
'backward') so NO look-ahead leaks in (only funding known at/before each
timestamp is used).
"""

import numpy as np
import pandas as pd
import requests

FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
PREMIUM_INDEX_URL = "https://fapi.binance.com/fapi/v1/premiumIndex"
OPEN_INTEREST_HIST_URL = "https://fapi.binance.com/futures/data/openInterestHist"

_MS_PER_DAY = 24 * 60 * 60 * 1000


def fetch_funding(symbol='BTCUSDT', days=700):
    """Fetch perpetual funding-rate history from Binance USDT-M Futures (keyless).

    Funding prints every 8h. The endpoint caps each page at 1000 rows, so this
    pages startTime forward until it covers `days` (or the data runs out). For
    BTCUSDT, history reaches back to the perp's ~Sept-2019 listing.

    Returns a DataFrame [ts (tz-aware UTC), funding_rate (float)] sorted ascending
    and de-duplicated, or None on any network/HTTP/parse failure.
    """
    end_ms = int(pd.Timestamp.utcnow().timestamp() * 1000)
    start_ms = end_ms - days * _MS_PER_DAY

    rows = []
    cursor = start_ms
    # Hard cap on iterations so a pathological response can never loop forever:
    # 8h cadence => 3 rows/day; 1000 rows/page => ~333 days/page. Plenty of slack.
    max_pages = max(1, days // 300 + 5)
    try:
        for _ in range(max_pages):
            params = {
                'symbol': symbol,
                'startTime': cursor,
                'endTime': end_ms,
                'limit': 1000,
            }
            resp = requests.get(FUNDING_URL, params=params, timeout=15)
            resp.raise_for_status()
            page = resp.json()
            if not isinstance(page, list) or not page:
                break
            rows.extend(page)
            last_ft = page[-1]['fundingTime']
            if len(page) < 1000 or last_ft >= end_ms:
                break
            # Advance one ms past the last row to avoid re-fetching it.
            cursor = last_ft + 1
    except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
        print(f"  [exogenous] funding fetch failed: {exc}")
        return None

    if not rows:
        print("  [exogenous] funding fetch returned no data")
        return None

    df = pd.DataFrame(rows)
    if 'fundingTime' not in df.columns or 'fundingRate' not in df.columns:
        print("  [exogenous] funding response missing expected fields")
        return None

    df = df[['fundingTime', 'fundingRate']].copy()
    df['ts'] = pd.to_datetime(df['fundingTime'], unit='ms', utc=True)
    df['funding_rate'] = pd.to_numeric(df['fundingRate'], errors='coerce')
    df = df[['ts', 'funding_rate']].dropna()
    df = df.drop_duplicates(subset='ts').sort_values('ts').reset_index(drop=True)
    return df


def fetch_open_interest(symbol='BTCUSDT', period='1h', days=30):
    """Fetch open-interest history from Binance USDT-M Futures (keyless).

    ***HISTORY LIMITATION: Binance only serves the latest ~1 month (~30 days) of
    open-interest stats from this endpoint.*** Requests for older data simply
    return nothing — there is no key tier that unlocks more. `days` is therefore
    effectively clamped to ~30; larger values won't backfill.

    `period` is one of Binance's buckets ("5m","15m","30m","1h","2h","4h","6h",
    "12h","1d"); limit is capped at 500 rows/request, which at 1h covers ~20 days
    in a single call, so this makes a single request (no paging — older pages
    don't exist anyway).

    Returns a DataFrame [ts (tz-aware UTC), open_interest (float, contracts)]
    sorted ascending, or None on any network/HTTP/parse failure.
    """
    params = {
        'symbol': symbol,
        'period': period,
        'limit': 500,
    }
    try:
        resp = requests.get(OPEN_INTEREST_HIST_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError) as exc:
        print(f"  [exogenous] open-interest fetch failed: {exc}")
        return None

    if not isinstance(data, list) or not data:
        print("  [exogenous] open-interest fetch returned no data "
              "(note: only ~30 days of history exists)")
        return None

    df = pd.DataFrame(data)
    if 'timestamp' not in df.columns or 'sumOpenInterest' not in df.columns:
        print("  [exogenous] open-interest response missing expected fields")
        return None

    df = df[['timestamp', 'sumOpenInterest']].copy()
    df['ts'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df['open_interest'] = pd.to_numeric(df['sumOpenInterest'], errors='coerce')
    df = df[['ts', 'open_interest']].dropna()
    df = df.drop_duplicates(subset='ts').sort_values('ts').reset_index(drop=True)
    return df


def fetch_basis(symbol='BTCUSDT'):
    """Fetch a futures-basis snapshot from Binance premiumIndex (keyless).

    This is a single point-in-time read (the endpoint exposes no history):
    basis = markPrice / indexPrice - 1. Positive basis ~ contango / bullish
    leveraged positioning; negative ~ backwardation. Useful as a current-state
    feature, not a backfillable series.

    Returns a one-row DataFrame [ts (tz-aware UTC), mark_price, index_price,
    basis (float), last_funding_rate] or None on any network/HTTP/parse failure.
    """
    params = {'symbol': symbol}
    try:
        resp = requests.get(PREMIUM_INDEX_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError) as exc:
        print(f"  [exogenous] basis fetch failed: {exc}")
        return None

    # With a symbol, the endpoint returns a single object.
    if isinstance(data, list):
        data = data[0] if data else None
    if not isinstance(data, dict) or 'markPrice' not in data or 'indexPrice' not in data:
        print("  [exogenous] basis response missing expected fields")
        return None

    try:
        mark = float(data['markPrice'])
        index = float(data['indexPrice'])
    except (TypeError, ValueError) as exc:
        print(f"  [exogenous] basis parse failed: {exc}")
        return None
    if index == 0:
        print("  [exogenous] basis index price is zero; cannot compute basis")
        return None

    ts = pd.to_datetime(data.get('time', pd.Timestamp.utcnow().timestamp() * 1000),
                        unit='ms', utc=True)
    last_funding = pd.to_numeric(data.get('lastFundingRate'), errors='coerce')
    df = pd.DataFrame([{
        'ts': ts,
        'mark_price': mark,
        'index_price': index,
        'basis': mark / index - 1.0,
        'last_funding_rate': float(last_funding) if pd.notna(last_funding) else np.nan,
    }])
    return df


_FUNDING_COLUMNS = ['funding_rate', 'funding_z', 'funding_cum_24h']


def funding_features(funding_df, index):
    """Build funding-derived features aligned (look-ahead-free) onto `index`.

    Parameters
    ----------
    funding_df : pd.DataFrame or None
        Output of `fetch_funding` ([ts, funding_rate]). If None, a DataFrame of
        the right columns filled with NaN is returned (no crash).
    index : pd.DatetimeIndex
        Target (typically hourly) timestamps to align onto.

    Returns
    -------
    pd.DataFrame indexed by `index` with columns:
      - funding_rate    : last funding known at/before each ts (8h -> hourly ffill)
      - funding_z       : rolling ~30d (720h) z-score of funding_rate
      - funding_cum_24h : rolling 24h sum of funding_rate

    Alignment uses pandas.merge_asof(direction='backward'), so each timestamp
    only ever sees funding that had already printed at/before it — NO look-ahead.
    """
    idx = pd.DatetimeIndex(index)

    # Degrade gracefully: no funding -> all-NaN frame of the right shape.
    if funding_df is None or len(funding_df) == 0:
        return pd.DataFrame(np.nan, index=idx, columns=_FUNDING_COLUMNS)

    src = funding_df[['ts', 'funding_rate']].dropna().copy()
    src['ts'] = pd.to_datetime(src['ts'], utc=True)
    src = src.drop_duplicates(subset='ts').sort_values('ts').reset_index(drop=True)

    # Normalize the target index to tz-aware UTC for the merge.
    idx_utc = idx.tz_localize('UTC') if idx.tz is None else idx.tz_convert('UTC')

    # merge_asof requires the left key sorted; carry an explicit position column
    # so the result can be restored to the caller's original `index` order.
    target = (pd.DataFrame({'ts': idx_utc, '_pos': np.arange(len(idx_utc))})
              .sort_values('ts')
              .reset_index(drop=True))

    # merge_asof requires identical key dtype INCLUDING datetime resolution;
    # yfinance indices can be 's' while Binance ts parse to 'ms'. Coerce both ns.
    target['ts'] = target['ts'].astype('datetime64[ns, UTC]')
    src['ts'] = src['ts'].astype('datetime64[ns, UTC]')

    merged = pd.merge_asof(target, src, on='ts', direction='backward')

    # Compute rolling features in sorted-time order, then restore caller order.
    fr = merged['funding_rate']
    # ~30d window on an hourly index = 720h. min_periods=1 keeps early rows usable.
    roll_mean = fr.rolling(720, min_periods=1).mean()
    roll_std = fr.rolling(720, min_periods=1).std()
    merged['funding_z'] = (fr - roll_mean) / roll_std.replace(0.0, np.nan)
    merged['funding_cum_24h'] = fr.rolling(24, min_periods=1).sum()

    merged = merged.sort_values('_pos')
    out = pd.DataFrame({
        'funding_rate': merged['funding_rate'].values,
        'funding_z': merged['funding_z'].values,
        'funding_cum_24h': merged['funding_cum_24h'].values,
    }, index=idx)
    return out


if __name__ == '__main__':
    # ----------------------------------------------------------------------
    # Synthetic smoke test — NO network. Builds a tiny fake 8h funding series
    # and an hourly index, runs funding_features, and asserts shape/columns and
    # the look-ahead-free alignment behaviour.
    # ----------------------------------------------------------------------
    print("[exogenous] running synthetic smoke test (no network)...")

    # Fake funding: three 8h prints on 2024-01-01.
    fake_funding = pd.DataFrame({
        'ts': pd.to_datetime([
            '2024-01-01 00:00:00',
            '2024-01-01 08:00:00',
            '2024-01-01 16:00:00',
        ], utc=True),
        'funding_rate': [0.0001, -0.0002, 0.0003],
    })

    # Hourly index spanning the funding prints and a bit beyond.
    hourly = pd.date_range('2024-01-01 00:00:00', '2024-01-01 23:00:00',
                           freq='h', tz='UTC')

    feats = funding_features(fake_funding, hourly)

    # Shape & columns.
    assert list(feats.columns) == _FUNDING_COLUMNS, feats.columns.tolist()
    assert len(feats) == len(hourly), (len(feats), len(hourly))
    assert feats.index.equals(hourly), "index not preserved"

    # Look-ahead-free alignment: each hour carries the most recent PRIOR print.
    assert feats.loc[pd.Timestamp('2024-01-01 00:00:00', tz='UTC'), 'funding_rate'] == 0.0001
    assert feats.loc[pd.Timestamp('2024-01-01 07:00:00', tz='UTC'), 'funding_rate'] == 0.0001
    assert feats.loc[pd.Timestamp('2024-01-01 08:00:00', tz='UTC'), 'funding_rate'] == -0.0002
    assert feats.loc[pd.Timestamp('2024-01-01 15:00:00', tz='UTC'), 'funding_rate'] == -0.0002
    assert feats.loc[pd.Timestamp('2024-01-01 16:00:00', tz='UTC'), 'funding_rate'] == 0.0003

    # 24h cumulative at the last hour = sum of the three distinct prints carried
    # forward across 24 hourly rows (00-07: 0.0001 x8, 08-15: -0.0002 x8,
    # 16-23: 0.0003 x8) over a trailing 24-row window.
    last_cum = feats['funding_cum_24h'].iloc[-1]
    expected_cum = (0.0001 * 8) + (-0.0002 * 8) + (0.0003 * 8)
    assert abs(last_cum - expected_cum) < 1e-12, (last_cum, expected_cum)

    # None-input path returns the right all-NaN frame, no crash.
    nan_feats = funding_features(None, hourly)
    assert list(nan_feats.columns) == _FUNDING_COLUMNS
    assert len(nan_feats) == len(hourly)
    assert nan_feats['funding_rate'].isna().all()

    print("[exogenous] smoke test PASSED:")
    print(feats.head(10).to_string())
