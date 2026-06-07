# Target Date/Time Configuration Guide

## Overview

The Bitcoin price forecasters (`btc_forecast.py` and `btc_forecast_lite.py`) now support configurable target dates and times. By default, both predict for **next Wednesday at 00:00 UTC**, but you can specify any target date and time.

## Quick Examples

### Default (Wednesday Midnight UTC)
```bash
docker run --rm -v $(pwd)/results:/app/results fin-forecaster:latest \
  python btc_forecast_lite.py
```

### Wednesday 12:00 PM UTC
```bash
docker run --rm -v $(pwd)/results:/app/results fin-forecaster:latest \
  python btc_forecast_lite.py --target-date next-wednesday --target-hour 12
```

### Friday 18:00 UTC (6 PM)
```bash
docker run --rm -v $(pwd)/results:/app/results fin-forecaster:latest \
  python btc_forecast_lite.py --target-date next-friday --target-hour 18
```

### Specific Date (June 15, 2026 at 14:00 UTC)
```bash
docker run --rm -v $(pwd)/results:/app/results fin-forecaster:latest \
  python btc_forecast_lite.py --target-date 2026-06-15 --target-hour 14
```

## Usage Methods

### Method 1: Command-Line Arguments

```bash
python btc_forecast.py --target-date <date> --target-hour <hour>
python btc_forecast_lite.py --target-date <date> --target-hour <hour>
```

**Arguments:**
- `--target-date`: Target date specification (optional, defaults to next-wednesday)
  - Format: `next-<day>` (e.g., `next-monday`, `next-wednesday`, `next-friday`)
  - Format: `YYYY-MM-DD` (e.g., `2026-06-15`)
  - Valid days: monday, tuesday, wednesday, thursday, friday, saturday, sunday

- `--target-hour`: Hour in UTC 0-23 (optional, defaults to 0 for midnight)
  - Examples: `0` (midnight), `12` (noon), `18` (6 PM)

### Method 2: Environment Variables

```bash
export TARGET_DATE=next-wednesday
export TARGET_HOUR=12
python btc_forecast.py
```

Or as one-liners:

```bash
TARGET_DATE=next-friday TARGET_HOUR=18 python btc_forecast_lite.py
```

### Method 3: Docker with Environment Variables

```bash
docker run --rm \
  -e TARGET_DATE=next-wednesday \
  -e TARGET_HOUR=12 \
  -v $(pwd)/results:/app/results \
  fin-forecaster:latest
```

### Method 4: Docker Compose Override

Edit `docker-compose.yml`:

```yaml
services:
  btc-forecaster:
    environment:
      - TARGET_DATE=next-wednesday
      - TARGET_HOUR=12
```

Then run:

```bash
docker-compose up
```

## Date Format Examples

### Relative Day Format (Recommended)
```
next-monday    → Next Monday at specified hour
next-tuesday   → Next Tuesday at specified hour
next-wednesday → Next Wednesday at specified hour
next-thursday  → Next Thursday at specified hour
next-friday    → Next Friday at specified hour
next-saturday  → Next Saturday at specified hour
next-sunday    → Next Sunday at specified hour
```

### Absolute Date Format
```
2026-06-10     → June 10, 2026 at specified hour
2026-06-15     → June 15, 2026 at specified hour
2026-12-25     → December 25, 2026 at specified hour
```

## Time Examples (UTC)

```
--target-hour 0   → 00:00 UTC (midnight)
--target-hour 6   → 06:00 UTC
--target-hour 12  → 12:00 UTC (noon)
--target-hour 18  → 18:00 UTC (6 PM)
--target-hour 23  → 23:00 UTC
```

## Complete Examples

### CNN-LSTM Full Stack - Saturday 9 AM UTC
```bash
docker run --rm \
  -v $(pwd)/results:/app/results \
  fin-forecaster:latest \
  python btc_forecast.py --target-date next-saturday --target-hour 9
```

### Lite Version - Thursday Midnight
```bash
docker run --rm \
  -v $(pwd)/results:/app/results \
  fin-forecaster:latest \
  python btc_forecast_lite.py --target-date next-thursday --target-hour 0
```

### Specific Date - May 1, 2026 at 3 PM UTC
```bash
docker run --rm \
  -v $(pwd)/results:/app/results \
  fin-forecaster:latest \
  python btc_forecast_lite.py --target-date 2026-05-01 --target-hour 15
```

### With Docker Compose and Custom Command
```bash
docker-compose run --rm btc-forecaster \
  python btc_forecast.py --target-date next-friday --target-hour 12
```

## Output Format

The forecast report shows the target in ISO 8601 format:

```
Target: Wednesday 12:00 UTC (2026-06-10)
Hours until target: 62.0
```

## API Response

When invoked via environment variables or arguments, both forecasters return output in the format:

```
Target: <Day> <Hour>:00 UTC (<Date>)
Hours until target: <Hours>
```

## Error Handling

### Invalid Day Name
```bash
$ python btc_forecast.py --target-date next-funday
Unknown day: funday. Using next-wednesday.
```

### Invalid Date Format
```bash
$ python btc_forecast.py --target-date 06/10/2026
Invalid date format: 06/10/2026. Using next-wednesday.
```

### Invalid Hour
```bash
--target-hour 25  # Will be treated as 25:00 UTC (invalid, but Python handles gracefully)
```

## Defaults

If no parameters are specified:
- **Target Date**: Next Wednesday
- **Target Hour**: 00:00 UTC (midnight)
- **Example**: Wednesday 2026-06-10 00:00 UTC (50 hours from current time)

## Use Cases

1. **Trading Window Forecasts**: Predict price at market open/close times
   ```bash
   --target-date next-monday --target-hour 9  # NY market open
   ```

2. **Weekly Reports**: Consistent forecasts for Fridays
   ```bash
   --target-date next-friday --target-hour 16  # End of trading day
   ```

3. **Scheduled Predictions**: Automate via cron
   ```bash
   0 12 * * * docker run ... --target-date next-sunday --target-hour 0
   ```

4. **Multi-timeframe Analysis**: Generate forecasts for multiple targets
   ```bash
   for day in monday wednesday friday; do
     docker run ... --target-date next-$day --target-hour 12
   done
   ```
