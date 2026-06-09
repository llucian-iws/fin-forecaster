FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Unbuffer stdout so progress past the training bar is visible in real time
ENV PYTHONUNBUFFERED=1

# Copy prediction scripts
COPY btc_forecast.py .
COPY btc_forecast_lite.py .
COPY volatility.py .
COPY backtest.py .
COPY exogenous.py .
COPY forecast_post.py .
COPY metrics.py .

# Create results directory with proper permissions for volume mount
RUN mkdir -p /app/results && chmod 777 /app/results

# Default command (full sophisticated version)
CMD ["python", "btc_forecast.py"]
