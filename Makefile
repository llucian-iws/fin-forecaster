.PHONY: help install run docker-build docker-run clean

help:
	@echo "Bitcoin Price Forecaster - Available Commands"
	@echo "=============================================="
	@echo "  make install      - Install dependencies (local Python)"
	@echo "  make run          - Run forecast (local Python)"
	@echo "  make docker-build - Build Docker image"
	@echo "  make docker-run   - Run forecast in Docker"
	@echo "  make clean        - Clean up results"

install:
	pip install -r requirements.txt

run:
	python3 btc_forecast.py

docker-build:
	docker-compose build

docker-run:
	docker-compose up --build

clean:
	rm -rf results/*.png results/*.txt results/__pycache__
	mkdir -p results
