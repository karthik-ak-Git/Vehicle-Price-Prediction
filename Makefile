# Makefile for Vehicle Price Prediction

.PHONY: help install install-dev test lint format clean run-api run-ui docker-build benchmark

# Default target
help:
	@echo "Vehicle Price Prediction - Make Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install        Install production dependencies"
	@echo "  make install-dev    Install development dependencies"
	@echo ""
	@echo "Development:"
	@echo "  make format         Format code with black and isort"
	@echo "  make lint           Run linting (flake8, mypy)"
	@echo "  make test           Run test suite"
	@echo "  make test-cov       Run tests with coverage report"
	@echo ""
	@echo "Running:"
	@echo "  make run-api        Start FastAPI server"
	@echo "  make run-ui         Start Streamlit dashboard"
	@echo "  make train          Train the model"
	@echo "  make evaluate       Evaluate the model"
	@echo ""
	@echo "Performance:"
	@echo "  make benchmark      Run performance benchmarks"
	@echo "  make load-test      Run load tests with locust"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build   Build Docker images"
	@echo "  make docker-up      Start services with docker-compose"
	@echo "  make docker-down    Stop services"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean          Clean temporary files"
	@echo "  make clean-all      Clean all generated files"

# Installation
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install pytest pytest-cov black flake8 isort mypy pre-commit locust
	pre-commit install

# Code Quality
format:
	@echo "Formatting code..."
	black .
	isort .
	@echo "✓ Code formatted"

lint:
	@echo "Running linters..."
	flake8 .
	mypy . --install-types --non-interactive || true
	@echo "✓ Linting complete"

# Testing
test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=. --cov-report=html --cov-report=term-missing

# Data Processing and Training
process-data:
	python data/dataloader.py --dataset_dir dataset/ --out outputs/

train:
	python train.py

evaluate:
	python evaluate.py

# Running Services
run-api:
	uvicorn api_app:app --host 0.0.0.0 --port 8000 --reload

run-ui:
	streamlit run streamlit_app.py

# Performance Testing
benchmark:
	python benchmark.py

load-test:
	locust -f performance_test.py --host=http://localhost:8000 --headless -u 10 -r 2 -t 60s

# Docker
docker-build:
	docker build -t vehicle-price-prediction:latest -f Dockerfile .
	docker build -t vehicle-price-prediction-streamlit:latest -f Dockerfile.streamlit .

docker-up:
	docker-compose up --build

docker-down:
	docker-compose down

# Cleaning
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/
	rm -rf .coverage
	@echo "✓ Cleaned temporary files"

clean-all: clean
	rm -rf venv/
	rm -rf outputs/*.pkl outputs/*.joblib outputs/*.json outputs/*.csv outputs/*.png
	rm -rf models/*.pkl
	rm -rf logs/*.log
	@echo "✓ Cleaned all generated files"

# Pre-commit
pre-commit:
	pre-commit run --all-files

# Full pipeline
all: format lint test

# CI simulation
ci: format lint test-cov
	@echo "✓ CI pipeline complete"
