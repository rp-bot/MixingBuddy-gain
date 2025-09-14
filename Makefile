# Makefile for LLM LoRA Fine-tuning Project

.PHONY: help install install-dev clean test lint format type-check train evaluate infer prepare-data

# Default target
help:
	@echo "Available targets:"
	@echo "  install       Install production dependencies"
	@echo "  install-dev   Install development dependencies"
	@echo "  clean         Clean up generated files"
	@echo "  test          Run tests"
	@echo "  lint          Run linting checks"
	@echo "  format        Format code with black and isort"
	@echo "  type-check    Run type checking with mypy"
	@echo "  train         Train the model"
	@echo "  train-qwen2-audio  Train with Qwen2-Audio model"
	@echo "  evaluate      Evaluate the model"
	@echo "  evaluate-qwen2-audio  Evaluate Qwen2-Audio model"
	@echo "  infer         Run inference"
	@echo "  infer-qwen2-audio  Run inference with Qwen2-Audio model"
	@echo "  prepare-data  Prepare training data"
	@echo "  create-musdb-dataset  Create MUSDB18 mixing dataset"
	@echo "  create-musdb-test  Create test MUSDB18 dataset (5 tracks)"
	@echo "  create-multi-input-dataset  Create multi-input dataset for Phase 2"
	@echo "  create-multi-input-test  Create test multi-input dataset (5 tracks)"

# Installation
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

setup: install-dev
	@echo "Development environment set up successfully!"

# Code quality
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	rm -rf outputs/
	rm -rf cache/
	rm -rf logs/

test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

test-fast:
	pytest tests/ -v --cov=src --cov-report=term-missing -x

test-coverage:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing --cov-report=xml

test-unit:
	pytest tests/ -v --cov=src --cov-report=term-missing -m "not slow and not gpu and not integration"

test-integration:
	pytest tests/ -v --cov=src --cov-report=term-missing -m "integration"

test-gpu:
	pytest tests/ -v --cov=src --cov-report=term-missing -m "gpu"

test-slow:
	pytest tests/ -v --cov=src --cov-report=term-missing -m "slow"

lint:
	flake8 src scripts tests --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 src scripts tests --count --exit-zero --max-complexity=10 --max-line-length=88 --statistics

format:
	black src scripts tests
	isort src scripts tests

type-check:
	mypy src --ignore-missing-imports


# Data operations
prepare-data:
	python scripts/prepare_data.py

create-musdb-dataset:
	python scripts/create_musdb_dataset.py

create-musdb-test:
	python scripts/create_musdb_dataset.py --limit-tracks 5

create-multi-input-dataset:
	python scripts/create_musdb_dataset.py --config configs/data/musdb_dataset.yaml

create-multi-input-test:
	python scripts/create_musdb_dataset.py --config configs/data/musdb_dataset.yaml --limit-tracks 5

# Training and evaluation
train:
	python scripts/train.py

train-qwen2-audio:
	python scripts/train.py model=qwen2_audio_7b

evaluate:
	python scripts/evaluate.py

evaluate-qwen2-audio:
	python scripts/evaluate.py model=qwen2_audio_7b

infer:
	python scripts/inference.py

infer-qwen2-audio:
	python scripts/inference.py model=qwen2_audio_7b

# Model operations
train-lora-7b:
	python scripts/train.py model=llama2_7b

train-lora-13b:
	python scripts/train.py model=llama2_13b

evaluate-wandb:
	python scripts/evaluate.py experiment_tracking=wandb

evaluate-mlflow:
	python scripts/evaluate.py experiment_tracking=mlflow

# Development helpers
jupyter:
	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

notebook:
	jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# Docker operations (if using Docker)
docker-build:
	docker build -t llm-lora-finetuning .

docker-run:
	docker run --gpus all -it llm-lora-finetuning

# Environment setup
env-setup:
	python -m venv venv
	@echo "Virtual environment created. Activate with: source venv/bin/activate"

# Quick development workflow
dev-setup: env-setup
	@echo "Activating virtual environment..."
	@echo "Run: source venv/bin/activate && make install-dev"

# Full pipeline
pipeline: prepare-data train evaluate

# CI/CD simulation
ci-test: lint type-check test-unit

# Test coverage analysis
coverage-report:
	pytest tests/ --cov=src --cov-report=html --cov-report=term-missing
	@echo "Coverage report generated in htmlcov/index.html"

coverage-clean:
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf coverage.xml

# Documentation
docs:
	@echo "Generating documentation..."
	@echo "Documentation would be generated here"

# Model serving (placeholder)
serve:
	@echo "Starting model server..."
	@echo "Model serving would be implemented here"

# Monitoring (placeholder)
monitor:
	@echo "Starting monitoring..."
	@echo "Monitoring would be implemented here"
