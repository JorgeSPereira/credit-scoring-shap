.PHONY: install format lint test clean train train-quick api mlflow fairness all

# Install dependencies
install:
	poetry install

# Format code
format:
	poetry run black src/ tests/
	poetry run isort src/ tests/

# Lint code
lint:
	poetry run flake8 src/ tests/ --max-line-length=88 --extend-ignore=E203,D100,D104
	poetry run mypy src/ --ignore-missing-imports

# Run tests
test:
	poetry run pytest tests/ -v --cov=src/credit_scoring

# Clean artifacts
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	rm -rf .pytest_cache .coverage htmlcov mlruns

# Train model with Optuna optimization (default)
train:
	poetry run python -m credit_scoring.models.train

# Train model without Optuna (faster)
train-quick:
	poetry run python -m credit_scoring.models.train --no-optuna

# Start API server
api:
	poetry run uvicorn credit_scoring.api.main:app --reload --host 0.0.0.0 --port 8000

# Start MLflow UI
mlflow:
	poetry run mlflow ui --port 5000

# Run fairness notebook
fairness:
	poetry run jupyter nbconvert --execute notebooks/04_fairness.ipynb --to notebook --inplace

# Run full pipeline
all: format lint test train

