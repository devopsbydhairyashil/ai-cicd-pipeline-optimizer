# AI-Powered CI/CD Optimization
**Repository purpose:** A small, production-oriented example that demonstrates how to collect CI logs, train a lightweight model
to predict build failures, and integrate predictions into the CI pipeline to reduce broken builds.

## What this is
This project contains:
- A data pipeline that converts raw CI logs into features (`src/ingest_logs.py`)
- A simple ML trainer (`src/train_model.py`) using scikit-learn to predict build success/failure
- A prediction service (`src/predict_service.py`) that returns a probability a build will fail
- GitHub Actions workflow that uses the prediction service before running expensive tests
- Dockerfile and Terraform placeholders for deployment

> The README is written from the perspective of an engineer who actually implemented and maintains the project.

## Quick start (local)
1. Create a virtualenv and install:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
2. Generate synthetic CI data and train:
```bash
python src/generate_sample_data.py
python src/train_model.py --data data/ci_sample.csv --out model/model.pkl
```
3. Run the prediction service:
```bash
python src/predict_service.py --model model/model.pkl --port 8000
```
4. Simulate a pre-check in CI:
```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"commit":"abc123","duration":340,"tests_run":120,"failures_last_24h":2}'
```

## Files of interest
- `src/ingest_logs.py` — example parser that extracts features from raw logs.
- `src/train_model.py` — preprocess + train a RandomForestClassifier.
- `src/predict_service.py` — lightweight Flask service for predictions.
- `.github/workflows/precheck.yml` — CI workflow that queries the prediction service before running heavy steps.

## Design notes & realism tweaks
- Feature engineering is intentionally pragmatic: build duration, tests run, failure trend windows.
- The pre-check gate is advisory by default; it can be configured to fail builds when predicted risk > 0.7.
- The model is small and explainable; a `model/feature_importances.txt` file is created by training to justify alerts.

## License
MIT — see LICENSE file.
