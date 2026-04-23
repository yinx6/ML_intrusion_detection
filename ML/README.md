# IDS-ML: Network Intrusion Detection with Machine Learning

This project provides a practical baseline implementation for an intrusion detection system (IDS) based on machine learning, aligned with your project specification:

- Supervised detection of known attacks (Random Forest, XGBoost)
- Unsupervised anomaly detection (Isolation Forest)
- Reproducible preprocessing and evaluation pipeline
- MLflow experiment tracking
- FastAPI prediction endpoint
- Streamlit demo dashboard

## Project Structure

- `src/ids_ml/data_loader.py`: dataset loading and target handling
- `src/ids_ml/feature_engineering.py`: preprocessing pipeline
- `src/ids_ml/models/`: model builders
- `src/ids_ml/evaluation.py`: metrics and cross-validation
- `src/ids_ml/tracking.py`: MLflow helpers
- `src/ids_ml/pipelines/train.py`: end-to-end training command
- `src/ids_ml/api/main.py`: REST API for inference
- `src/ids_ml/dashboard/app.py`: Streamlit UI
- `scripts/train.py`: CLI entrypoint

## Quick Start

1. Create environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Prepare data CSV:
- Place your CSV (CICIDS / UNSW / NSL-KDD exported as tabular file) in `../Dataset` or any path.
- Ensure there is a target column (`Label` by default).

3. Train baseline models:

```bash
python scripts/train.py --data ../Dataset/your_file.csv --target Label
```

4. Launch API:

```bash
uvicorn ids_ml.api.main:app --reload --app-dir src
```

5. Launch dashboard:

```bash
streamlit run src/ids_ml/dashboard/app.py
```

## Notes

- For very large datasets, start with a sampled subset to validate the pipeline.
- Use `--sample-size` in training script to accelerate first runs.
- Artifacts are saved to `artifacts/`.
