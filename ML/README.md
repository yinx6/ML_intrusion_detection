# IDS-ML — Network Intrusion Detection with Machine Learning

An end-to-end machine learning pipeline for network intrusion detection, built on the **CIC-IDS 2018** dataset.

---

## Features

| Component | Details |
|---|---|
| **Supervised models** | Random Forest, XGBoost, MLP (scikit-learn / XGBoost) |
| **Unsupervised model** | Isolation Forest (anomaly detection) |
| **Preprocessing** | StandardScaler · OneHotEncoder · SMOTE / RandomOverSampler |
| **Evaluation** | Stratified 5-fold CV (train set) + held-out test set (20 %) |
| **Metrics** | F1-Macro · Precision · Recall · False Positive Rate (macro) |
| **Experiment tracking** | MLflow (params, metrics, model registry) |
| **Inference API** | FastAPI REST endpoint (`POST /predict`) |
| **Dashboard** | Streamlit model-comparison UI |

---

## Project Structure

```
ML/
├── scripts/
│   └── train.py                  # CLI entry point
├── src/ids_ml/
│   ├── __init__.py
│   ├── data_loader.py            # CSV loading, sanitisation, train/test split
│   ├── feature_engineering.py   # ColumnTransformer (scaler + encoder)
│   ├── evaluation.py            # Cross-validation & metrics
│   ├── tracking.py              # MLflow helpers
│   ├── models/
│   │   ├── supervised.py        # Random Forest, XGBoost, MLP builders
│   │   └── unsupervised.py      # Isolation Forest builder
│   ├── pipelines/
│   │   └── train.py             # Full train_all() pipeline
│   ├── api/
│   │   └── main.py              # FastAPI application
│   └── dashboard/
│       └── app.py               # Streamlit dashboard
├── artifacts/                   # Saved models (.joblib) & metrics CSV
├── mlruns/                      # MLflow experiment data
└── requirements.txt
```

---

## Quick Start

### 1 — Install dependencies

```bash
cd ML
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS
pip install -r requirements.txt
```

### 2 — Prepare the dataset

Place CIC-IDS 2018 CSV files inside `../Dataset/`:

```
SFE/
├── Dataset/
│   ├── 02-14-2018.csv
│   ├── 02-15-2018.csv
│   └── ...
└── ML/
```

Each CSV must contain a `Label` column (or specify a custom column with `--target`).

### 3 — Train the models

```bash
python scripts/train.py --data ../Dataset/02-14-2018.csv
```

The pipeline will:
1. Load and sanitise the CSV
2. **Split 80 % train / 20 % test** (stratified, reproducible)
3. Run **5-fold cross-validation** on the train set
4. **Fit** the final pipeline on the full train set
5. **Evaluate** on the held-out test set
6. Save models to `artifacts/` and log everything to MLflow

---

## Training CLI Reference

```
python scripts/train.py [OPTIONS]
```

| Flag | Default | Description |
|---|---|---|
| `--data` | *(required)* | Path to the CSV dataset |
| `--target` | `Label` | Target column name |
| `--output-dir` | `artifacts` | Directory to save models & metrics |
| `--sample-size` | `None` | Limit rows for fast experiments |
| `--test-size` | `0.2` | Fraction of data held out for testing |
| `--split-seed` | `42` | Random seed for the train/test split |
| `--tracking-uri` | `None` | MLflow tracking URI (uses local `mlruns/` by default) |

**Examples:**

```bash
# Standard run (20 % test holdout)
python scripts/train.py --data ../Dataset/02-14-2018.csv

# Quick smoke-test on 50 000 rows
python scripts/train.py --data ../Dataset/02-14-2018.csv --sample-size 50000

# Custom split with a different seed
python scripts/train.py --data ../Dataset/02-14-2018.csv --test-size 0.15 --split-seed 7
```

---

## Output Metrics

Each trained model produces two sets of metrics:

| Prefix | Source | Description |
|---|---|---|
| `cv_*` | 5-fold CV on train set | Cross-validated estimate of generalisation |
| `test_*` | Held-out test set | Final honest evaluation on unseen data |

Metrics saved to `artifacts/metrics_summary.csv` and logged to MLflow:

- `cv_f1_macro` / `test_f1_macro`
- `cv_precision_macro` / `test_precision_macro`
- `cv_recall_macro` / `test_recall_macro`
- `cv_false_positive_rate_macro`

---

## Launching the API

The API loads a trained model from `artifacts/` and exposes a `/predict` endpoint.

```bash
uvicorn ids_ml.api.main:app --reload --app-dir src
```

**Environment variables** (optional overrides):

| Variable | Default | Description |
|---|---|---|
| `IDS_MODEL_PATH` | `artifacts/random_forest_pipeline.joblib` | Path to the model pipeline |
| `IDS_ENCODER_PATH` | `artifacts/random_forest_label_encoder.joblib` | Path to the label encoder |

**Endpoints:**

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Returns `{"status": "ok", "model_loaded": true/false}` |
| `POST` | `/predict` | Classifies traffic records |

**Example request:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"records": [{"feature_1": 0.5, "feature_2": 100, ...}]}'
```

Interactive docs available at: http://localhost:8000/docs

---

## Launching the Dashboard

```bash
streamlit run src/ids_ml/dashboard/app.py
```

Displays a comparison table and bar chart of model F1-Macro scores from `artifacts/metrics_summary.csv`. Run training first to populate the data.

---

## Viewing MLflow Experiments

```bash
mlflow ui --backend-store-uri mlruns
```

Then open http://localhost:5000 to browse runs, compare hyperparameters, and download logged models.

---

## Notes

- For very large CSV files (e.g. `02-20-2018.csv` at 4 GB), start with `--sample-size 100000` to validate the pipeline before a full run.
- Artifacts directory keeps a `.gitkeep` file so the folder is tracked by Git even when empty (models are in `.gitignore`).
- All random seeds are fixed to `42` by default for full reproducibility.
