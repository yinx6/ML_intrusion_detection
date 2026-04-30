# SFE — Network Intrusion Detection with Machine Learning

End-to-end machine learning project for network intrusion detection using the **CIC-IDS 2018** dataset.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![MLflow](https://img.shields.io/badge/Tracking-MLflow-orange)](https://mlflow.org/)
[![Dataset](https://img.shields.io/badge/Dataset-CIC--IDS%202018-green)](https://www.unb.ca/cic/datasets/ids-2018.html)

---

## Repository Layout

```
SFE/
├── Dataset/          # Raw CIC-IDS 2018 CSV files (not committed — see .gitignore)
│   ├── 02-14-2018.csv
│   ├── 02-15-2018.csv
│   └── ... (10 files total, ~6.9 GB)
└── ML/               # Core ML project
    ├── scripts/      # CLI entry points
    ├── src/ids_ml/   # Source package
    │   ├── data_loader.py          # Load, clean, train/test split
    │   ├── feature_engineering.py # Preprocessing pipeline
    │   ├── evaluation.py          # CV & metrics
    │   ├── tracking.py            # MLflow integration
    │   ├── models/                # Supervised & unsupervised model builders
    │   ├── pipelines/             # End-to-end training pipeline
    │   ├── api/                   # FastAPI REST inference server
    │   └── dashboard/             # Streamlit UI
    ├── artifacts/    # Trained models & metrics (generated, not committed)
    ├── mlruns/       # MLflow experiment data (generated)
    └── requirements.txt
```

---

## What This Project Does

| Stage | Details |
|---|---|
| **Data ingestion** | Reads CIC-IDS 2018 CSVs; drops NaN/Inf rows and constant columns |
| **Train/test split** | Stratified 80/20 split — test set is never seen during training |
| **Preprocessing** | StandardScaler (numeric) + OneHotEncoder (categorical) per-model |
| **Class balancing** | SMOTE (≥6 samples/class) or RandomOverSampler fallback |
| **Supervised models** | Random Forest · XGBoost · MLP — trained and compared |
| **Unsupervised** | Isolation Forest for anomaly detection |
| **Evaluation** | 5-fold Stratified CV on train set + final evaluation on held-out test |
| **Metrics** | F1-Macro · Precision · Recall · False Positive Rate |
| **Tracking** | All runs logged to MLflow (params, metrics, model artifacts) |
| **Inference** | FastAPI `POST /predict` endpoint |
| **UI** | Streamlit dashboard for metric comparison |

---

## Getting Started

### 1 — Clone & set up environment

```bash
git clone https://github.com/yinx6/ML_intrusion_detection.git
cd ML_intrusion_detection/ML

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

pip install -r requirements.txt
```

### 2 — Add the dataset

Download the [CIC-IDS 2018 dataset](https://www.unb.ca/cic/datasets/ids-2018.html) and place the CSV files in `../Dataset/` (relative to the `ML/` directory). The folder is excluded from Git via `.gitignore`.

### 3 — Train models

```bash
python scripts/train.py --data ../Dataset/02-14-2018.csv
```

For a quick test on a subset:

```bash
python scripts/train.py --data ../Dataset/02-14-2018.csv --sample-size 50000
```

### 4 — Launch the API

```bash
uvicorn ids_ml.api.main:app --reload --app-dir src
# → http://localhost:8000/docs
```

### 5 — Launch the dashboard

```bash
streamlit run src/ids_ml/dashboard/app.py
# → http://localhost:8501
```

### 6 — Browse MLflow runs

```bash
mlflow ui --backend-store-uri ML/mlruns
# → http://localhost:5000
```

---

## Documentation

Full usage details, CLI reference, and API documentation are in **[ML/README.md](ML/README.md)**.

---

## Dataset

| Property | Value |
|---|---|
| Name | CIC-IDS 2018 |
| Source | Canadian Institute for Cybersecurity |
| Size | ~6.9 GB (10 CSV files) |
| Target column | `Label` |
| Attack types | DoS, DDoS, Bruteforce, Infiltration, Bot, Web Attacks |

> **Note:** Dataset files are not committed to Git. Add them manually to `Dataset/`.
