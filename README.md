# SFE: Network Intrusion Detection with Machine Learning

This repository contains an end-to-end machine learning project for network intrusion detection.

## Structure

- **`ML/`**: Contains the core machine learning pipeline, API, and dashboard.
  - See [`ML/README.md`](ML/README.md) for detailed instructions on training, launching the API, and running the dashboard.
- **`Dataset/`**: (Ignored in Git) Directory for large raw dataset files like CSVs.

## Overview

This project provides a practical baseline implementation for an intrusion detection system (IDS) using machine learning techniques:
- **Supervised Detection**: Random Forest, XGBoost
- **Unsupervised Anomaly Detection**: Isolation Forest
- **Experiment Tracking**: MLflow
- **Serving**: FastAPI prediction endpoint
- **UI**: Streamlit demo dashboard

## Getting Started

Navigate to the `ML/` directory and install the requirements:

```bash
cd ML
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

For full instructions, refer to the [ML README](ML/README.md).
