from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import mlflow
import mlflow.sklearn

logger = logging.getLogger(__name__)


def setup_mlflow(tracking_uri: str | None = None, experiment_name: str = "ids-ml") -> None:
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def log_run(
    model_name: str,
    model_object: object,
    metrics: Dict[str, float],
    params: Dict[str, object],
    artifact_path: Path,
) -> None:
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        # Bug 9 fix: log the raw joblib artifact AND register the model properly
        # via mlflow.sklearn so it appears in the MLflow Model Registry for inference.
        mlflow.log_artifact(str(artifact_path))
        try:
            mlflow.sklearn.log_model(model_object, artifact_path=model_name)
        except Exception as exc:
            logger.warning("mlflow.sklearn.log_model failed for %s: %s", model_name, exc)
