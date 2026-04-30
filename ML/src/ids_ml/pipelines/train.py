
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import joblib
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.metrics import f1_score, precision_score, recall_score

from ids_ml.data_loader import load_dataset, make_split, sanitize_dataframe
from ids_ml.evaluation import evaluate_classifier
from ids_ml.feature_engineering import build_preprocessor, encode_labels
from ids_ml.models.supervised import get_supervised_models
from ids_ml.tracking import log_run, setup_mlflow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _build_sampler(y_encoded) -> object:
    class_counts = pd.Series(y_encoded).value_counts()
    min_class_count = int(class_counts.min())
    if min_class_count >= 6:
        return SMOTE(random_state=42)
    # Fallback for very rare classes where SMOTE would fail.
    return RandomOverSampler(random_state=42)


def train_all(
    data_path: str | Path,
    target_col: str = "Label",
    output_dir: str | Path = "artifacts",
    sample_size: int | None = None,
    tracking_uri: str | None = None,
    test_size: float = 0.2,
    split_seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    logger.info("Loading dataset from %s …", data_path)
    x, y = load_dataset(data_path, target_col=target_col)
    clean_x = sanitize_dataframe(x)
    y = y.loc[clean_x.index]

    if sample_size and len(clean_x) > sample_size:
        logger.info("Sampling %d rows from %d total …", sample_size, len(clean_x))
        sampled_idx = clean_x.sample(n=sample_size, random_state=42).index
        clean_x = clean_x.loc[sampled_idx]
        y = y.loc[sampled_idx]

    # Avoid mixed object dtypes that break OneHotEncoder.
    object_cols = clean_x.select_dtypes(include=["object"]).columns
    if len(object_cols) > 0:
        clean_x[object_cols] = clean_x[object_cols].astype(str)

    x = clean_x
    y_encoded, label_encoder = encode_labels(y)

    # ── Train / Test split ────────────────────────────────────────────────
    x_train, x_test, y_train, y_test = make_split(
        x, y_encoded, test_size=test_size, random_state=split_seed
    )
    logger.info(
        "Dataset split: %d train rows, %d test rows (test_size=%.2f, seed=%d)",
        len(x_train), len(x_test), test_size, split_seed,
    )

    models = get_supervised_models(n_classes=len(set(y_encoded)), random_state=42)

    setup_mlflow(tracking_uri=tracking_uri)
    all_metrics: Dict[str, Dict[str, float]] = {}

    for name, model in models.items():
        logger.info("── Training model: %s ──", name)

        # Bug 1 & 2 fix: build a fresh preprocessor AND sampler for each model.
        # Previously a single shared instance was reused across all models,
        # meaning every model after the first was using an already-fitted
        # preprocessor/sampler from the previous iteration.
        preprocessor = build_preprocessor(x)
        sampler = _build_sampler(y_encoded)

        pipeline = ImbPipeline(
            steps=[
                ("prep", preprocessor),
                ("sampler", sampler),
                ("clf", model),
            ]
        )

        logger.info("  Running %d-fold cross-validation on train set …", 5)
        metrics = evaluate_classifier(pipeline, x_train, y_train, cv_splits=5)
        all_metrics[name] = {f"cv_{k}": v for k, v in metrics.items()}
        logger.info("  CV metrics: %s", metrics)

        logger.info("  Fitting final pipeline on full train set …")
        pipeline.fit(x_train, y_train)

        # ── Held-out test evaluation ───────────────────────────────────
        y_test_pred = pipeline.predict(x_test)
        test_metrics = {
            "test_f1_macro": float(f1_score(y_test, y_test_pred, average="macro")),
            "test_precision_macro": float(precision_score(y_test, y_test_pred, average="macro", zero_division=0)),
            "test_recall_macro": float(recall_score(y_test, y_test_pred, average="macro", zero_division=0)),
        }
        all_metrics[name].update(test_metrics)
        logger.info("  Test metrics: %s", test_metrics)

        model_file = output / f"{name}_pipeline.joblib"
        labels_file = output / f"{name}_label_encoder.joblib"
        joblib.dump(pipeline, model_file)
        joblib.dump(label_encoder, labels_file)
        logger.info("  Saved → %s", model_file)

        log_run(
            model_name=name,
            model_object=pipeline,
            metrics=all_metrics[name],
            params={
                "dataset": str(data_path),
                "target_col": target_col,
                "sample_size": sample_size or -1,
                "test_size": test_size,
                "split_seed": split_seed,
            },
            artifact_path=model_file,
        )

    summary_path = output / "metrics_summary.csv"
    pd.DataFrame(all_metrics).T.to_csv(summary_path, index=True)
    logger.info("Metrics summary saved → %s", summary_path)
    return all_metrics
