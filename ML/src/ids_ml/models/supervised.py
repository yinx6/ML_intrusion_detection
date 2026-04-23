from __future__ import annotations

from typing import Dict

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier


def get_supervised_models(n_classes: int, random_state: int = 42) -> Dict[str, object]:
    xgb_objective = "binary:logistic" if n_classes <= 2 else "multi:softprob"
    xgb_kwargs = {"eval_metric": "logloss"}
    if n_classes > 2:
        xgb_kwargs["num_class"] = n_classes

    return {
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=350,
            max_depth=8,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            objective=xgb_objective,
            random_state=random_state,
            n_jobs=-1,
            **xgb_kwargs,
        ),
        # Bug 8 fix: max_iter=60 caused ConvergenceWarning on large datasets.
        # Increased to 300 and enabled early_stopping to avoid overfitting.
        "mlp": MLPClassifier(
            hidden_layer_sizes=(256, 128),
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            random_state=random_state,
        ),
    }
