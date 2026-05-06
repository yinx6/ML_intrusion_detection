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
        # Performance notes:
        # - solver='sgd' with momentum converges faster than adam on large
        #   tabular datasets because each step is cheaper (no adaptive moments
        #   to track) and sklearn's MLP is pure-NumPy / single-threaded.
        # - Smaller architecture (128, 64) is sufficient for structured IDS
        #   features and cuts forward/back-prop time per batch by ~4x vs (256,128).
        # - batch_size=2048 reduces the number of gradient steps per epoch,
        #   keeping each iteration fast after SMOTE over-sampling.
        # - n_iter_no_change=5 (from 10) makes early stopping trigger sooner
        #   once the validation loss plateaus, avoiding wasted epochs.
        # - max_iter=100 is now the hard ceiling; early stopping will usually
        #   stop well before this on a well-scaled dataset.
        "mlp": MLPClassifier(
            hidden_layer_sizes=(128, 64),
            solver="sgd",
            momentum=0.9,
            alpha=1e-4,
            learning_rate="invscaling",
            learning_rate_init=0.01,
            power_t=0.5,
            batch_size=2048,
            max_iter=100,
            shuffle=True,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=5,
            random_state=random_state,
        ),
    }
