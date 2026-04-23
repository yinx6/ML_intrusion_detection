from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict


def _false_positive_rate_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute macro-averaged False Positive Rate across all classes.

    For each class c:
        FPR_c = FP_c / (FP_c + TN_c)

    The final value is the unweighted mean over all classes.
    The previous implementation computed the overall error rate, which is
    not the FPR and was misleading for multi-class IDS evaluation.
    """
    classes = np.unique(np.concatenate([y_true, y_pred]))
    fprs: list[float] = []
    for cls in classes:
        fp = int(np.sum((y_pred == cls) & (y_true != cls)))
        tn = int(np.sum((y_pred != cls) & (y_true != cls)))
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fprs.append(fpr)
    return float(np.mean(fprs))


def evaluate_classifier(model, x, y, cv_splits: int = 5) -> Dict[str, float]:
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    y_pred = cross_val_predict(model, x, y, cv=cv, n_jobs=1)

    return {
        "f1_macro": float(f1_score(y, y_pred, average="macro")),
        "precision_macro": float(precision_score(y, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y, y_pred, average="macro", zero_division=0)),
        # Bug 3 fix: proper per-class FPR, macro-averaged.
        "false_positive_rate_macro": _false_positive_rate_macro(y, y_pred),
    }
