from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def load_dataset(path: str | Path, target_col: str = "Label") -> Tuple[pd.DataFrame, pd.Series]:
    """Load a CSV dataset and split features/target."""
    df = pd.read_csv(path, low_memory=False)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found. Available: {list(df.columns)[:10]}...")

    y = df[target_col].astype(str)
    x = df.drop(columns=[target_col])
    return x, y


def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean NaN/Inf and drop constant columns (nunique == 1 only)."""
    n_before = len(df)
    cleaned = df.replace([float("inf"), float("-inf")], pd.NA).dropna(axis=0)
    n_dropped = n_before - len(cleaned)
    if n_dropped > 0:
        logger.warning(
            "sanitize_dataframe: dropped %d / %d rows (%.1f%%) due to NaN/Inf values.",
            n_dropped,
            n_before,
            100.0 * n_dropped / max(n_before, 1),
        )

    # Bug 7 fix: only drop truly constant columns (nunique == 1).
    # The previous ratio-based filter incorrectly removed informative binary features.
    constant_cols = [c for c in cleaned.columns if cleaned[c].nunique(dropna=True) <= 1]
    if constant_cols:
        logger.info("sanitize_dataframe: dropping %d constant column(s): %s", len(constant_cols), constant_cols)
    return cleaned.drop(columns=constant_cols)


def make_split(
    x: pd.DataFrame,
    y: np.ndarray | pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Stratified train/test split.

    Parameters
    ----------
    x:
        Feature matrix (already sanitized).
    y:
        Label array (encoded integers or raw strings — stratification works
        for both).
    test_size:
        Fraction of samples to reserve for testing (default 0.20 = 20 %).
    random_state:
        Seed for reproducibility.

    Returns
    -------
    x_train, x_test, y_train, y_test
    """
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    logger.info(
        "make_split: train=%d rows (%.0f%%), test=%d rows (%.0f%%)",
        len(x_train),
        100.0 * len(x_train) / len(x),
        len(x_test),
        100.0 * len(x_test) / len(x),
    )
    return x_train, x_test, y_train, y_test
