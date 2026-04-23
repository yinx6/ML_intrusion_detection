from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import pandas as pd

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
