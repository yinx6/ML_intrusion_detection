from __future__ import annotations

from sklearn.ensemble import IsolationForest


def get_isolation_forest(random_state: int = 42) -> IsolationForest:
    return IsolationForest(
        n_estimators=250,
        contamination="auto",
        random_state=random_state,
        n_jobs=-1,
    )
