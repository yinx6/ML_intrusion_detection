from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

MODEL_PATH = os.getenv("IDS_MODEL_PATH", "artifacts/random_forest_pipeline.joblib")
ENCODER_PATH = os.getenv("IDS_ENCODER_PATH", "artifacts/random_forest_label_encoder.joblib")


class PredictRequest(BaseModel):
    records: List[Dict[str, Any]]


pipeline = None
label_encoder = None


def _load_artifacts() -> None:
    """Load model pipeline and label encoder from disk."""
    global pipeline, label_encoder
    if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
        # Bug 4 fix: warn instead of silently returning so the missing files
        # are visible in server logs rather than first appearing at /predict time.
        logger.warning(
            "Model artifacts not found — model_path=%s encoder_path=%s. "
            "The /predict endpoint will return 503 until artifacts are present.",
            MODEL_PATH,
            ENCODER_PATH,
        )
        return
    pipeline = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    logger.info("Artifacts loaded successfully.")


# Bug 5 fix: replace deprecated @app.on_event("startup") with the modern
# lifespan context manager introduced in FastAPI 0.93+.
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    _load_artifacts()
    yield  # application runs here


app = FastAPI(title="IDS-ML API", version="1.0", lifespan=lifespan)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "model_loaded": pipeline is not None}


@app.post("/predict")
def predict(payload: PredictRequest) -> Dict[str, Any]:
    if pipeline is None or label_encoder is None:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded yet.")

    frame = pd.DataFrame(payload.records)
    pred = pipeline.predict(frame)
    labels = label_encoder.inverse_transform(pred)
    return {"predictions": labels.tolist()}
