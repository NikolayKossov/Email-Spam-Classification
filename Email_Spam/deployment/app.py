"""FastAPI service for the trained email-spam classifier."""

from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

# Importing the module is required when joblib restores FunctionTransformer.
from Email_Spam import email_spam_pipeline as _email_spam_pipeline  # noqa: F401


LOGGER = logging.getLogger(__name__)
DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[1] / "Email_Spam_Final_Pipeline.pkl"
)


class EmailPredictionRequest(BaseModel):
    text: str = Field(
        min_length=1,
        max_length=200_000,
        description="Raw email text, optionally including the Subject line.",
        examples=["Free money! Click here now to claim your prize."],
    )


class EmailPredictionResponse(BaseModel):
    prediction: int
    label: str
    spam_score: float
    decision_threshold: float
    model: str


def _model_path() -> Path:
    return Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH))).resolve()


def _load_artifact() -> dict:
    path = _model_path()
    if not path.is_file():
        raise FileNotFoundError(f"Model artifact was not found: {path}")

    artifact = joblib.load(path)
    required = {"pipeline", "decision_threshold", "positive_class"}
    missing = required - set(artifact)
    if missing:
        raise ValueError(f"Model artifact is missing keys: {sorted(missing)}")
    return artifact


def _positive_class_score(pipeline, texts) -> np.ndarray:
    """Return the same continuous score that was used to tune the threshold."""
    if hasattr(pipeline, "predict_proba"):
        return np.asarray(pipeline.predict_proba(texts))[:, 1]
    if hasattr(pipeline, "decision_function"):
        return np.asarray(pipeline.decision_function(texts), dtype=float)
    raise TypeError("The loaded pipeline has no probability or decision score")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.artifact = _load_artifact()
    yield
    app.state.artifact = None


app = FastAPI(
    title="Email Spam Classification API",
    version="1.0.0",
    description=(
        "Classifies raw email text with the locked preprocessing pipeline and "
        "the decision threshold selected on validation data."
    ),
    lifespan=lifespan,
)


@app.get("/health")
def health(request: Request) -> dict:
    artifact = request.app.state.artifact
    return {
        "status": "ok",
        "model_loaded": artifact is not None,
        "positive_class": artifact["positive_class"],
        "decision_threshold": float(artifact["decision_threshold"]),
    }


@app.get("/metadata")
def metadata(request: Request) -> dict:
    artifact = request.app.state.artifact
    metrics = artifact.get("final_test_metrics", {})
    return {
        "selected_model": str(artifact.get("selected_model", "unknown")),
        "text_representation": str(
            artifact.get("text_representation", "unknown")
        ),
        "feature_selection": str(
            artifact.get("feature_selection", "unknown")
        ),
        "positive_class": str(artifact["positive_class"]),
        "decision_threshold": float(artifact["decision_threshold"]),
        "final_test_metrics": {
            str(key): float(value)
            for key, value in dict(metrics).items()
            if isinstance(value, (int, float, np.integer, np.floating))
        },
    }


@app.post("/predict", response_model=EmailPredictionResponse)
def predict(payload: EmailPredictionRequest, request: Request):
    artifact = request.app.state.artifact
    try:
        score = float(
            _positive_class_score(artifact["pipeline"], [payload.text])[0]
        )
    except Exception as exc:
        LOGGER.exception("Email prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed") from exc

    threshold = float(artifact["decision_threshold"])
    prediction = int(score >= threshold)
    return EmailPredictionResponse(
        prediction=prediction,
        label="spam" if prediction else "not_spam",
        spam_score=score,
        decision_threshold=threshold,
        model=str(artifact.get("selected_model", "email_spam_pipeline")),
    )
