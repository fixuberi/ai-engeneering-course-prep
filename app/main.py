"""FastAPI service for the Iris classifier.

Run with:  uv run uvicorn app.main:app --reload
Then open: http://127.0.0.1:8000/docs
"""

from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI

from app.schemas import IrisFeatures, PredictResponse

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.joblib"


def field_name(feature_name):
    """Turn 'sepal length (cm)' into 'sepal_length' to match IrisFeatures."""
    return feature_name.removesuffix(" (cm)").replace(" ", "_")


def load_bundle(path=MODEL_PATH):
    """Load the saved model and check its features match the request schema."""
    if not path.exists():
        raise RuntimeError(f"Model file not found: {path}. Train it first: uv run python train.py")

    bundle = joblib.load(path)
    model_fields = [field_name(name) for name in bundle["feature_names"]]
    schema_fields = list(IrisFeatures.model_fields)
    if sorted(model_fields) != sorted(schema_fields):
        raise RuntimeError(f"Model features {model_fields} do not match API fields {schema_fields}")

    bundle["field_order"] = model_fields
    return bundle


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.bundle = load_bundle()
    yield


app = FastAPI(
    title="Iris ML service",
    description="Predict an iris species from flower measurements.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.post("/predict", response_model=PredictResponse)
def predict(features: IrisFeatures) -> PredictResponse:
    """Predict the iris class for one flower."""
    bundle = app.state.bundle
    row = [[getattr(features, name) for name in bundle["field_order"]]]

    pipeline = bundle["pipeline"]
    class_id = int(pipeline.predict(row)[0])
    proba = pipeline.predict_proba(row)[0]

    target_names = bundle["target_names"]
    return PredictResponse(
        class_id=class_id,
        class_name=target_names[class_id],
        probabilities={name: round(float(p), 4) for name, p in zip(target_names, proba)},
    )
