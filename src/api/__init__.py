from pathlib import Path
import json
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]  # Projekt-Root
DEFAULT_MODEL_PATH = ROOT / "models" / "baseline.joblib"
DEFAULT_META_PATH = ROOT / "models" / "baseline.joblib.meta.json"

app = FastAPI(title="Airbnb Berlin – Price Prediction", version="0.1.0")

# ---------- Pydantic Schemas ----------


class ListingIn(BaseModel):
    room_type: str
    neighbourhood: str
    accommodates: int = Field(ge=0)
    bedrooms: float = Field(ge=0)
    bathrooms_num: float = Field(ge=0)
    minimum_nights: int = Field(ge=0)
    number_of_reviews: int = Field(ge=0)
    reviews_per_month: float = Field(ge=0)
    availability_365: int = Field(ge=0, le=365)


class PredictionOut(BaseModel):
    price_eur: float

# ---------- Model Load ----------


def _load_model_and_meta(model_path: Path = DEFAULT_MODEL_PATH,
                         meta_path: Path = DEFAULT_META_PATH):
    model = joblib.load(model_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    features = meta.get("features")
    if not features:
        raise RuntimeError("No 'features' field in model meta.")
    return model, features


MODEL, FEATURES = _load_model_and_meta()

# ---------- Routes ----------


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/features")
def features():
    return {"features": FEATURES}


@app.post("/predict", response_model=PredictionOut)
def predict(payload: ListingIn):
    # DataFrame in der selben Reihenfolge wie im Training
    row = pd.DataFrame([payload.model_dump()])[FEATURES]
    pred = float(MODEL.predict(row)[0])
    return PredictionOut(price_eur=round(pred, 2))
