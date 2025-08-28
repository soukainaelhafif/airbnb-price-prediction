from pathlib import Path
from typing import List, Optional
import json
import joblib
import pandas as pd
import os
from fastapi import FastAPI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# project root
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = ROOT / "models" / "baseline.joblib"
DEFAULT_META_PATH = ROOT / "models" / "baseline.joblib.meta.json"

# Allow overriding via .env
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))
META_PATH = Path(os.getenv("META_PATH", str(DEFAULT_META_PATH)))

app = FastAPI(title="Airbnb Berlin – Price Prediction", version="0.2.0")

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


class ModelInfo(BaseModel):
    features: List[str]
    metrics: Optional[dict] = None
    created: Optional[str] = None
    model_path: str


class PredictionBatchOut(BaseModel):
    prices_eur: List[float]

# ---------- Model Load ----------


def _load_model_and_meta(model_path: Path = MODEL_PATH,
                         meta_path: Path = META_PATH):
    model = joblib.load(model_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    features = meta.get("features")
    if not features:
        raise RuntimeError("No 'features' field in model meta.")
    return model, features, meta


MODEL, FEATURES, META = _load_model_and_meta()

# ---------- Routes ----------


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/features")
def features():
    return {"features": FEATURES}


@app.get("/model_info", response_model=ModelInfo)
def model_info():
    return ModelInfo(
        features=FEATURES,
        metrics=META.get("metrics") if META else None,
        created=META.get("created") if META else None,
        model_path=str(MODEL_PATH),
    )


@app.post("/predict", response_model=PredictionOut)
def predict(payload: ListingIn):
    # Build a one-row DataFrame with the same columns/order as during training
    row = pd.DataFrame([payload.model_dump()])[FEATURES]
    pred = float(MODEL.predict(row)[0])
    return PredictionOut(price_eur=round(pred, 2))


@app.post("/predict_batch", response_model=PredictionBatchOut)
def predict_batch(items: List[ListingIn]):
    if not items:
        return PredictionBatchOut(prices_eur=[])
    df = pd.DataFrame([it.model_dump() for it in items])[FEATURES]
    preds = MODEL.predict(df)
    return PredictionBatchOut(prices_eur=[round(float(x), 2) for x in preds])
