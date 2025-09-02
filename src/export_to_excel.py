# src/export_to_excel.py
import os
import json
from pathlib import Path
import pandas as pd
import joblib

# --- Paths ---
# InsideAirbnb Berlin
DATA_PATH = os.getenv("DATA_PATH", "data/listings.csv.gz")
MODEL_PATH = os.getenv("MODEL_PATH", "models/baseline.joblib")
META_PATH = os.getenv("META_PATH",  "models/baseline.joblib.meta.json")
OUT_DIR = Path("outputs")
OUT_FILE = OUT_DIR / "airbnb_predictions_sample.xlsx"

OUT_DIR.mkdir(exist_ok=True)


def _ensure_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _basic_clean(df):
    # clean price symbols and decimals
    if "price" in df.columns:
        df["price"] = (
            df["price"].astype(str)
            .str.replace(r"[\$,€]", "", regex=True)
            .str.replace(",", "", regex=False)
        )
    num_cols = [
        "accommodates", "bedrooms", "bathrooms_num", "minimum_nights",
        "number_of_reviews", "reviews_per_month", "availability_365", "price"
    ]
    df = _ensure_numeric(df, num_cols)
    for c in num_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0)
    for c in ["room_type", "neighbourhood"]:
        if c in df.columns:
            df[c] = df[c].fillna("Unknown")
    return df


def main():
    # 1) load & clean data
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df = _basic_clean(df)

    # 2) load model + features
    model = joblib.load(MODEL_PATH)
    try:
        with open(META_PATH, "r", encoding="utf-8") as f:
            features = json.load(f).get("features", [])
    except FileNotFoundError:
        features = [
            "room_type", "neighbourhood", "accommodates", "bedrooms", "bathrooms_num",
            "minimum_nights", "number_of_reviews", "reviews_per_month", "availability_365"
        ]

    # 3) prepare inference frame
    X = df.copy()
    for col in features:
        if col not in X.columns:
            X[col] = "Unknown" if col in ("room_type", "neighbourhood") else 0

    # 4) predict
    y_pred = model.predict(X[features])

    # 5) tidy result table for Excel
    keep = [
        "id", "room_type", "neighbourhood", "accommodates", "bedrooms", "bathrooms_num",
        "minimum_nights", "number_of_reviews", "reviews_per_month", "availability_365", "price"
    ]
    keep = [c for c in keep if c in df.columns]
    out = df[keep].copy()
    out["predicted_price"] = y_pred
    if "price" in out.columns:
        out["error"] = out["predicted_price"] - out["price"]

    # keep it small for GitHub
    out_sample = out.head(1000)
    out_sample.to_excel(OUT_FILE, index=False)
    print(f"✅ Wrote {len(out_sample)} rows -> {OUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
