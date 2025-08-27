from __future__ import annotations
from pathlib import Path
from datetime import datetime
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TARGET = "price"
NUMERIC = [
    "accommodates", "bedrooms", "bathrooms_num",
    "minimum_nights", "number_of_reviews",
    "reviews_per_month", "availability_365",
]
CATEGORICAL = ["room_type", "neighbourhood"]
FEATURES = CATEGORICAL + NUMERIC

def _make_preprocessor() -> ColumnTransformer:
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([("num", num_pipe, NUMERIC),
                              ("cat", cat_pipe, CATEGORICAL)])

def train_on_df(df: pd.DataFrame):
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    X = df[FEATURES]
    y = df[TARGET].astype(float)

    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=42)

    pipe = Pipeline([
        ("prep", _make_preprocessor()),
        ("model", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)),
    ])
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_va)

    mae = float(mean_absolute_error(y_va, pred))
    rmse = float(np.sqrt(mean_squared_error(y_va, pred)))
    r2 = float(r2_score(y_va, pred))
    metrics = {"mae": mae, "rmse": rmse, "r2": r2,
               "n_train": int(len(X_tr)), "n_valid": int(len(X_va))}
    return pipe, metrics

def save_model(pipe: Pipeline, out_path: Path, meta: dict):
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out)
    meta_path = out.with_suffix(out.suffix + ".meta.json")
    meta = {**meta, "created": datetime.now().isoformat(timespec="seconds"),
            "features": FEATURES, "target": TARGET}
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

def load_data(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path, low_memory=False)

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Train baseline Airbnb price model")
    ap.add_argument("--data", default="data/berlin_clean.csv")
    ap.add_argument("--out", default="models/baseline.joblib")
    args = ap.parse_args()

    df = load_data(args.data)
    pipe, metrics = train_on_df(df)
    print("Metrics:", metrics)
    save_model(pipe, args.out, {"metrics": metrics})
    print(f"Saved model to {args.out}")

if __name__ == "__main__":
    main()