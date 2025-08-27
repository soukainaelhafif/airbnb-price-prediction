from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


# ---------- helpers from notebook ----------

def to_euro(x) -> float | np.ndarray:
    """Turn strings like '€1.234,56' or '1,234' into float euros."""
    if pd.isna(x):
        return np.nan
    s = str(x).replace("\xa0", "").replace("€", "").replace("$", "").strip()
    if "," in s and "." in s:                   # e.g. 1.234,56 -> 1234.56
        s = s.replace(".", "").replace(",", ".")
    elif "," in s:                              # e.g. 1,234   -> 1234
        s = s.replace(",", "")
    elif "." in s and len(s.split(".")[-1]) == 3:  # e.g. 1.234 -> 1234
        s = s.replace(".", "")
    try:
        return float(s)
    except ValueError:
        return np.nan


def simple_bath(x) -> float:
    """Parse bathrooms_text like '1.5 baths', 'half bath', '1,5' -> float."""
    if pd.isna(x):
        return np.nan
    s = str(x).lower().replace(",", ".")
    if "half" in s and not any(ch.isdigit() for ch in s):
        return 0.5
    buf = []
    for ch in s:
        if ch.isdigit() or ch == ".":
            buf.append(ch)
        elif buf:
            break
    try:
        return float("".join(buf)) if buf else np.nan
    except ValueError:
        return np.nan


def pick_neighbourhood_col(df: pd.DataFrame) -> str | None:
    if "neighbourhood_cleansed" in df.columns:
        return "neighbourhood_cleansed"
    if "neighbourhood" in df.columns:
        return "neighbourhood"
    return None


# ---------- main prepare function ----------

FEATURE_CANDIDATES = [
    "room_type", "neighbourhood", "accommodates", "bedrooms", "bathrooms_num",
    "minimum_nights", "number_of_reviews", "reviews_per_month", "availability_365"
]
TARGET = "price"


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    # price -> numeric euros
    if "price" not in df.columns:
        raise KeyError("Column 'price' not found in input data.")
    df = df.copy()
    df["price"] = df["price"].apply(to_euro)

    # neighbourhood column -> 'neighbourhood'
    nbh = pick_neighbourhood_col(df)
    if nbh and nbh != "neighbourhood":
        df["neighbourhood"] = df[nbh]
    elif nbh is None:
        # no neighbourhood in this snapshot; keep missing column to later drop
        df["neighbourhood"] = np.nan

    # bathrooms_num from best available source
    if "bathrooms" in df.columns:
        df["bathrooms_num"] = pd.to_numeric(df["bathrooms"], errors="coerce")
    elif "bathrooms_text" in df.columns:
        df["bathrooms_num"] = df["bathrooms_text"].apply(simple_bath)
    else:
        df["bathrooms_num"] = np.nan

    # cast categoricals (optional)
    if "room_type" in df.columns:
        df["room_type"] = df["room_type"].astype("category")
    if "neighbourhood" in df.columns:
        df["neighbourhood"] = df["neighbourhood"].astype("category")

    # ensure typical numeric cols are numeric
    for c in ["accommodates", "bedrooms", "minimum_nights",
              "number_of_reviews", "reviews_per_month", "availability_365"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # pick available features
    features = [c for c in FEATURE_CANDIDATES if c in df.columns]
    cols = features + [TARGET]
    out = df[cols].dropna()
    if len(out) == 0:
        raise ValueError("No rows left after dropna(). Check input columns.")
    return out


# ---------- CLI ----------

def main():
    p = argparse.ArgumentParser(
        description="Clean InsideAirbnb listing data and export a compact feature table.")
    p.add_argument("--in", dest="inp", required=True,
                   help="Path to raw listings CSV/CSV.GZ (project root)")
    p.add_argument("--out", dest="out", required=True,
                   help="Output CSV path for cleaned features")
    args = p.parse_args()

    inp = Path(args.inp)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {inp} ...")
    df = pd.read_csv(inp, low_memory=False)
    print("Raw shape:", df.shape)

    clean = prepare_features(df)
    clean.to_csv(out, index=False)
    print(f"Saved {out} | Shape: {clean.shape}")
    print("Columns:", list(clean.columns))


if __name__ == "__main__":
    main()
