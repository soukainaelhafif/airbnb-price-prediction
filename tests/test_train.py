import numpy as np
import pandas as pd
from pathlib import Path
from src.models.train import train_on_df, save_model


def _dummy_df(n=150, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "room_type": rng.choice(["Entire home/apt", "Private room", "Shared room"], size=n),
        "neighbourhood": rng.choice(["Mitte", "Friedrichshain", "Kreuzberg"], size=n),
        "accommodates": rng.integers(1, 6, size=n),
        "bedrooms": rng.integers(0, 3, size=n),
        "bathrooms_num": rng.choice([0.5, 1.0, 1.5, 2.0], size=n),
        "minimum_nights": rng.integers(1, 7, size=n),
        "number_of_reviews": rng.integers(0, 50, size=n),
        "reviews_per_month": rng.random(n) * 5,
        "availability_365": rng.integers(0, 365, size=n),
    })
    base = 40 + 15*df["accommodates"] + 10 * \
        df["bedrooms"] + 8*df["bathrooms_num"]
    room_adj = df["room_type"].map(
        {"Entire home/apt": 40, "Private room": 0, "Shared room": -10})
    noise = rng.normal(0, 10, size=n)
    df["price"] = (base + room_adj + noise).clip(20, 500)
    return df


def test_train_and_save(tmp_path: Path):
    df = _dummy_df()
    pipe, metrics = train_on_df(df)
    assert metrics["rmse"] >= 0 and metrics["mae"] >= 0
    out = tmp_path / "model.joblib"
    save_model(pipe, out, {"metrics": metrics})
    assert out.exists()
    preds = pipe.predict(df.drop(columns=["price"]).head(5))
    assert preds.shape == (5,)
