from fastapi.testclient import TestClient
import numpy as np
from src import api as api_mod


def test_health_ok():
    """Health endpoint should return a JSON object with status='ok'."""
    client = TestClient(api_mod.app)
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, dict)
    assert data["status"] == "ok"


def test_features_returned():
    """Features endpoint should return a JSON object with a non-empty list."""
    client = TestClient(api_mod.app)
    r = client.get("/features")
    assert r.status_code == 200
    data = r.json()
    assert "features" in data
    feats = data["features"]
    assert isinstance(feats, list)
    assert len(feats) > 0


def test_predict_with_dummy_model(monkeypatch):
    """
    Patch MODEL & FEATURES in the api module so the test is fast and does not
    depend on the real joblib model file.
    """
    class DummyModel:
        def predict(self, X):
            return np.array([123.45])

    monkeypatch.setattr(api_mod, "MODEL", DummyModel())
    monkeypatch.setattr(api_mod, "FEATURES", [
        "room_type", "neighbourhood", "accommodates", "bedrooms", "bathrooms_num",
        "minimum_nights", "number_of_reviews", "reviews_per_month", "availability_365"
    ])

    client = TestClient(api_mod.app)
    payload = {
        "room_type": "Entire home/apt",
        "neighbourhood": "Mitte",
        "accommodates": 2,
        "bedrooms": 1,
        "bathrooms_num": 1.0,
        "minimum_nights": 2,
        "number_of_reviews": 10,
        "reviews_per_month": 0.3,
        "availability_365": 120
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    assert r.json()["price_eur"] == 123.45


def test_predict_validation_error():
    """Wrong types should trigger FastAPI/Pydantic validation (422)."""
    client = TestClient(api_mod.app)
    bad_payload = {"room_type": 123}  # wrong type on purpose
    r = client.post("/predict", json=bad_payload)
    assert r.status_code == 422


def test_model_info_endpoint():
    """Model info endpoint should expose features and model path (and maybe metrics)."""
    client = TestClient(api_mod.app)
    r = client.get("/model_info")
    assert r.status_code == 200
    data = r.json()
    assert "features" in data and isinstance(data["features"], list)
    # should match what the API has loaded
    assert data["features"] == api_mod.FEATURES
    assert "model_path" in data and isinstance(data["model_path"], str)
    # metrics/created may be None depending on the meta file
    if data.get("metrics") is not None:
        assert isinstance(data["metrics"], dict)


def test_predict_batch_with_dummy_model(monkeypatch):
    """
    Batch prediction should return one price per input item.
    """
    class DummyModel:
        def predict(self, X):
            return np.array([111.11, 222.22])

    monkeypatch.setattr(api_mod, "MODEL", DummyModel())
    monkeypatch.setattr(api_mod, "FEATURES", [
        "room_type", "neighbourhood", "accommodates", "bedrooms", "bathrooms_num",
        "minimum_nights", "number_of_reviews", "reviews_per_month", "availability_365",
    ])

    client = TestClient(api_mod.app)
    payloads = [
        {
            "room_type": "Entire home/apt",
            "neighbourhood": "Mitte",
            "accommodates": 2,
            "bedrooms": 1.0,
            "bathrooms_num": 1.0,
            "minimum_nights": 2,
            "number_of_reviews": 10,
            "reviews_per_month": 0.3,
            "availability_365": 120,
        },
        {
            "room_type": "Entire home/apt",
            "neighbourhood": "Kreuzberg",
            "accommodates": 4,
            "bedrooms": 2.0,
            "bathrooms_num": 1.0,
            "minimum_nights": 3,
            "number_of_reviews": 5,
            "reviews_per_month": 0.2,
            "availability_365": 200,
        },
    ]
    r = client.post("/predict_batch", json=payloads)
    assert r.status_code == 200
    assert r.json()["prices_eur"] == [111.11, 222.22]


def test_predict_batch_empty_list():
    """Empty batch should return an empty list, not an error."""
    client = TestClient(api_mod.app)
    r = client.post("/predict_batch", json=[])
    assert r.status_code == 200
    assert r.json()["prices_eur"] == []
