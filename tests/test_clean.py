import pandas as pd
from src.clean import to_euro, simple_bath, prepare_features

def test_to_euro():
    assert to_euro("€1.234,56") == 1234.56
    assert to_euro("1,234") == 1234.0
    assert to_euro("999") == 999.0

def test_simple_bath():
    assert simple_bath("1.5 baths") == 1.5
    assert simple_bath("1,5") == 1.5
    assert simple_bath("half bath") == 0.5

def test_prepare_features_minimal():
    data = {
        "price": ["€100", "€200", "€300"],
        "room_type": ["Entire home/apt", "Private room", "Entire home/apt"],
        "neighbourhood": ["A", "B", "A"],
        "accommodates": [2, 1, 3],
        "bedrooms": [1, 1, 2],
        "bathrooms_text": ["1 bath", "half bath", "1.5 baths"],
        "minimum_nights": [1, 2, 1],
        "number_of_reviews": [10, 0, 5],
        "reviews_per_month": [0.4, 0.0, 0.2],
        "availability_365": [120, 200, 50],
    }
    df = pd.DataFrame(data)
    out = prepare_features(df)
    assert {"price","room_type","neighbourhood","accommodates","bedrooms","bathrooms_num",
            "minimum_nights","number_of_reviews","reviews_per_month","availability_365"}.issubset(out.columns)
    assert len(out) == 3