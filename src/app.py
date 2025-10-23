from fastapi import FastAPI, Query
import pandas as pd
import joblib

# Initialize FastAPI app
app = FastAPI(
    title="Airbnb Price Prediction API",
    description="Predict nightly Airbnb prices in Berlin 🇩🇪",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Disable model loading (for Render demo)
model = None
meta = {}

@app.get("/")
def root():
    return {"message": "Welcome to the Airbnb Price Prediction API 🚀"}

@app.get("/predict")
def predict(
    room_type: str = Query(..., example="Entire home/apt"),
    neighbourhood: str = Query(..., example="Kreuzberg"),
    accommodates: int = Query(..., example=2),
    bedrooms: float = Query(..., example=1.0),
    bathrooms_num: float = Query(..., example=1.0),
    minimum_nights: int = Query(..., example=3),
    number_of_reviews: int = Query(..., example=50),
    reviews_per_month: float = Query(..., example=2.0),
    availability_365: int = Query(..., example=150)
):
    sample = pd.DataFrame([{
        "room_type": room_type,
        "neighbourhood": neighbourhood,
        "accommodates": accommodates,
        "bedrooms": bedrooms,
        "bathrooms_num": bathrooms_num,
        "minimum_nights": minimum_nights,
        "number_of_reviews": number_of_reviews,
        "reviews_per_month": reviews_per_month,
        "availability_365": availability_365
    }])

    # Fake prediction for demo (no model loaded)
    if model is None:
        demo_price = 80 + accommodates * 15 + bedrooms * 25  # simple example
        return {"predicted_price_demo": round(float(demo_price), 2)}

    # If model exists, make a real prediction
    pred = model.predict(sample)[0]
    return {"predicted_price": round(float(pred), 2)}