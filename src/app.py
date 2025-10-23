from fastapi import FastAPI, Query
import pandas as pd
import joblib

# Initialize FastAPI app
app = FastAPI(title="Airbnb Price Prediction API")

# Load trained model
import os
model_path = os.getenv("MODEL_PATH", "/models/price_model.pkl")
model = joblib.load(model_path)

@app.get("/")
def home():
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
    # Create input DataFrame
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

    # Make prediction
    pred = model.predict(sample)[0]
    return {"predicted_price": round(float(pred), 2)}