import pandas as pd
import joblib

# Load the saved model
model = joblib.load("models/price_model.pkl")
print("Model loaded successfully!")

# Create a sample input (you can replace this with real data)
sample = pd.DataFrame([{
    "room_type": "Entire home/apt",
    "neighbourhood": "Kreuzberg",
    "accommodates": 2,
    "bedrooms": 1.0,
    "bathrooms_num": 1.0,
    "minimum_nights": 3,
    "number_of_reviews": 50,
    "reviews_per_month": 2.0,
    "availability_365": 150
}])

# Make prediction
pred = model.predict(sample)[0]

print(f"Predicted price: {pred:.2f} EUR per night")