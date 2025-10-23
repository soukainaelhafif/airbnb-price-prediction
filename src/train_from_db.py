import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Connect to the PostgreSQL database
DB_URL = "postgresql+psycopg2://souka:password@localhost:5432/airbnb"
engine = create_engine(DB_URL)

print("Connected to PostgreSQL database!")

# Load data directly from the table
query = "SELECT * FROM listings;"
df = pd.read_sql(query, engine)

print(f"Loaded {len(df)} rows from database.")

# Prepare features and target
X = df.drop(columns=["price"])
y = df["price"]

# Handle missing values
X = X.fillna(0)

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

# Define preprocessing (encode categorical + keep numeric)
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols),
    ]
)

# Build pipeline: preprocessing + model
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=100, random_state=42)),
    ]
)

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Model trained successfully!")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R² Score: {r2:.3f}")

# Save model
import os
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/price_model.pkl")
print("Model saved to 'models/price_model.pkl'")