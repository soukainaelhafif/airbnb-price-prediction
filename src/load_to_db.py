import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError

# Database connection URL
DB_URL = "postgresql+psycopg2://souka:password@localhost:5432/airbnb"

try:
    # Connect to PostgreSQL
    engine = create_engine(DB_URL)
    print("Successfully connected to the PostgreSQL database!")

    # Load the cleaned data
    df = pd.read_csv("data/berlin_clean.csv")
    print(f"Loaded {len(df)} rows from 'data/berlin_clean.csv'")

    # Write data to the 'listings' table (replace if it already exists)
    df.to_sql("listings", engine, if_exists="replace", index=False)
    print(f"Successfully inserted {len(df)} rows into the 'listings' table!")

except FileNotFoundError:
    print("Error: The file 'data/berlin_clean.csv' was not found.")
except SQLAlchemyError as e:
    print("Database error:", e)
except Exception as e:
    print("Unexpected error:", e)
