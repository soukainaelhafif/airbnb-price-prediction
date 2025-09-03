FROM python:3.11-slim

LABEL org.opencontainers.image.title="Airbnb Berlin – Price Prediction API"
LABEL org.opencontainers.image.description="FastAPI service that predicts nightly Airbnb prices in Berlin. Includes sklearn training pipeline and metadata-aware model loading."
LABEL org.opencontainers.image.source="https://github.com/soukainaelhafif/airbnb-price-prediction"
LABEL org.opencontainers.image.licenses="MIT"
LABEL org.opencontainers.image.authors="Soukaina Elhafif <souka.elhafif@gmail.com>"


WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY src ./src
COPY README.md .

# Runtime configuration 
ENV PYTHONPATH=/app
ENV MODEL_PATH=/models/baseline.joblib
ENV META_PATH=/models/baseline.joblib.meta.json

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]