"""FastAPI application for credit scoring predictions.

This module provides REST API endpoints for credit risk assessment,
enabling real-time predictions from trained models.
"""

from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from credit_scoring.api.schemas import (
    CreditApplication,
    HealthResponse,
    PredictionResponse,
)
from credit_scoring.models.train import load_model

# Initialize FastAPI app
app = FastAPI(
    title="Credit Scoring API",
    description="API for credit risk assessment using ML models with SHAP interpretability",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable
MODEL = None
MODEL_PATH = Path(__file__).parents[3] / "models" / "best_model_optuna.joblib"


def get_model():
    """Load model lazily."""
    global MODEL
    if MODEL is None:
        if MODEL_PATH.exists():
            MODEL = load_model(MODEL_PATH)
        else:
            # Fallback to any available model
            fallback_path = Path(__file__).parents[3] / "models" / "credit_scoring_model.joblib"
            if fallback_path.exists():
                MODEL = load_model(fallback_path)
    return MODEL


def classify_risk(probability: float) -> str:
    """Classify risk level based on probability."""
    if probability < 0.3:
        return "Low"
    elif probability < 0.6:
        return "Medium"
    else:
        return "High"


@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint."""
    model = get_model()
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        version="1.0.0",
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check."""
    model = get_model()
    return HealthResponse(
        status="healthy" if model else "degraded",
        model_loaded=model is not None,
        version="1.0.0",
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(application: CreditApplication):
    """Predict credit risk for an application.

    Args:
        application: Credit application data.

    Returns:
        Prediction with probability and risk level.
    """
    model = get_model()
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure model file exists.",
        )

    # Convert to DataFrame with original column names
    data = {
        "Attribute1": [application.checking_status],
        "Attribute2": [application.duration],
        "Attribute3": [application.credit_history],
        "Attribute4": [application.purpose],
        "Attribute5": [float(application.credit_amount)],
        "Attribute6": [application.savings_status],
        "Attribute7": [application.employment],
        "Attribute8": [application.installment_rate],
        "Attribute9": [application.personal_status],
        "Attribute10": [application.other_parties],
        "Attribute11": [application.residence_since],
        "Attribute12": [application.property_magnitude],
        "Attribute13": [application.age],
        "Attribute14": [application.other_payment_plans],
        "Attribute15": [application.housing],
        "Attribute16": [application.existing_credits],
        "Attribute17": [application.job],
        "Attribute18": [application.num_dependents],
        "Attribute19": [application.own_telephone],
        "Attribute20": [application.foreign_worker],
    }

    df = pd.DataFrame(data)

    # Make prediction
    prediction = int(model.predict(df)[0])
    probability = float(model.predict_proba(df)[0][1])
    risk_level = classify_risk(probability)

    return PredictionResponse(
        prediction=prediction,
        probability_default=round(probability, 4),
        risk_level=risk_level,
        model_version="1.0.0",
    )


@app.post("/predict/batch")
async def predict_batch(applications: list[CreditApplication]):
    """Batch prediction endpoint.

    Args:
        applications: List of credit applications.

    Returns:
        List of predictions.
    """
    results = []
    for app in applications:
        result = await predict(app)
        results.append(result)
    return results


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
