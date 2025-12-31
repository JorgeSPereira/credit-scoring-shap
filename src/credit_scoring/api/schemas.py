"""Pydantic schemas for API request/response validation."""

from typing import Optional

from pydantic import BaseModel, Field


class CreditApplication(BaseModel):
    """Schema for credit application input."""

    # Numerical features
    duration: int = Field(..., ge=1, le=72, description="Loan duration in months")
    credit_amount: float = Field(..., ge=0, description="Credit amount requested")
    installment_rate: int = Field(..., ge=1, le=4, description="Installment rate (% of income)")
    residence_since: int = Field(..., ge=1, le=4, description="Years at current residence")
    age: int = Field(..., ge=18, le=100, description="Age in years")
    existing_credits: int = Field(..., ge=1, le=4, description="Number of existing credits")
    num_dependents: int = Field(..., ge=1, le=2, description="Number of dependents")

    # Categorical features
    checking_status: str = Field(..., description="Checking account status (A11-A14)")
    credit_history: str = Field(..., description="Credit history (A30-A34)")
    purpose: str = Field(..., description="Loan purpose (A40-A410)")
    savings_status: str = Field(..., description="Savings account status (A61-A65)")
    employment: str = Field(..., description="Employment duration (A71-A75)")
    personal_status: str = Field(..., description="Personal status (A91-A95)")
    other_parties: str = Field(..., description="Other debtors/guarantors (A101-A103)")
    property_magnitude: str = Field(..., description="Property type (A121-A124)")
    other_payment_plans: str = Field(..., description="Other installment plans (A141-A143)")
    housing: str = Field(..., description="Housing type (A151-A153)")
    job: str = Field(..., description="Job type (A171-A174)")
    own_telephone: str = Field(..., description="Has telephone (A191-A192)")
    foreign_worker: str = Field(..., description="Is foreign worker (A201-A202)")

    class Config:
        json_schema_extra = {
            "example": {
                "duration": 24,
                "credit_amount": 5000.0,
                "installment_rate": 3,
                "residence_since": 2,
                "age": 35,
                "existing_credits": 1,
                "num_dependents": 1,
                "checking_status": "A12",
                "credit_history": "A32",
                "purpose": "A43",
                "savings_status": "A63",
                "employment": "A73",
                "personal_status": "A93",
                "other_parties": "A101",
                "property_magnitude": "A121",
                "other_payment_plans": "A143",
                "housing": "A152",
                "job": "A173",
                "own_telephone": "A192",
                "foreign_worker": "A201",
            }
        }


class PredictionResponse(BaseModel):
    """Schema for prediction response."""

    prediction: int = Field(..., description="0 = Good Credit, 1 = Bad Credit (Default)")
    probability_default: float = Field(..., ge=0, le=1, description="Probability of default")
    risk_level: str = Field(..., description="Risk classification: Low, Medium, or High")
    model_version: str = Field(..., description="Model version used for prediction")


class HealthResponse(BaseModel):
    """Schema for health check response."""

    status: str
    model_loaded: bool
    version: str
