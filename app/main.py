from pathlib import Path
from typing import Literal, Optional

import joblib
import numpy as np
import pandas as pd
import sklearn
from fastapi import FastAPI
from pydantic import BaseModel, Field

# ---------- 1. Load the saved model ONCE, when the app starts ----------
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "churn_model.joblib"
artifact = joblib.load(MODEL_PATH)

pipeline= artifact["pipeline"]
threshold= artifact["threshold"]
features= artifact["features"]

if artifact["sklearn_version"] != sklearn.__version__:
    print("WARNING: scikit-learn version differs from training. Retrain or reinstall.")


# ---------- 2. Describe what a valid customer looks like ----------
YesNo= Literal["Yes", "No"]

class Customer(BaseModel):
    Gender: Literal["Female", "Male"]
    Age: int = Field(ge=18, le=100)
    Under30: YesNo
    SeniorCitizen: YesNo
    Married: YesNo
    Dependents: YesNo
    NumberofDependents: int = Field(ge=0, le=20)
    Population: int = Field(ge=0)
    Quarter: str = "Q3"
    ReferredaFriend: YesNo
    Number_of_Referrals: int = Field(ge=0)
    TenureinMonths: int = Field(ge=0, le=100)
    Offer: Optional[Literal["Offer A", "Offer B", "Offer C", "Offer D", "Offer E"]] = None
    PhoneService: YesNo
    AvgMonthlyLongDistanceCharges: float = Field(ge=0)
    MultipleLines: YesNo
    InternetService: YesNo
    InternetType: Optional[Literal["Cable", "DSL", "Fiber Optic"]] = None
    AvgMonthlyGBDownload: int = Field(ge=0)
    OnlineSecurity: YesNo
    OnlineBackup: YesNo
    DeviceProtectionPlan: YesNo
    PremiumTechSupport: YesNo
    StreamingTV: YesNo
    StreamingMovies: YesNo
    StreamingMusic: YesNo
    UnlimitedData: YesNo
    Contract: Literal["Month-to-Month", "One Year", "Two Year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal["Bank Withdrawal", "Credit Card", "Mailed Check"]
    MonthlyCharge: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)
    TotalRefunds: float = Field(ge=0)
    TotalExtraDataCharges: int = Field(ge=0)
    TotalLongDistanceCharges: float = Field(ge=0)
    TotalRevenue: float = Field(ge=0)

# ---------- 3. The app and its endpoints ----------
app =FastAPI(title= "Churn Prediction API")

@app.get("/")
def home():
    return {"status": "ok", "message": "Churn API is running"}

@app.post("/predict")
def predict(customer: Customer):
    row= customer.model_dump()
    row = {k: (np.nan if v is None else v) for k, v in row.items()}
    df = pd.DataFrame([row])[features]

    proba = float(pipeline.predict_proba(df)[0, 1])

    return {
        "churn_probability": round(proba, 4),
        "will_churn": proba >= threshold,
        "threshold": round(threshold, 3),
    }


@app.get("/health")
def health():
    return {"Status": "healthy", "model_loaded": pipeline is not None}