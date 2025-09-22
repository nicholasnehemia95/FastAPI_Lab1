# app.py
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from joblib import load

app = FastAPI(title="FastAPI - Pruned Decision Tree", version="1.0.0")

# Load bundle
bundle = load("artifacts_dt_pruned.joblib")
model = bundle["model"]
feature_names = bundle["feature_names"]
columns_to_drop = bundle["columns_to_drop"]
categorical_maps = bundle["categorical_maps"]
columns_to_scale = bundle["columns_to_scale"]
scaler = bundle.get("scaler", None)

class InsuranceRecord(BaseModel):
    # Original columns (as in your raw CSV)
    id: Optional[int] = Field(None, description="Ignored at inference")
    Gender: str
    Age: int
    Driving_License: int
    Region_Code: float
    Previously_Insured: int
    Vehicle_Age: str
    Vehicle_Damage: str
    Annual_Premium: float
    Policy_Sales_Channel: float
    Vintage: int
    Response: Optional[int] = Field(None, description="Not required for prediction")

def preprocess_one(payload: InsuranceRecord) -> pd.DataFrame:
    """Replicates your preprocessing for a single record."""
    row = pd.DataFrame([payload.dict()])

    # 1) Drop unwanted columns
    row = row.drop(columns=[c for c in columns_to_drop if c in row.columns], errors="ignore")

    # 2) Encode categorical variables (strict; unknowns raise 400)
    for col, mapping in categorical_maps.items():
        if col not in row.columns:
            raise HTTPException(status_code=400, detail=f"Missing required column: {col}")
        val = row.at[0, col]
        if val not in mapping:
            raise HTTPException(status_code=400, detail=f"Unknown category for '{col}': {val}")
        row[col] = mapping[val]

    # 3) Scale continuous columns with the saved RobustScaler
    if scaler is None:
        raise HTTPException(
            status_code=500,
            detail="Scaler not found in artifacts. Please dump 'robust_scaler.joblib' during preprocessing."
        )
    for c in columns_to_scale:
        if c not in row.columns:
            raise HTTPException(status_code=400, detail=f"Missing required column: {c}")
    row_scaled = row.copy()
    row_scaled[columns_to_scale] = scaler.transform(row[columns_to_scale])

    # 4) Reorder/align to training feature order
    #    (Any missing columns should raise a clear error)
    missing = [c for c in feature_names if c not in row_scaled.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing features after preprocessing: {missing}")

    row_ordered = row_scaled[feature_names]
    return row_ordered

@app.get("/meta")
def meta():
    meta = {k: v for k, v in bundle.items() if k != "model" and k != "scaler"}
    meta["has_scaler"] = (scaler is not None)
    return meta

@app.post("/predict")
def predict(record: InsuranceRecord):
    X = preprocess_one(record)
    pred = int(model.predict(X)[0])
    proba = getattr(model, "predict_proba", None)
    p1 = float(proba(X)[0, 1]) if proba else None
    return {"prediction": pred, "proba_1": p1}
