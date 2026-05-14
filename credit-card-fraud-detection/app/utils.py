import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from src.feature_engineering import engineer_features

# Resolve project root (two levels up from app/utils.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def predict(df):
    model = joblib.load(PROJECT_ROOT / "models" / "fraud_model.pkl")
    threshold = joblib.load(PROJECT_ROOT / "models" / "optimal_threshold.pkl")
    scaler = joblib.load(PROJECT_ROOT / "data" / "processed" / "scaler.joblib")

    df_fe = engineer_features(df, "Amount", "Time")
    X_scaled = scaler.transform(df_fe)

    prob = model.predict_proba(X_scaled)[:,1]

    return { "fraud_probability": prob, "fraud_detected": (prob >= threshold).astype(int) }