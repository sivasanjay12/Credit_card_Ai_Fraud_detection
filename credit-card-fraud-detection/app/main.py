from fastapi import FastAPI
from app.schema import PredictionRequest, PredictionResponse
from app.utils import predict
import pandas as pd


app = FastAPI(title="Credit Card Fraud Detection API",
              description="An API to detect credit card fraud using a pre-trained machine learning model.",
              version="1.0.0")

@app.get("/health", tags=["Health Check"])
def health_check():
    return {"status": "API is running"}

@app.put("/example")
def example_update():
    return {"message": "PUT method works"}

@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
def get_prediction(request: PredictionRequest):
    input_data = request.model_dump()
    df = pd.DataFrame([input_data])
    prediction = predict(df)
    
    response = PredictionResponse(
        fraud_detected=bool(prediction["fraud_detected"][0]),
        fraud_probability=float(prediction["fraud_probability"][0])
    )
    
    return response