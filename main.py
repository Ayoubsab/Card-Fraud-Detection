from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
import joblib
import pandas as pd
import numpy as np

app = FastAPI(title="Fraud Detection API")


try:
    model_pipeline = joblib.load('./models/fraud_detection_pipeline_v1.joblib')
    print("Model pipeline loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model_pipeline = None

class Transaction(BaseModel):
    features: List[float]

    @field_validator("features")
    @classmethod
    def check_length(cls, v):
        # n_features may not be set at class definition time, check at runtime
        return v

@app.get("/")
def health_check():
    return {"status": "online", "model_loaded": model_pipeline is not None}

@app.post("/predict")
async def predict(transaction: Transaction):
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([transaction.model_dump()])

        # Pipeline Execution
        probs = model_pipeline.predict_proba(input_df)[0, 1]
        prediction = int(model_pipeline.predict(input_df)[0])

        # Return results
        return {
            "fraud_probability": round(float(probs), 4),
            "is_fraud": bool(prediction),
            "reconstruction_error": float(input_df.get('ae_reconstruction_error', [0])[0]) if 'ae_reconstruction_error' in input_df else "N/A"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)