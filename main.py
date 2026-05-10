from typing import List
from fastapi import FastAPI, HTTPException
from Transaction import Transaction
import joblib
import pandas as pd
import numpy as np
import traceback
import csv
app = FastAPI(title="Fraud Detection API")


try:
    model_pipeline = joblib.load('./models/fraud_detection_pipeline_v1.joblib')
    print("Model pipeline loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model_pipeline = None


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
        input_df = input_df.drop(columns=['TransactionID'], errors='ignore')
        input_df = input_df.fillna(value=np.nan)
        print(input_df.head())
        # Pipeline Execution
        probs = model_pipeline.predict_proba(input_df)[0][1]
        prediction = probs >= 0.5

        # Return results
        return {
            "fraud_probability": round(float(probs), 4),
            "is_fraud": bool(prediction),
            "reconstruction_error": float(input_df.get('ae_reconstruction_error', [0])[0]) if 'ae_reconstruction_error' in input_df else "N/A"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=traceback.format_exc()  # ← full traceback instead of str(e)
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)