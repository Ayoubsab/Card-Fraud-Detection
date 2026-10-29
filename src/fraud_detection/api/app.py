import joblib
import pandas as pd
import json
import logging

from fastapi import FastAPI, HTTPException, Request
from contextlib import asynccontextmanager

from fraud_detection.api.schemas import Transaction, HealthResponse, PredictionResponse
from fraud_detection.config import MODEL_PATH, METADATA_PATH


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.model_pipeline = joblib.load(MODEL_PATH)
        logger.info("Model pipeline loaded successfully.")
    except Exception:
        logger.exception("Error loading model.")
        app.state.model_pipeline = None

    try:
        with open(METADATA_PATH) as f:
            app.state.metadata = json.load(f)
            logger.info("Metadata loaded successfully.")
    except Exception:
        logger.exception("Error loading metadata.")
        app.state.metadata = {}

    yield

app = FastAPI(title="Fraud Detection API", lifespan=lifespan)


@app.get("/")
@app.get("/health")
def health_check(request: Request) -> HealthResponse:
    
    if request.app.state.model_pipeline is not None:
        
        model_loaded = True
        model_version = request.app.state.metadata.get("git_commit")
        trained_at = request.app.state.metadata.get("created_at")
        health_response = HealthResponse(status="online", model_loaded=model_loaded, 
                                         model_version=model_version, trained_at=trained_at)
    else:
        health_response = HealthResponse(status="degraded", model_loaded=False)
    return health_response

@app.post("/predict")
def predict(request: Request, transaction: Transaction) -> PredictionResponse:
    if request.app.state.model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([transaction.model_dump()])
        input_df = input_df.drop(columns=['TransactionID'], errors='ignore')

        # Pipeline Execution
        probs = request.app.state.model_pipeline.predict_proba(input_df)[0][1]
        prediction = probs >= request.app.state.metadata.get("threshold", 0.5)

        # Return results
        return PredictionResponse(
            fraud_probability= round(float(probs), 4),
            is_fraud= bool(prediction),
            threshold=request.app.state.metadata.get("threshold", 0.5),
            model_version=request.app.state.metadata.get("git_commit")
        )
        

    except Exception:
        logger.exception("Prediction failed.")
        raise HTTPException(
            status_code=500, 
            detail="Prediction failed."
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)