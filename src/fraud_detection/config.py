"""Project paths and constants. Paths can be overridden with environment variables."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.getenv("FRAUD_DATA_DIR", PROJECT_ROOT / "data"))
MODELS_DIR = Path(os.getenv("FRAUD_MODELS_DIR", PROJECT_ROOT / "models"))
MODEL_PATH = Path(os.getenv("FRAUD_MODEL_PATH", MODELS_DIR / "fraud_detection_pipeline.joblib"))

TARGET = "isFraud"
ID_COLUMN = "TransactionID"

RANDOM_STATE = 42
