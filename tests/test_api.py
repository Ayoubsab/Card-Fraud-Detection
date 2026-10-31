import json

import joblib
import pytest
from fastapi.testclient import TestClient

from fraud_detection.api import app as app_module
from fraud_detection.pipeline import build_pipeline


@pytest.fixture
def client(raw_transactions, tmp_path, monkeypatch):
    """API client serving a tiny model trained on the synthetic data, instead of the real one."""
    X = raw_transactions.drop(columns="isFraud")
    y = raw_transactions["isFraud"]

    pipeline = build_pipeline()
    # set_params changes the classifier inside the pipeline, the whole pipeline is fitted and saved
    pipeline[-1].set_params(n_estimators=10)
    pipeline.fit(X, y)

    model_path = tmp_path / "model.joblib"
    metadata_path = tmp_path / "metadata.json"
    joblib.dump(pipeline, model_path)
    metadata_path.write_text(json.dumps({"threshold": 0.5, "git_commit": "test"}))

    # app.py imported these names from config, so they are patched where the app reads them
    monkeypatch.setattr(app_module, "MODEL_PATH", model_path)
    monkeypatch.setattr(app_module, "METADATA_PATH", metadata_path)

    with TestClient(app_module.app) as client:
        yield client


VALID_REQUEST = {"TransactionID": 1, "TransactionDT": 86400, "TransactionAmt": 50.0, "card1": 1000}


def test_health_reports_loaded_model(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "online"
    assert response.json()["model_version"] == "test"


def test_predict_returns_probability_and_decision(client):
    response = client.post("/predict", json=VALID_REQUEST)

    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["fraud_probability"] <= 1
    assert isinstance(body["is_fraud"], bool)
    assert body["threshold"] == 0.5
    assert body["model_version"] == "test"


def test_unknown_field_is_rejected(client):
    # A typo must not silently drop the amount
    request = {"TransactionID": 1, "TransactionDT": 86400, "TransactionAmnt": 50.0}

    response = client.post("/predict", json=request)

    assert response.status_code == 422


def test_predict_without_model_returns_503(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "MODEL_PATH", tmp_path / "missing.joblib")

    # The app logs the loading error and starts in degraded mode instead of crashing
    with TestClient(app_module.app) as client:
        assert client.get("/health").json()["status"] == "degraded"
        assert client.post("/predict", json=VALID_REQUEST).status_code == 503
