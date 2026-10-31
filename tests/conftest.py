import numpy as np
import pandas as pd
import pytest

from fraud_detection.pipeline import build_pipeline

N_ROWS = 300


@pytest.fixture
def raw_transactions() -> pd.DataFrame:
    """Small synthetic dataset with the raw columns the pipeline uses, so tests run without the Kaggle data."""
    rng = np.random.default_rng(0)
    n = N_ROWS
    return pd.DataFrame({
        "TransactionDT": np.sort(rng.integers(86400, 86400 * 30, n)),
        "TransactionAmt": rng.uniform(1, 500, n).round(2),
        "ProductCD": rng.choice(["W", "C", "H"], n),
        "card1": rng.choice([1000, 2000, 3000, 4000], n),
        "card2": rng.choice([100.0, 200.0, np.nan], n),
        "addr1": rng.choice([100.0, 200.0, np.nan], n),
        "D1": rng.choice([0.0, 5.0, 10.0], n),
        "P_emaildomain": rng.choice(["gmail.com", "yahoo.co.uk", None], n),
        "R_emaildomain": rng.choice(["gmail.com", None], n),
        "DeviceInfo": rng.choice(["SM-G930", "iOS Device", "Windows", None], n),
        "id_30": rng.choice(["Windows 10", "iOS 11.2", None], n),
        "id_31": rng.choice(["chrome 63.0", "safari", None], n),
        "id_33": rng.choice(["1920x1080", None], n),
        "isFraud": rng.choice([0, 1], n, p=[0.9, 0.1]),
    })


@pytest.fixture
def fitted_features(raw_transactions):
    """Feature steps of the pipeline (everything except XGBoost), fitted on the synthetic data."""
    X = raw_transactions.drop(columns="isFraud")
    return build_pipeline()[:-1].fit(X)
