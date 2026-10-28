import gc
from pathlib import Path

import pandas as pd

from fraud_detection.config import DATA_DIR, ID_COLUMN


def load_data(split: str = "train", data_dir: Path = DATA_DIR, nrows: int | None = None) -> pd.DataFrame:
    """
    Load and merge the transaction and identity tables for the given split.
    `nrows` keeps only the first transactions (in time order), for quick runs.
    """
    transactions = pd.read_csv(data_dir / f"{split}_transaction.csv", nrows=nrows)
    identity = pd.read_csv(data_dir / f"{split}_identity.csv")

    df = transactions.merge(identity, on=ID_COLUMN, how="left", validate="one_to_one")

    del transactions, identity
    gc.collect()
    return df
