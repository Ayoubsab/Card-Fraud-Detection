"""
Train the fraud detection pipeline.

    python -m fraud_detection.train                      # full run
    python -m fraud_detection.train --sample-rows 50000  # quick run
    python -m fraud_detection.train --no-autoencoder     # ablation without the AE feature
    python -m fraud_detection.train --no-uid-features    # ablation without the customer features
    python -m fraud_detection.train --no-refit           # keep the train-only model (no refit on train + val)

Validation picks the number of trees and the threshold, then the final model is refit on
train + validation and evaluated once on test.

Writes to the output directory:
    fraud_detection_pipeline.joblib  fitted pipeline (preprocessing + AE + XGBoost)
    metrics.json                     test set metrics
    metadata.json                    threshold, data splits, versions, git commit
"""

import argparse
import gc
import json
import logging
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, confusion_matrix, precision_recall_curve,
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve,
)

from fraud_detection.config import ID_COLUMN, MODEL_PATH, MODELS_DIR, PROJECT_ROOT, RANDOM_STATE, TARGET
from fraud_detection.data import load_data
from fraud_detection.pipeline import build_pipeline

logger = logging.getLogger(__name__)

VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
EARLY_STOPPING_ROUNDS = 100


def time_split(df: pd.DataFrame, val_fraction: float = VAL_FRACTION, test_fraction: float = TEST_FRACTION):
    """
    Split by time: train on the past, validate on the following period, test on the most recent one.
    A random split would leak future patterns into training.
    """
    df = df.sort_values("TransactionDT", kind="stable")
    n = len(df)
    train_end = int(n * (1 - val_fraction - test_fraction))
    val_end = int(n * (1 - test_fraction))
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


def best_f1_threshold(y_true, y_prob) -> float:
    """Decision threshold that maximizes F1 on the given (validation) set."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    # The last precision/recall pair has no threshold
    return float(thresholds[np.argmax(f1[:-1])])


def recall_at_fpr(y_true, y_prob, max_fpr: float) -> float:
    """Share of fraud caught while flagging at most `max_fpr` of legitimate transactions."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(tpr[fpr <= max_fpr].max())


def evaluate(y_true, y_prob, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "auprc": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "threshold": threshold,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "recall_at_1pct_fpr": recall_at_fpr(y_true, y_prob, 0.01),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n_samples": int(len(y_true)),
        "fraud_rate": float(np.mean(y_true)),
    }


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            # "-dirty" suffix when the model was trained from uncommitted code
            ["git", "describe", "--always", "--dirty"], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def package_versions(*packages: str) -> dict:
    """Installed versions, read from package metadata so torch isn't imported when unused."""
    versions = {}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


def train(
    sample_rows: int | None = None,
    use_autoencoder: bool = True,
    use_uid_features: bool = True,
    refit: bool = True,
    output_dir: Path = MODELS_DIR,
) -> dict:
    # torch is seeded by the autoencoder itself (random_state), only when it is used
    np.random.seed(RANDOM_STATE)

    logger.info("Loading data%s...", f" (first {sample_rows} rows)" if sample_rows else "")
    data = load_data("train", nrows=sample_rows)
    train_df, val_df, test_df = time_split(data)
    del data
    gc.collect()

    def split_xy(df):
        return df.drop(columns=[TARGET, ID_COLUMN]), df[TARGET]

    X_train, y_train = split_xy(train_df)
    X_val, y_val = split_xy(val_df)
    X_test, y_test = split_xy(test_df)
    logger.info("Split sizes: train=%d, val=%d, test=%d", len(X_train), len(X_val), len(X_test))

    # Calculate scale_pos_weight to handle class imbalance
    imbalance_ratio = float((y_train == 0).sum() / (y_train == 1).sum())
    pipeline = build_pipeline(
        scale_pos_weight=imbalance_ratio, use_autoencoder=use_autoencoder, use_uid_features=use_uid_features
    )

    # Fit the feature steps first so the validation set can be transformed for early stopping.
    # The steps are fitted in place, so `pipeline` ends up fully fitted.
    features, classifier = pipeline[:-1], pipeline[-1]
    logger.info("Fitting feature pipeline...")
    X_train_t = features.fit_transform(X_train, y_train)
    X_val_t = features.transform(X_val)

    logger.info("Training XGBoost with early stopping on validation AUPRC...")
    classifier.set_params(early_stopping_rounds=EARLY_STOPPING_ROUNDS)
    classifier.fit(X_train_t, y_train, eval_set=[(X_val_t, y_val)], verbose=False)
    logger.info("Best iteration: %d", classifier.best_iteration)

    # The threshold is chosen on validation, the test set is only used for the final report
    threshold = best_f1_threshold(y_val, classifier.predict_proba(X_val_t)[:, 1])
    best_iteration = int(classifier.best_iteration)
    del X_train_t, X_val_t
    gc.collect()

    if refit:
        # Validation was only needed to pick the number of trees and the threshold. Refit on
        # train + validation so the final model, and its customer lookup tables, include the
        # most recent month, as the production model would.
        X_full, y_full = pd.concat([X_train, X_val]), pd.concat([y_train, y_val])
        refit_ratio = float((y_full == 0).sum() / (y_full == 1).sum())
        pipeline = build_pipeline(
            scale_pos_weight=refit_ratio, use_autoencoder=use_autoencoder, use_uid_features=use_uid_features
        )
        pipeline[-1].set_params(n_estimators=best_iteration + 1)
        logger.info("Refitting on train + validation (%d rows, %d trees)...", len(X_full), best_iteration + 1)
        pipeline.fit(X_full, y_full)
        classifier = pipeline[-1]
        del X_full, y_full
        gc.collect()

    metrics = evaluate(y_test.to_numpy(), pipeline.predict_proba(X_test)[:, 1], threshold)
    logger.info("Test AUPRC=%.4f | ROC-AUC=%.4f | precision=%.3f | recall=%.3f @ threshold=%.3f",
                metrics["auprc"], metrics["roc_auc"], metrics["precision"], metrics["recall"], threshold)

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "threshold": threshold,
        "use_autoencoder": use_autoencoder,
        "use_uid_features": use_uid_features,
        "sample_rows": sample_rows,
        "refit_on_train_val": refit,
        "best_iteration": best_iteration,
        "scale_pos_weight": imbalance_ratio,
        "n_features": int(classifier.n_features_in_),
        "splits": {
            name: {
                "rows": int(len(df)),
                "fraud_rate": float(df[TARGET].mean()),
                "transaction_dt": [int(df["TransactionDT"].min()), int(df["TransactionDT"].max())],
            }
            for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]
        },
        "versions": package_versions("pandas", "scikit-learn", "xgboost", "torch"),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / MODEL_PATH.name
    joblib.dump(pipeline, model_path)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    logger.info("Saved model, metrics and metadata to %s", output_dir)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train the fraud detection pipeline.")
    parser.add_argument("--sample-rows", type=int, default=None,
                        help="Use only the first N transactions (quick runs).")
    parser.add_argument("--no-autoencoder", action="store_true",
                        help="Train without the autoencoder feature (ablation).")
    parser.add_argument("--no-uid-features", action="store_true",
                        help="Train without the customer (uid) features (ablation).")
    parser.add_argument("--no-refit", action="store_true",
                        help="Keep the model trained on train only instead of refitting on train + validation.")
    parser.add_argument("--output-dir", type=Path, default=MODELS_DIR,
                        help="Where to write the model, metrics and metadata.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    train(
        sample_rows=args.sample_rows,
        use_autoencoder=not args.no_autoencoder,
        use_uid_features=not args.no_uid_features,
        refit=not args.no_refit,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
