import torch
import xgboost as xgb
from sklearn.pipeline import Pipeline

from fraud_detection.autoencoder.bridge import AutoencoderPipeline
from fraud_detection.config import RANDOM_STATE
from fraud_detection.features.device import DevicePreprocessor
from fraud_detection.features.preprocessor import DataPreprocessor
from fraud_detection.features.uid import UidAggregator


def build_pipeline(
    scale_pos_weight: float = 1.0, use_autoencoder: bool = False, use_uid_features: bool = True
) -> Pipeline:
    """Customer features -> device cleaning -> feature engineering -> autoencoder anomaly score -> XGBoost."""
    steps = []
    if use_uid_features:
        # First, it needs raw columns (DeviceInfo) that the next steps drop
        steps.append(('uid_aggregator', UidAggregator()))
    steps += [
        ('device_preprocessor', DevicePreprocessor(min_frequency=250)),
        ('general_preprocessor', DataPreprocessor(nan_threshold=0.01)),
    ]
    if use_autoencoder:
        # Imported here so torch is only loaded when the autoencoder is used:
        # on macOS, torch's OpenMP runtime crashes xgboost training in the same process.
        from fraud_detection.autoencoder.extractor import AutoencoderFeatureExtractor

        steps.append(('autoencoder_bridge', AutoencoderPipeline(
            AutoencoderFeatureExtractor(
                epochs=20, batch_size=256, learning_rate=0.001, random_state=RANDOM_STATE
            )
        )))
    steps.append(('classifier', xgb.XGBClassifier(
        # Upper bound, the actual number of trees is set by early stopping in train.py
        n_estimators=5000,
        max_depth=6,
        learning_rate=0.05,
        min_child_weight=8,
        subsample=0.957,
        colsample_bytree=0.689,
        scale_pos_weight=scale_pos_weight,
        tree_method='hist',
        enable_categorical=True,
        random_state=RANDOM_STATE,
        eval_metric='aucpr'
    )))
    return Pipeline(steps)
