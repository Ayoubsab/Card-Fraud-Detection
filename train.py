import sys
sys.path.append('..')

import pandas as pd
import gc
import xgboost as xgb
import joblib
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score, classification_report
from Preprocessing.DataPreprocessor import DataPreprocessor
from Preprocessing.DevicePreprocessor import DevicePreprocessor
from Scripts.AutoEncoder import AutoEncoder
from Scripts.AutoencoderFeatureExtractor import AutoencoderFeatureExtractor
from Scripts.AutoencoderPipeline import AutoencoderPipeline


def load_data(data_path='./data/'):
    """Loads and merges transaction and identity datasets."""
    print("Loading data...")
    train_id = pd.read_csv(f'{data_path}train_identity.csv')
    train_trans = pd.read_csv(f'{data_path}train_transaction.csv')
    
    df = pd.merge(train_trans, train_id, on='TransactionID', how='left')
    
    # Cleanup memory immediately
    del train_id, train_trans
    gc.collect()
    return df


def train_model():

    data = load_data()
    y = data['isFraud']
    X = data.drop(['isFraud', 'TransactionID'], axis=1)
    
    # Time-based split: Use first 80% for training, last 20% for validation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    del X, y, data
    gc.collect()
    
    # Calculate scale_pos_weight to handle class imbalance
    imbalance_ratio = (len(y_train) - sum(y_train)) / sum(y_train)
    
    print("Assembling Hybrid Pipeline...")

    # This ensures the Autoencoder only fits on legitimate data to prevent leakage
    pipeline = Pipeline([
        ('device_preprocessor', DevicePreprocessor(min_frequency=250)),
        ('general_preprocessor', DataPreprocessor(nan_threshold=0.01)),
        ('autoencoder_bridge', AutoencoderPipeline(
            AutoencoderFeatureExtractor(epochs=1, batch_size=256, learning_rate=0.0001)
        )),
        ('classifier', xgb.XGBClassifier(
            n_estimators=7,
            max_depth=6,
            learning_rate=0.026,
            min_child_weight=8,
            subsample=0.957,
            colsample_bytree=0.689,
            scale_pos_weight=imbalance_ratio,
            tree_method='hist', 
            enable_categorical=True, 
            random_state=42,
            eval_metric='aucpr'
        ))
    ])

    # Training
    print("Beginning Training (this will take several minutes)...")
    pipeline.fit(X_train, y_train)

    # Evaluation
    print("\n--- Final Evaluation ---")
    y_probs = pipeline.predict_proba(X_test)[:, 1]
    auprc = average_precision_score(y_test, y_probs)
    
    print(f"PR-AUC (Average Precision): {auprc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, pipeline.predict(X_test)))

    # Serialization
    joblib.dump(pipeline, './models/fraud_detection_pipeline_v1.joblib')
    print("Model saved as 'fraud_detection_pipeline_v1.joblib'")

if __name__ == "__main__":
    train_model()