# Card-Fraud-Detection

A fraud detection system combining an **unsupervised PyTorch autoencoder** with a supervised **XGBoost classifier**, served via a FastAPI endpoint.

---

## Problem 

Credit card fraud is a classic **extreme class imbalance** problem. On the Kaggle dataset, only 0.17% of transactions are fraudulent (492 out of 284,807). This means:

- Accuracy is a useless metric — a model predicting "all legit" scores 99.83%
- The right metric is **AUPRC** (Area Under Precision-Recall Curve)

Two complementary approaches are explored:
1. **Autoencoder (unsupervised)** — trained only on normal transactions; fraud = high reconstruction error
2. **XGBoost (supervised)** — uses labels directly with cost-sensitive weighting

---

## Dataset

[IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection/data) — real e-commerce transactions from Vesta Corporation.

| Stat | Value |
|---|---|
| Total transactions | 590,540 |
| Fraudulent | ~20,663 (3.5%) |
| Transaction features | ~390 (V, C, D, M columns + card/email/address) |
| Identity features | 41 (device, browser — joined on TransactionID) |
| Identity coverage | ~25% of transactions have identity rows |
| Label | `isFraud` (0 = legit, 1 = fraud) |


Download `train_transaction.csv` and `train_identity.csv` and place both in `data/`.

---

## Pipeline architecture

```
Raw data (train_transaction.csv + train_identity.csv)
       ↓
DevicePreprocessor     — brand mapping, frequency filtering, device flag
       ↓
DataPreprocessor       — feature engineering, NaN flags + fill, memory reduction
       ↓
Autoencoder      — trains on normal txns only, adds ae_reconstruction_error feature
       ↓
XGBClassifier          — scale_pos_weight, hist tree method, AUCPR optimization
       ↓
FastAPI endpoint       — calibrated probabilities 
```

**Key engineering decisions:**
- `fit`/`transform` separation on all preprocessors —> no data leakage
- NaN flags added before filling —> preserves "absence" as signal
- Autoencoder trained exclusively on non-fraud transactions
- `ae_reconstruction_error` injected as a feature into XGBoost (semi-supervised)
- Used Optuna (Bayesian optimization) to optimize XGBoost's hyperparameters

---

## Feature engineering

| Feature | Source | Signal |
|---|---|---|
| `TransactionAmt_log` | TransactionAmt | Normalizes skewed distribution |
| `TransactionAmt_decimal` | TransactionAmt | 
| `hour_of_day`, `day_of_week` | TransactionDT | Time od the day/week | 
| `email_match` | P/R emaildomain | Mismatch = strong fraud signal |
| `P_email_suffix`, `R_email_suffix` | P/R emaildomain | Country-level domain signal |
| `OS`, `browser` | id_30, id_31 | Device fingerprinting |
| `screen_width`, `screen_height` | id_33 | Device fingerprinting |
| `device_name` | DeviceInfo | Brand-level device signal |
| `had_id` | DeviceInfo | Missing fingerprint = fraud signal |
| `ae_reconstruction_error` | Autoencoder | 
| `*_is_nan` flags | Various | NaN as explicit feature |

---

## Results

| Model | AUPRC | Lift over baseline | Notes |
|---|---|---|---|
| Random baseline | 0.035 | 1x | Fraud rate |
| XGBoost | 0.514 | — | Full pipeline |
| XGBoost + Autoencoder | 0.518 | 15x | Full hybrid pipeline |

---

## Repo structure

```
fraud-detection/
├── data/
│   ├── train_transaction.csv   
│   └── train_identity.csv      
├── Preprocessing/
│   ├── DevicePreprocessor.py   
│   ├── DataPreprocessor.py     
├── Scripts/            
│   ├── AutoEncoder.py   
│   ├── AutoencoderFeatureExtractor.py             
│   └── Scripts/AutoencoderPipeline.py
├── models/                     # saved artifacts (gitignored)
├── train.py
├── main.py
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
pip install -r requirements.txt

# 1. Download from Kaggle (free account + competition sign-up required)
#    https://www.kaggle.com/competitions/ieee-fraud-detection/data

# 2. Train model
python train.py

# 3. Start API
python main.py
# GET  http://localhost:8000/docs   ← interactive Swagger UI
# POST http://localhost:8000/predict
```

---

