# Card Fraud Detection

End-to-end fraud detection on real e-commerce transactions: feature engineering, an XGBoost model evaluated on a strictly future time period, and a FastAPI service that scores single transactions exactly like the training data.

**Test month results:** AUPRC **0.610** (17× random), ROC-AUC **0.913**. Flagging 1% of legitimate transactions catches **55%** of fraud.

---

## Problem

The [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection/data) dataset has 590,540 card transactions from Vesta, about 430 raw features (transaction, card, email, device and anonymized V/C/D/M columns), and a **3.5% fraud rate**.

With this class imbalance, accuracy means nothing: predicting "never fraud" scores 96.5%. The main metric is **AUPRC** (area under the precision-recall curve). A random model scores the fraud rate, 0.035. I also report **recall at 1% false positive rate**: how much fraud is caught when only 1% of legitimate customers get flagged. That's the kind of limit a fraud team actually works with.

## Results

All models are evaluated on the same untouched **last month** of data (88,581 transactions, 3,083 frauds).

| Model | AUPRC | ROC-AUC | Recall @ 1% FPR |
|---|---|---|---|
| Random | 0.035 | 0.500 | 1% |
| XGBoost on engineered features | 0.535 | 0.891 | 45.6% |
| + customer (uid) features | 0.551 | 0.889 | 48.1% |
| **+ refit on train + validation (final model)** | **0.610** | **0.913** | **54.8%** |

### Experiments that did not make it into the final model

| Experiment | AUPRC | Finding |
|---|---|---|
| Autoencoder anomaly score as an XGBoost feature | 0.521 | No gain. XGBoost on 737 features already captures "unusual"; the reconstruction error alone reaches 0.11 AUPRC and drifts over time |
| Same, with out-of-fold errors | 0.521 | Ruled out in-sample leakage: training rows were only ~3% "cleaner" than unseen rows. The train/test gap was temporal drift |
| Past-only customer statistics (running sums) | 0.542 | Worse than the lookup tables. With no history before day 1, "first transaction" mostly meant "early in the dataset", not "new customer" |

The autoencoder code is still in the repo (`--autoencoder`) as a documented experiment.

## Approach

```
raw transaction + identity
        │
        ▼
UidAggregator         pseudo customer id → per-customer stats, frequency encodings
DevicePreprocessor    device brand mapping, rare devices → "Others", had_id flag
DataPreprocessor      feature engineering, NaN flags + fill, schema enforcement
        │
        ▼
XGBoost               scale_pos_weight, early stopping on validation AUPRC
```

XGBoost's tree hyperparameters (depth, min_child_weight, subsample, colsample_bytree) were tuned with Optuna (Bayesian optimization) in [notebooks/02_modeling.ipynb](notebooks/02_modeling.ipynb).

**Customer features.** The data has no customer id, but fraud is labeled per customer: once a card is compromised, its later transactions are fraud too. `D1` is "days since the card's first transaction", so `day - D1` is constant per card, and `card1 + addr1 + (day - D1)` gives a pseudo customer id. In a 100k-transaction sample, 76% of customers with any fraud had a majority of fraudulent transactions. The model never sees the id itself, only aggregates: transaction count, amount mean and standard deviation, **amount relative to the customer's average**, distinct emails and devices per card, and frequency encodings. They are stored as lookup tables at fit time, so the API can enrich a single request without any history.

**Evaluation.** The data is split **by time** (70% train / 15% validation / 15% test) because production always predicts the future. A random split would leak future fraud patterns into training. Validation picks the number of trees (early stopping) and the decision threshold (best F1). The model is then refit on train + validation and **evaluated once on test**.

**Training/serving consistency.** The preprocessors learn their full output schema in `fit` (columns, order, dtypes, category levels) and enforce it in `transform`. An API request with missing fields, empty strings, unseen categories or junk numbers is processed exactly like a training row. I verified this by checking that single-row output matches batch output on 737 columns.

## API

```bash
uvicorn fraud_detection.api.app:app
```

| Endpoint | |
|---|---|
| `GET /health` | Model status, version (git commit) and training date; `degraded` if no model is loaded |
| `POST /predict` | Fraud probability and decision, using the threshold chosen on validation |
| `GET /docs` | Interactive Swagger UI |

```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
     -d '{"TransactionID": 1, "TransactionDT": 86400, "TransactionAmt": 50.0, "card1": 13926}'
```
```json
{"fraud_probability": 0.0475, "is_fraud": false, "threshold": 0.4612, "model_version": "c28dadc"}
```

Fields left out are treated as missing. Unknown fields are rejected with a 422, so a typo like `TransactionAmnt` doesn't silently drop the amount. Errors are logged on the server, and clients only get a generic message.

## Quickstart

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install -e .

# Download train_transaction.csv and train_identity.csv from Kaggle into data/
python -m fraud_detection.train                      # full dataset, writes models/
python -m fraud_detection.train --sample-rows 50000  # quick run
uvicorn fraud_detection.api.app:app
```

Each run writes `models/metrics.json` (test metrics) and `models/metadata.json` (threshold, data splits, package versions, git commit, with `-dirty` when trained from uncommitted code).

## Project structure

```
src/fraud_detection/
├── config.py               paths, overridable with FRAUD_DATA_DIR / FRAUD_MODEL_PATH
├── data.py                 loads and joins transactions + identity
├── pipeline.py             builds the sklearn pipeline
├── train.py                time split, early stopping, threshold, refit, metrics
├── features/
│   ├── uid.py              customer id and aggregates
│   ├── device.py           device cleaning
│   └── preprocessor.py     feature engineering and schema enforcement
├── autoencoder/            experiment: PyTorch autoencoder anomaly score
└── api/
    ├── app.py              FastAPI service
    └── schemas.py          request / response models
notebooks/                  EDA and modeling exploration
```

## Limitations and next steps

- **Label delay.** The refit uses the last month's labels. In real life chargebacks take weeks to arrive, so a production model would train on slightly older labels.
- **Threshold after refit.** The threshold is chosen on the train-only model and reused after the refit, whose scores shift slightly. Calibration, or choosing the threshold by a target false-positive rate, would make the operating point more stable.
- **Customer history is frozen at training time.** A production system would update customer statistics continuously through a feature store. That would also enable velocity features ("transactions in the last hour").
- **Next:** unit tests and CI, a batch scoring endpoint, a Docker image, and a LightGBM/CatBoost ensemble.

macOS note: torch is pinned below 2.11, whose bundled OpenMP runtime segfaults when xgboost runs in the same process.
