import pytest

from fraud_detection.train import best_f1_threshold, recall_at_fpr, time_split


def test_time_split_never_puts_future_rows_in_train(raw_transactions):
    raw_transactions = raw_transactions.sample(frac=1, random_state=0)

    train, val, test = time_split(raw_transactions)

    max_train = train["TransactionDT"].max()
    min_val = val["TransactionDT"].min()
    max_val = val["TransactionDT"].max()
    min_test = test["TransactionDT"].min()

    assert max_train <= min_val
    assert max_val <= min_test
    assert len(train) + len(val) + len(test) == len(raw_transactions)


def test_best_f1_threshold_on_perfectly_separated_scores():
    y = [0, 0, 1, 1]
    probs = [0.1, 0.2, 0.8, 0.9]

    f1_threshold = best_f1_threshold(y, probs)

    assert f1_threshold == pytest.approx(0.8)


def test_recall_at_fpr_is_one_when_scores_are_perfect():
    y = [0, 0, 1, 1]
    probs = [0.1, 0.2, 0.8, 0.9]

    share_fraud = recall_at_fpr(y, probs, 0.01)

    assert share_fraud == 1
