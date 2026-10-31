import pandas as pd
import pytest


def make_request(**fields) -> pd.DataFrame:
    """One transaction as the API builds it: a single-row DataFrame with only the fields sent."""
    return pd.DataFrame([{"TransactionDT": 86400, "TransactionAmt": 50.0, **fields}])


def assert_no_nan_in_numeric(out, fitted_features):
    numeric_cols = fitted_features[-1].numeric_cols_
    assert not out[numeric_cols].isna().any().any()


@pytest.mark.parametrize("row", [0, 150, 299])
def test_single_row_matches_batch(fitted_features, raw_transactions, row):
    X = raw_transactions.drop(columns="isFraud")

    batch = fitted_features.transform(X)
    single = fitted_features.transform(X.iloc[[row]])

    # Checks values, column order, dtypes and category levels
    pd.testing.assert_frame_equal(single, batch.iloc[[row]])


def test_request_with_only_required_fields(fitted_features):
    out = fitted_features.transform(make_request())

    assert list(out.columns) == fitted_features[-1].feature_names_out_
    assert_no_nan_in_numeric(out, fitted_features)


def test_unseen_and_empty_categories_become_missing(fitted_features):
    out = fitted_features.transform(make_request(ProductCD="Z", P_emaildomain=""))

    assert out["ProductCD"].iloc[0] == "missing"
    assert out["P_emaildomain"].iloc[0] == "missing"


def test_junk_number_is_treated_as_missing(fitted_features):
    out = fitted_features.transform(make_request(TransactionAmt="abc"))

    assert_no_nan_in_numeric(out, fitted_features)
