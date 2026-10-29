import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

SECONDS_PER_DAY = 86400

# Frequency encoded columns: how common this card / region / email domain is
FREQ_COLS = ["card1", "card2", "addr1", "P_emaildomain"]
NUMERIC_FREQ_COLS = {"card1", "card2", "addr1"}

# Number of distinct values per customer (many emails or devices on one card is suspicious)
NUNIQUE_COLS = {"P_emaildomain": "uid_email_nunique", "DeviceInfo": "uid_device_nunique"}


def _numeric(X, col):
    '''Column as floats; absent columns and API nulls become NaN.'''
    if col not in X.columns:
        return pd.Series(np.nan, index=X.index, dtype=float)
    return pd.to_numeric(X[col], errors='coerce').astype(float)


def build_uid(X: pd.DataFrame) -> pd.DataFrame:
    '''
    Rebuild a pseudo customer id, since the dataset has none.

    D1 is "days since the card was first used", so day - D1 is the day the card was
    first used: a constant per card. Combined with card1 (card) and addr1 (billing
    region) it identifies a customer. uid is missing when any of its parts is missing,
    otherwise all customers without an address would be merged together.
    '''
    day = np.floor(_numeric(X, 'TransactionDT') / SECONDS_PER_DAY)
    d1n = day - _numeric(X, 'D1')

    parts = [_numeric(X, 'card1'), _numeric(X, 'addr1'), d1n]
    valid = parts[0].notna() & parts[1].notna() & parts[2].notna()

    # Nullable integers so keys read "13926_315_-9" rather than "13926.0_315.0_-9.0"
    as_str = [part.round().astype('Int64').astype(str) for part in parts]
    uid = (as_str[0] + "_" + as_str[1] + "_" + as_str[2]).where(valid)

    return pd.DataFrame({'D1n': d1n, 'uid': uid}, index=X.index)


class UidAggregator(BaseEstimator, TransformerMixin):
    '''
    Customer level features: per-uid statistics and frequency encodings.

    The statistics are computed on the training data in `fit` and stored as lookup
    tables, so a single API request can be enriched without any history. Customers
    unseen in training get NaN statistics, unseen frequency values get 0.
    The uid itself is never returned: it would be memorized and not generalize.
    '''

    def _freq_key(self, X, col):
        if col in NUMERIC_FREQ_COLS:
            return _numeric(X, col)
        if col not in X.columns:
            return pd.Series(np.nan, index=X.index, dtype=object)
        return X[col].astype(object).replace("", np.nan)

    def fit(self, X, y=None):
        keys = build_uid(X)

        frame = pd.DataFrame({'uid': keys['uid'], 'amt': _numeric(X, 'TransactionAmt')})
        for col in NUNIQUE_COLS:
            frame[col] = X[col].astype(object) if col in X.columns else np.nan

        groups = frame.dropna(subset=['uid']).groupby('uid')
        stats = pd.DataFrame({
            'uid_count': groups.size(),
            'uid_amt_mean': groups['amt'].mean(),
            'uid_amt_std': groups['amt'].std(),
        })
        for col, name in NUNIQUE_COLS.items():
            stats[name] = groups[col].nunique()
        self.uid_stats_ = stats.astype(np.float32)

        self.freq_maps_ = {col: self._freq_key(X, col).value_counts() for col in FREQ_COLS}

        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self, 'uid_stats_')
        keys = build_uid(X)

        new_cols = self.uid_stats_.reindex(keys['uid'].to_numpy()).set_axis(X.index)
        new_cols['D1n'] = keys['D1n']

        # Is this amount unusual for this customer?
        new_cols['amt_to_uid_mean'] = _numeric(X, 'TransactionAmt') / new_cols['uid_amt_mean']

        for col in FREQ_COLS:
            new_cols[f'{col}_freq'] = self._freq_key(X, col).map(self.freq_maps_[col]).fillna(0)

        return pd.concat([X, new_cols], axis=1)
