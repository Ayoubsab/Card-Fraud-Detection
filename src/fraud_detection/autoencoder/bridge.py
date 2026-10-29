import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import KFold

from fraud_detection.config import RANDOM_STATE

class AutoencoderPipeline(BaseEstimator, TransformerMixin):
    def __init__(self, ae_extractor, n_splits=3):
        self.ae_extractor = ae_extractor
        self.n_splits = n_splits

    def fit(self, X, y=None):

        self.numeric_cols_ = list(X.select_dtypes(include='number').columns)
        # Train AE only on numeric data
        self.ae_extractor.fit(X[self.numeric_cols_], y)
        return self

    def transform(self, X):
        check_is_fitted(self, 'numeric_cols_')
        X_out = X.copy()

        # Get the reconstruction error from the numeric data
        X_with_ae = self.ae_extractor.transform(X_out[self.numeric_cols_])
        # Append just the new error column to our main dataframe
        X_out['ae_reconstruction_error'] = X_with_ae['ae_reconstruction_error']
        return X_out

    def fit_transform(self, X, y=None):
        '''
        Out-of-fold reconstruction errors for the training rows: each row is scored by an
        autoencoder that never saw it, so the errors XGBoost trains on look like the errors
        it gets at inference. The final autoencoder (used by `transform`) is fit on all rows.
        '''
        self.numeric_cols_ = list(X.select_dtypes(include='number').columns)
        X_num = X[self.numeric_cols_]
        oof_error = np.empty(len(X))

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=RANDOM_STATE)

        for train_idx, val_idx in kf.split(X_num):
            fold_ae_extractor = clone(self.ae_extractor)
            fold_ae_extractor.fit(X_num.iloc[train_idx], None if y is None else y.iloc[train_idx])
            fold_errors = fold_ae_extractor.transform(X_num.iloc[val_idx])['ae_reconstruction_error']
            oof_error[val_idx] = fold_errors.to_numpy()

        self.fit(X, y)
        X_out = X.copy()
        X_out['ae_reconstruction_error'] = oof_error

        return X_out