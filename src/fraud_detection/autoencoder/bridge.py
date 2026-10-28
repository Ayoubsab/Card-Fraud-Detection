from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class AutoencoderPipeline(BaseEstimator, TransformerMixin):
    def __init__(self, ae_extractor):
        self.ae_extractor = ae_extractor

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
