from sklearn.base import BaseEstimator, TransformerMixin

class AutoencoderPipeline(BaseEstimator, TransformerMixin):
    def __init__(self, ae_extractor):
        self.ae_extractor = ae_extractor
        self.numeric_cols = None
        
    def fit(self, X, y=None):

        self.numeric_cols = X.select_dtypes(include=['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']).columns
        # Train AE only on numeric data
        self.ae_extractor.fit(X[self.numeric_cols], y)
        return self
        
    def transform(self, X):
        X_out = X.copy()
        # Get the reconstruction error from the numeric data
        X_with_ae = self.ae_extractor.transform(X[self.numeric_cols])
        # Append just the new error column to our main dataframe
        X_out['ae_reconstruction_error'] = X_with_ae['ae_reconstruction_error']
        return X_out