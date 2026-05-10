import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import re

class DevicePreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self, min_frequency=250):
        self.min_frequency = min_frequency
        self.valid_devices = set()
        self.device_mapping = {
            r'sm|samsung|gt-': 'Samsung',
            r'moto|xt': 'Motorola',
            r'lg-': 'LG',
            r'rv:': 'RV',
            r'huawei|ale-|-l': 'Huawei',
            r'blade': 'ZTE',
            r'linux': 'Linux',
            r'htc': 'HTC',
            r'asus': 'Asus'
        }

    def _clean_names(self, df):
        devices = df['DeviceInfo'].fillna("missing").str.lower()
        device_names = devices.str.split('/', expand=True)[0]

        for pattern, brand in self.device_mapping.items():
            mask = device_names.str.contains(pattern, regex=True, na=False)

            device_names.loc[mask] = brand

        return device_names

    def fit(self, X, y=None):
        '''
        It finds out which devices actually appear >= 250 times. 
        '''

        if 'DeviceInfo' not in X.columns:
            return self
        
        cleaned_names = self._clean_names(X)
        
        # Calculate frequencies on the full batch
        counts = cleaned_names.value_counts()
        
        self.valid_devices = set(counts[counts >= self.min_frequency].index)
        
        self.n_features_in_ = X.shape[1]
        return self
    
    def transform(self, X):
        
        X = X.copy()

        if 'DeviceInfo' not in X.columns:
            return X
        
        X['had_id'] = X['DeviceInfo'].notna().astype(int) # Add a device flag

        mapped_names = self._clean_names(X)

        X['device_name'] = np.where(
            mapped_names.isin(self.valid_devices), 
            mapped_names, 
            'Others'
        )

        X = X.drop('DeviceInfo', axis=1)

        return X

