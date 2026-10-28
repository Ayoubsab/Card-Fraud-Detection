import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

DEVICE_MAPPING = {
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


class DevicePreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self, min_frequency=250):
        self.min_frequency = min_frequency

    @staticmethod
    def _device_info(X):
        '''
        Return DeviceInfo with empty strings treated as missing.
        If the column is absent, every row is missing.
        '''
        if 'DeviceInfo' not in X.columns:
            return pd.Series(np.nan, index=X.index, dtype=object)
        return X['DeviceInfo'].astype(object).replace("", np.nan)

    def _clean_names(self, device_info):
        devices = device_info.fillna("missing").astype(str).str.lower()
        device_names = devices.str.split('/', expand=True)[0]

        for pattern, brand in DEVICE_MAPPING.items():
            mask = device_names.str.contains(pattern, regex=True, na=False)
            device_names.loc[mask] = brand

        return device_names

    def fit(self, X, y=None):
        '''
        It finds out which devices actually appear >= min_frequency times.
        '''
        cleaned_names = self._clean_names(self._device_info(X))

        # Calculate frequencies on the full batch
        counts = cleaned_names.value_counts()

        self.valid_devices_ = set(counts[counts >= self.min_frequency].index)

        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self, 'valid_devices_')

        X = X.copy()
        device_info = self._device_info(X)

        X['had_id'] = device_info.notna().astype(int)  # Add a device flag

        mapped_names = self._clean_names(device_info)

        X['device_name'] = np.where(
            mapped_names.isin(self.valid_devices_),
            mapped_names,
            'Others'
        )

        return X.drop(columns='DeviceInfo', errors='ignore')
