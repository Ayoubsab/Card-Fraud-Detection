import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

MISSING = "missing"

# Raw columns replaced by engineered features
RAW_COLS_TO_DROP = ["TransactionAmt", "id_30", "id_31", "id_33"]


class DataPreprocessor(BaseEstimator, TransformerMixin):
    '''
    Feature engineering + NaN handling.

    The output schema (columns, order, dtypes, category levels) is learned in `fit`
    and enforced in `transform`, so a single API request is processed exactly
    like a row of the training set.
    '''

    def __init__(self, nan_threshold=0.01):
        self.nan_threshold = nan_threshold

    def _clean_inputs(self, X):
        '''
        Align the input to the columns seen in fit and enforce their types:
        omitted columns become NaN, numeric columns are coerced to numbers
        and empty strings in categorical columns are treated as missing.
        '''
        df = X.reindex(columns=self.input_columns_)

        for col in self.raw_numeric_cols_:
            if not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors='coerce')

        for col in self.raw_cat_cols_:
            df[col] = df[col].astype(object).replace("", np.nan)

        return df

    def _engineer_features(self, df):

        if "TransactionAmt" in df.columns:
            df["TransactionAmt_log"] = np.log1p(df["TransactionAmt"])
            amt_filled = df['TransactionAmt'].fillna(0)

            # Keep only the decimal part of the transaction amount
            df['TransactionAmt_decimal'] = ((amt_filled - amt_filled.astype(int)) * 1000).astype(int)

        if "TransactionDT" in df.columns:
            df["hour_of_day"] = (df["TransactionDT"] // 3600) % 24
            df["day_of_week"] = (df["TransactionDT"] // (3600 * 24)) % 7

        # Extract the suffix after the first dot from the emails
        #  'gmail.com'      → 'com'
        #  'yahoo.co.uk'    → 'co.uk'
        if "P_emaildomain" in df.columns:
            df['P_email_suffix'] = df["P_emaildomain"].str.split(".", n=1).str[1]

        if "R_emaildomain" in df.columns:
            df['R_email_suffix'] = df["R_emaildomain"].str.split(".", n=1).str[1]

        if "P_emaildomain" in df.columns and "R_emaildomain" in df.columns:
            # Check if the emails match
            df["email_match"] = (
                df["P_emaildomain"].astype(str) == df["R_emaildomain"].astype(str)
            ).astype(int)

        # Get the OS and version
        if "id_30" in df.columns:
            split = df['id_30'].str.split(' ', n=1, expand=True).reindex(columns=[0, 1])
            df['OS'] = split[0]
            df['version_OS'] = split[1]

        # Get the browser and version
        if "id_31" in df.columns:
            split = df['id_31'].str.split(' ', expand=True).reindex(columns=[0, 1])
            df['browser'] = split[0]
            df['version_browser'] = split[1]

        # Get the screen width and height
        if "id_33" in df.columns:
            split = df['id_33'].str.split('x', expand=True).reindex(columns=[0, 1])
            df['screen_width'] = split[0]
            df['screen_height'] = split[1]

        return df.drop(columns=RAW_COLS_TO_DROP, errors='ignore')

    def fit(self, X, y=None):

        self.input_columns_ = list(X.columns)
        self.raw_numeric_cols_ = list(X.select_dtypes(include='number').columns)
        self.raw_cat_cols_ = [col for col in self.input_columns_ if col not in self.raw_numeric_cols_]

        df = self._engineer_features(self._clean_inputs(X))

        self.numeric_cols_ = list(df.select_dtypes(include='number').columns)
        self.cat_cols_ = [col for col in df.columns if col not in self.numeric_cols_]

        self.flag_cols_ = [
            col for col in self.numeric_cols_ if df[col].isna().mean() > self.nan_threshold
        ]
        self.fill_values_ = df[self.numeric_cols_].median().fillna(0.0).to_dict()

        # "missing" is always a valid level, unseen values at inference are mapped to it
        self.categories_ = {
            col: sorted(set(df[col].dropna().astype(str)) | {MISSING})
            for col in self.cat_cols_
        }

        self.feature_names_out_ = list(df.columns) + [f"{col}_is_nan" for col in self.flag_cols_]
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self, 'feature_names_out_')

        df = self._engineer_features(self._clean_inputs(X))

        # --- Handle NaN values ---
        # 1- Numerical columns: flag NaN before filling to keep "absence" as a signal
        flags = pd.DataFrame(
            {f"{col}_is_nan": df[col].isna().astype(np.int8) for col in self.flag_cols_},
            index=df.index,
        )
        df[self.numeric_cols_] = df[self.numeric_cols_].fillna(self.fill_values_).astype(np.float32)

        # 2- Categorical columns: fixed levels learned in fit
        for col in self.cat_cols_:
            values = df[col].astype(object)
            values = values.where(values.notna(), MISSING).astype(str)
            values = values.where(values.isin(self.categories_[col]), MISSING)
            df[col] = pd.Categorical(values, categories=self.categories_[col])

        df = pd.concat([df, flags], axis=1)

        return df[self.feature_names_out_]

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, 'feature_names_out_')
        return np.asarray(self.feature_names_out_, dtype=object)
