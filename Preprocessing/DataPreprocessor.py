import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
import os


class DataPreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self, nan_threshold=0.01, memory_threshold=100):
        self.nan_threshold = nan_threshold
        self.memory_threshold = memory_threshold
        self.fill_values = {}
        self.flag_cols = []
        self.cat_cols = []


    def _reduce_memory(self, df):
        '''
        Reduce the memory usage (size of the df) by changing the type of each column. 
        '''
        
        for col in df.columns:
            col_type = df[col].dtype
            
            if col_type != object and col_type.name != 'category':
                c_min = df[col].min()
                c_max = df[col].max()
                
                if str(col_type).startswith('int'):
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)  
                
                else:
                    if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
                        

            elif col_type == object:
                df[col] = df[col].astype('category')

        return df

    def fit(self, X, y=None):
        
        self.cat_cols = X.select_dtypes(include=['object', 'category']).columns
        self.flag_cols = [
            col for col in X.columns
            if col not in self.cat_cols and X[col].isna().mean() > self.nan_threshold
        ]

        for col in X.select_dtypes(include='number').columns:
            self.fill_values[col] = X[col].median()

        return self


    def transform(self, X):
        
        df = X.copy()

        # --- Feature Engineering ---
        if "TransactionAmt" in df.columns: 
            df["TransactionAmt_log"] = np.log1p(df["TransactionAmt"])
            amt_filled = df['TransactionAmt'].fillna(0) # There is no NaN value in training data, but we may have it while testing the model
            
            # Keep only the decimal part of the transaction amount
            df['TransactionAmt_decimal'] = ((amt_filled - amt_filled.astype(int)) * 1000).astype(int)

        # Keep only the decimal part of the transaction amount
        if "TransactionDT" in df.columns:
            df["hour_of_day"] = (df["TransactionDT"] // 3600) % 24
            df["day_of_week"]  = (df["TransactionDT"] // (3600 * 24)) % 7

        # Extract the suffix after the first dot from the emails 
        #  'gmail.com'      → 'com'
        #  'yahoo.co.uk'    → 'co.uk'
        if "P_emaildomain" in df.columns:
            df['P_email_suffix'] = df["P_emaildomain"].str.split(".", n=1).str[1]
            df['P_email_suffix'] = df['P_email_suffix'].fillna("missing")

        if "R_emaildomain" in df.columns:
            df['R_email_suffix'] = df["R_emaildomain"].str.split(".", n=1).str[1]
            df['R_email_suffix'] = df['R_email_suffix'].fillna("missing")

        if "P_emaildomain" in df.columns and "R_emaildomain" in df.columns:
            # Check if the emails match
            df["email_match"] = (
                df["P_emaildomain"].astype(str) == df["R_emaildomain"].astype(str)
            ).astype(int)

        # Get the OS and version
        if "id_30" in df.columns:
            df['OS'] = df['id_30'].str.split(' ', expand=True)[0]
            df['version_OS'] = df['id_30'].str.split(' ', expand=True)[1]
            df['OS'] = df['OS'].fillna("missing")
            df['version_OS'] = df['version_OS'].fillna("missing")
        
        # Get the browser and version
        if "id_31" in df.columns:
            df['browser'] = df['id_31'].str.split(' ', expand=True)[0]
            df['version_browser'] = df['id_31'].str.split(' ', expand=True)[1]
            df['browser'] = df['browser'].fillna("missing")
            df['version_browser'] = df['version_browser'].fillna("missing")

        # Get the screen width and height
        if "id_33" in df.columns:
            df['screen_width'] = df['id_33'].str.split('x', expand=True)[0]
            df['screen_height'] = df['id_33'].str.split('x', expand=True)[1]
            df['screen_width']  = pd.to_numeric(df['screen_width'],  errors='coerce')
            df['screen_height'] = pd.to_numeric(df['screen_height'], errors='coerce')
        
        # Get the value of match_status
        if "id_34" in df.columns:
            df['id_34'] = df['id_34'].str.split(':', expand=True)[1]


        # --- Handle NaN values ---
        # 1- Categorical columns
        for col in self.cat_cols:
            if col in df.columns:
                if df[col].dtype.name == 'category' and "missing" not in df[col].cat.categories:
                    df[col] = df[col].cat.add_categories("missing")
                df[col] = df[col].fillna("missing")

        object_cols = df.select_dtypes(include=['object']).columns
        for col in object_cols:
            df[col] = df[col].astype('category')
        
        # 2- Numerical columns
        flagged_cols = {}
        for col in self.flag_cols:
            if col in df.columns:
                flagged_cols[f"{col}_is_nan"] = df[col].isna().astype(np.int8)
        
        for col in df.select_dtypes(include='number').columns:
            df[col] = df[col].fillna(self.fill_values.get(col, 0))

        if flagged_cols:
            new_flags_df = pd.DataFrame(flagged_cols, index=df.index)
            df = pd.concat([df, new_flags_df], axis=1)

        # --- Reduce memory usage ---
        if len(df) > self.memory_threshold:
            df = self._reduce_memory(df)

        return df

        


    


