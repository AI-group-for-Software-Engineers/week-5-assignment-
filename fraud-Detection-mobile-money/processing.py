"""
processing.py
Data cleaning and basic feature engineering utilities.
"""

import pandas as pd
import numpy as np

def basic_cleaning(df):
    # Ensure timestamp is datetime and remove duplicate rows
    df = df.copy()
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.drop_duplicates()
    return df

def feature_engineering(df):
    df = df.copy()

    if 'timestamp' in df.columns:
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek

    df['amount_log'] = np.log1p(df['amount'].fillna(0))

    return df
