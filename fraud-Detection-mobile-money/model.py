"""
model.py
Basic example training script for fraud detection.
Replace synthetic data with real dataset when available.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from sklearn.linear_model import LogisticRegression
import joblib

from processing import basic_cleaning, feature_engineering   # ✅ UPDATED IMPORT

def load_data():
    # Example synthetic dataset
    n = 10000
    rng = np.random.RandomState(42)
    df = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=n, freq='min'),  # ✅ avoid deprecated 'T'
        'amount': rng.exponential(scale=200, size=n),
        'payer_id': rng.randint(1, 3000, size=n),
        'payee_id': rng.randint(1, 3000, size=n),
    })
    df['label'] = (rng.rand(n) < 0.01).astype(int)  # ~1% fraud rate
    return df

def train():
    df = load_data()

    # Preprocess
    df = basic_cleaning(df)
    df = feature_engineering(df)

    X = df[['amount_log', 'hour', 'day_of_week']]
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = LogisticRegression()
    model.fit(X_train, y_train)

    preds = model.predict_proba(X_test)[:, 1]
    ap = average_precision_score(y_test, preds)

    print("AUPRC:", round(ap, 4))
    
    joblib.dump(model, "fraud_model.joblib")
    print("Model saved as fraud_model.joblib")

if __name__ == "__main__":
    train()

