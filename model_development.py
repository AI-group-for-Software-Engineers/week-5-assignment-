# model_development.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv('student_data.csv')  # Replace with your actual dataset

# Preprocessing (example)
df.fillna(0, inplace=True)
df = pd.get_dummies(df, drop_first=True)

# Features and target
X = df.drop('dropout', axis=1)
y = df['dropout']

# Split data
X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)