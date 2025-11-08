import pandas as pd
from preprocessing import basic_cleaning, feature_engineering

# Create a small mock dataset
data = {
    'timestamp': ['2023-01-01 10:00', '2023-01-01 11:30', None, '2023-01-02 09:45'],
    'amount': [2000, -500, 3000, 4500],
    'payer_id': ['A01', 'A02', 'A03', 'A04'],
    'payee_id': ['B01', 'B02', 'B03', 'B04']
}

df = pd.DataFrame(data)

print("=== Original Data ===")
print(df)

df_clean = basic_cleaning(df)
print("\n=== After basic_cleaning ===")
print(df_clean)

df_features = feature_engineering(df_clean)
print("\n=== After feature_engineering ===")
print(df_features)


