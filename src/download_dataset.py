"""
Download California Housing dataset and save as CSV
"""

import pandas as pd
from sklearn.datasets import fetch_california_housing

print("Loading California Housing dataset...")
housing = fetch_california_housing()

# Create DataFrame
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['MedHouseVal'] = housing.target

# Save to CSV
df.to_csv('data/housing.csv', index=False)
print(f"Dataset saved to data/housing.csv")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")