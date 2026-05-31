import pandas as pd
import numpy as np

train = pd.read_csv("dataset/train.csv")

# Extract overlapping timestamps (00:00 to 02:00)
df_48 = train[(train['day'] == 48) & (train['timestamp'].isin(['0:0', '0:15', '0:30', '0:45', '1:0', '1:15', '1:30', '1:45', '2:0']))]
df_49 = train[train['day'] == 49]

# Merge on geohash and timestamp
merged = pd.merge(df_48, df_49, on=['geohash', 'timestamp'], suffixes=('_48', '_49'))

print("--- Correlation between Day 48 and Day 49 Demand (Same Geohash & Timestamp) ---")
correlation = merged['demand_48'].corr(merged['demand_49'])
print(f"Pearson Correlation: {correlation:.4f}")

print("\nSample comparison:")
print(merged[['geohash', 'timestamp', 'demand_48', 'demand_49']].head(15))



