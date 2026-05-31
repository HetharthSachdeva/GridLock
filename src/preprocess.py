import pandas as pd
import numpy as np
import pygeohash as gh

def load_and_preprocess_data(train_path="dataset/train.csv", test_path="dataset/test.csv"):
    print("Loading data...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    # Identify test indices for splitting back later
    train['is_test'] = 0
    test['is_test'] = 1
    test['demand'] = np.nan
    
    # Combine datasets for unified preprocessing
    full_df = pd.concat([train, test], ignore_index=True)
    
    # 1. Temporal features
    print("Parsing timestamps...")
    def ts_to_minutes(ts):
        h, m = map(int, ts.split(':'))
        return h * 60 + m
    
    full_df['minutes'] = full_df['timestamp'].apply(ts_to_minutes)
    full_df['hour'] = full_df['minutes'] / 60.0
    
    # Cyclical trigonometric features for time of day
    full_df['sin_time'] = np.sin(2 * np.pi * full_df['minutes'] / 1440.0)
    full_df['cos_time'] = np.cos(2 * np.pi * full_df['minutes'] / 1440.0)
    
    # 2. Geohash decoding (Spatial features)
    print("Decoding geohashes...")
    # Cache geohash coordinates to speed up computation
    unique_geohashes = full_df['geohash'].unique()
    geohash_coords = {}
    for g in unique_geohashes:
        try:
            lat, lon = gh.decode(g)
            geohash_coords[g] = (lat, lon)
        except Exception:
            geohash_coords[g] = (np.nan, np.nan)
            
    full_df['latitude'] = full_df['geohash'].map(lambda g: geohash_coords[g][0])
    full_df['longitude'] = full_df['geohash'].map(lambda g: geohash_coords[g][1])
    
    # 3. Previous Day Demand Lookup (demand_last_day)
    print("Creating historical lag features...")
    # Group train day 48 demand by geohash and timestamp
    df_48 = train[train['day'] == 48]
    day48_lookup = df_48.groupby(['geohash', 'timestamp'])['demand'].mean().reset_index()
    day48_lookup.rename(columns={'demand': 'demand_last_day'}, inplace=True)
    
    # Merge back to full_df
    full_df = pd.merge(full_df, day48_lookup, on=['geohash', 'timestamp'], how='left')
    
    # 4. Day 49 Morning Trend Ratio
    # Compute trend of demand on day 49 compared to day 48 between 00:00 and 02:00
    print("Computing Day 49 morning trend ratio...")
    morning_timestamps = ['0:0', '0:15', '0:30', '0:45', '1:0', '1:15', '1:30', '1:45', '2:0']
    
    df_48_morning = train[(train['day'] == 48) & (train['timestamp'].isin(morning_timestamps))]
    df_49_morning = train[(train['day'] == 49) & (train['timestamp'].isin(morning_timestamps))]
    
    mean_48 = df_48_morning.groupby('geohash')['demand'].mean()
    mean_49 = df_49_morning.groupby('geohash')['demand'].mean()
    
    trend_ratio = mean_49 / (mean_48 + 1e-5)
    trend_ratio = trend_ratio.rename('day49_trend_ratio')
    
    # Map trend ratio back to full_df
    full_df = pd.merge(full_df, trend_ratio, on='geohash', how='left')
    
    # For Day 48 rows, set trend ratio to 1.0 (since there is no comparison)
    # For day 49 rows without prior morning data, fill NaN with 1.0 (no change)
    full_df.loc[full_df['day'] == 48, 'day49_trend_ratio'] = 1.0
    full_df['day49_trend_ratio'] = full_df['day49_trend_ratio'].fillna(1.0)
    
    # Clip extreme ratios to prevent huge outlier swings
    full_df['day49_trend_ratio'] = full_df['day49_trend_ratio'].clip(0.1, 5.0)
    
    # 5. Missing Value Imputation
    print("Imputing missing values...")
    full_df['RoadType'] = full_df['RoadType'].fillna('Missing')
    full_df['Weather'] = full_df['Weather'].fillna('Missing')
    
    # Median temperature per Weather
    weather_temp_medians = full_df.groupby('Weather')['Temperature'].transform('median')
    full_df['Temperature'] = full_df['Temperature'].fillna(weather_temp_medians)
    full_df['Temperature'] = full_df['Temperature'].fillna(full_df['Temperature'].median())
    
    # 6. Categorical Encoding
    print("Encoding categorical columns...")
    full_df['LargeVehicles'] = full_df['LargeVehicles'].map({'Allowed': 1, 'Not Allowed': 0}).fillna(0).astype(int)
    full_df['Landmarks'] = full_df['Landmarks'].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)
    
    # Label encode categorical columns for LightGBM
    for col in ['RoadType', 'Weather', 'geohash']:
        full_df[col] = full_df[col].astype('category')
        
    # Split back into train and test
    train_preprocessed = full_df[full_df['is_test'] == 0].drop(columns=['is_test', 'Index']).reset_index(drop=True)
    test_preprocessed = full_df[full_df['is_test'] == 1].drop(columns=['is_test', 'demand']).reset_index(drop=True)
    
    # Define features to use
    features = [
        'latitude', 'longitude', 'minutes', 'hour',
        'sin_time', 'cos_time', 'demand_last_day', 'day49_trend_ratio',
        'RoadType', 'NumberofLanes', 'LargeVehicles', 'Landmarks',
        'Temperature', 'Weather', 'geohash'
    ]
    
    target = 'demand'
    
    print(f"Preprocessing completed. Train shape: {train_preprocessed.shape}, Test shape: {test_preprocessed.shape}")
    return train_preprocessed, test_preprocessed, features, target

if __name__ == '__main__':
    train_df, test_df, features, target = load_and_preprocess_data()
    print("Features list:", features)
    print("Sample processed train row:\n", train_df[features].head(1))
