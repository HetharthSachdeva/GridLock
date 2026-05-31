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
    
    # Additional temporal features
    full_df['is_peak_hour'] = ((full_df['hour'] >= 7) & (full_df['hour'] <= 10)) | ((full_df['hour'] >= 17) & (full_df['hour'] <= 19))
    full_df['is_night'] = (full_df['hour'] < 6) | (full_df['hour'] >= 22)
    full_df['is_morning'] = (full_df['hour'] >= 6) & (full_df['hour'] < 12)
    full_df['is_afternoon'] = (full_df['hour'] >= 12) & (full_df['hour'] < 17)
    full_df['is_evening'] = (full_df['hour'] >= 17) & (full_df['hour'] < 22)
    
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
    
    # Also compute a geohash-level historical mean for fallback imputation
    geohash_day48_mean = df_48.groupby('geohash')['demand'].mean().reset_index()
    geohash_day48_mean.rename(columns={'demand': 'geohash_day48_mean'}, inplace=True)
    
    # Compute timestamp-level statistics (temporal pattern across all geohashes)
    timestamp_day48_mean = df_48.groupby('timestamp')['demand'].mean().reset_index()
    timestamp_day48_mean.rename(columns={'demand': 'timestamp_day48_mean'}, inplace=True)
    timestamp_day48_std = df_48.groupby('timestamp')['demand'].std().reset_index()
    timestamp_day48_std.rename(columns={'demand': 'timestamp_day48_std'}, inplace=True)
    
    # Compute road-type level patterns
    roadtype_day48_mean = df_48.groupby('RoadType')['demand'].mean().reset_index()
    roadtype_day48_mean.rename(columns={'demand': 'roadtype_day48_mean'}, inplace=True)
    
    # Merge back to full_df
    full_df = pd.merge(full_df, day48_lookup, on=['geohash', 'timestamp'], how='left')
    full_df = pd.merge(full_df, geohash_day48_mean, on='geohash', how='left')
    full_df = pd.merge(full_df, timestamp_day48_mean, on='timestamp', how='left')
    full_df = pd.merge(full_df, timestamp_day48_std, on='timestamp', how='left')
    full_df = pd.merge(full_df, roadtype_day48_mean, on='RoadType', how='left')
    
    # Fill missing day-48 lookups with the geohash's average demand from Day 48, then with global mean
    global_day48_mean = df_48['demand'].mean()
    full_df['demand_last_day'] = full_df['demand_last_day'].fillna(full_df['geohash_day48_mean'])
    full_df['demand_last_day'] = full_df['demand_last_day'].fillna(full_df['timestamp_day48_mean'])
    full_df['demand_last_day'] = full_df['demand_last_day'].fillna(global_day48_mean)
    
    # Fill timestamp stats with global if missing
    global_timestamp_std = df_48['demand'].std()
    full_df['timestamp_day48_std'] = full_df['timestamp_day48_std'].fillna(global_timestamp_std)
    full_df['roadtype_day48_mean'] = full_df['roadtype_day48_mean'].fillna(global_day48_mean)
    
    # 4. Day 49 Morning Trend Ratio
    # Compute trend of demand on day 49 compared to day 48 between 00:00 and 02:00
    print("Computing Day 49 morning trend ratio...")
    morning_timestamps = ['0:0', '0:15', '0:30', '0:45', '1:0', '1:15', '1:30', '1:45', '2:0']
    early_morning_timestamps = ['2:15', '2:30', '2:45', '3:0', '3:15', '3:30', '3:45', '4:0']
    
    df_48_morning = train[(train['day'] == 48) & (train['timestamp'].isin(morning_timestamps))].copy()
    df_49_morning = train[(train['day'] == 49) & (train['timestamp'].isin(morning_timestamps))].copy()
    df_48_early = train[(train['day'] == 48) & (train['timestamp'].isin(early_morning_timestamps))].copy()
    df_49_early = train[(train['day'] == 49) & (train['timestamp'].isin(early_morning_timestamps))].copy()
    
    df_48_morning['RoadType'] = df_48_morning['RoadType'].fillna('Missing')
    df_49_morning['RoadType'] = df_49_morning['RoadType'].fillna('Missing')
    df_48_early['RoadType'] = df_48_early['RoadType'].fillna('Missing')
    df_49_early['RoadType'] = df_49_early['RoadType'].fillna('Missing')
    
    mean_48 = df_48_morning.groupby('geohash')['demand'].mean()
    mean_49 = df_49_morning.groupby('geohash')['demand'].mean()
    
    trend_ratio = mean_49 / (mean_48 + 1e-5)
    trend_ratio = trend_ratio.rename('day49_trend_ratio')
    
    # Additional early morning trend for better test set alignment
    mean_48_early = df_48_early.groupby('geohash')['demand'].mean()
    mean_49_early = df_49_early.groupby('geohash')['demand'].mean()
    early_trend_ratio = mean_49_early / (mean_48_early + 1e-5)
    early_trend_ratio = early_trend_ratio.rename('early_morning_trend_ratio')
    
    # Group-level fallback ratios by RoadType
    roadtype_ratio_48 = df_48_morning.groupby('RoadType')['demand'].mean()
    roadtype_ratio_49 = df_49_morning.groupby('RoadType')['demand'].mean()
    roadtype_ratio = (roadtype_ratio_49 / (roadtype_ratio_48 + 1e-5)).rename('roadtype_trend_ratio')
    
    # Global fallback ratio
    global_ratio = df_49_morning['demand'].mean() / (df_48_morning['demand'].mean() + 1e-5)
    
    # Map trend ratio back to full_df
    full_df = pd.merge(full_df, trend_ratio, on='geohash', how='left')
    full_df = pd.merge(full_df, early_trend_ratio, on='geohash', how='left')
    full_df = pd.merge(full_df, roadtype_ratio.reset_index(), on='RoadType', how='left')
    
    # For Day 48 rows, set trend ratio to 1.0 (since there is no comparison)
    full_df.loc[full_df['day'] == 48, 'day49_trend_ratio'] = 1.0
    full_df.loc[full_df['day'] == 48, 'early_morning_trend_ratio'] = 1.0
    full_df['day49_trend_ratio'] = full_df['day49_trend_ratio'].fillna(full_df['roadtype_trend_ratio'])
    full_df['day49_trend_ratio'] = full_df['day49_trend_ratio'].fillna(global_ratio)
    full_df['early_morning_trend_ratio'] = full_df['early_morning_trend_ratio'].fillna(full_df['roadtype_trend_ratio'])
    full_df['early_morning_trend_ratio'] = full_df['early_morning_trend_ratio'].fillna(global_ratio)
    full_df['roadtype_trend_ratio'] = full_df['roadtype_trend_ratio'].fillna(global_ratio)
    full_df['geohash_day48_mean'] = full_df['geohash_day48_mean'].fillna(global_day48_mean)
    
    # Clip extreme ratios to prevent huge outlier swings
    full_df['day49_trend_ratio'] = full_df['day49_trend_ratio'].clip(0.1, 5.0)
    full_df['early_morning_trend_ratio'] = full_df['early_morning_trend_ratio'].clip(0.1, 5.0)
    
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
        'sin_time', 'cos_time', 'is_peak_hour', 'is_night', 'is_morning', 'is_afternoon', 'is_evening',
        'demand_last_day', 'geohash_day48_mean', 'timestamp_day48_mean', 'timestamp_day48_std', 'roadtype_day48_mean',
        'day49_trend_ratio', 'early_morning_trend_ratio', 'roadtype_trend_ratio', 'day',
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
