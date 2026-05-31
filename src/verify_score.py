import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import r2_score
from preprocess import load_and_preprocess_data

def calculate_r2():
    train_df, _, features, target = load_and_preprocess_data()
    
    # Re-generate Out-of-fold predictions
    from sklearn.model_selection import KFold
    oof_predictions = np.zeros(len(train_df))
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
        model_path = f"models/lgb_fold_{fold + 1}.pkl"
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        X_val = train_df.loc[val_idx, features]
        oof_predictions[val_idx] = model.predict(X_val)
        
    actual = train_df[target]
    r2 = r2_score(actual, oof_predictions)
    score = max(0, 100 * r2)
    
    print("--- Official Metric Evaluation ---")
    print(f"Out-of-Fold R2 Score: {r2:.6f}")
    print(f"Calculated Hackathon Score: {score:.4f} / 100.0000")

if __name__ == '__main__':
    calculate_r2()
