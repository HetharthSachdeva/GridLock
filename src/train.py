import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import pickle

# Import our preprocess function
from preprocess import load_and_preprocess_data

def train_model():
    # Load and preprocess data
    train_df, test_df, features, target = load_and_preprocess_data()
    
    print("\n--- Starting Training ---")
    print(f"Features used ({len(features)}): {features}")
    
    # Initialize variables for Out-Of-Fold (OOF) predictions
    oof_predictions = np.zeros(len(train_df))
    test_predictions = np.zeros(len(test_df))
    
    # 5-Fold Cross Validation
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    models = []
    oof_rmses = []
    oof_maes = []
    
    os.makedirs("models", exist_ok=True)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
        print(f"\n--- Fold {fold + 1} / {n_splits} ---")
        
        X_train, y_train = train_df.loc[train_idx, features], train_df.loc[train_idx, target]
        X_val, y_val = train_df.loc[val_idx, features], train_df.loc[val_idx, target]
        
        # Define LightGBM dataset
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Hyperparameters for LightGBM Regressor
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'n_estimators': 5000,
            'learning_rate': 0.02,
            'num_leaves': 50,
            'max_depth': 6,
            'min_child_samples': 30,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'reg_alpha': 0.5,
            'reg_lambda': 1.0,
            'random_state': 42 + fold,
            'verbose': -1,
            'n_jobs': -1
        }
        
        # Train model with early stopping
        model = lgb.train(
            params,
            train_data,
            valid_sets=[train_data, val_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=200, verbose=False),
                lgb.log_evaluation(period=200)
            ]
        )
        
        # Predict on validation fold
        val_preds = model.predict(X_val, num_iteration=model.best_iteration)
        oof_predictions[val_idx] = val_preds
        
        # Evaluate fold performance
        fold_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
        fold_mae = mean_absolute_error(y_val, val_preds)
        oof_rmses.append(fold_rmse)
        oof_maes.append(fold_mae)
        
        print(f"Fold {fold + 1} RMSE: {fold_rmse:.6f} | MAE: {fold_mae:.6f}")
        
        # Predict on test set (average predictions across all folds)
        test_preds = model.predict(test_df[features], num_iteration=model.best_iteration)
        test_predictions += test_preds / n_splits
        
        # Save model
        model_path = f"models/lgb_fold_{fold + 1}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
            
    # Final Out-Of-Fold Evaluation
    overall_rmse = np.sqrt(mean_squared_error(train_df[target], oof_predictions))
    overall_mae = mean_absolute_error(train_df[target], oof_predictions)
    overall_r2 = r2_score(train_df[target], oof_predictions)
    
    print("\n--- Out-Of-Fold Summary ---")
    print(f"Mean Fold RMSE: {np.mean(oof_rmses):.6f} ± {np.std(oof_rmses):.6f}")
    print(f"Overall OOF RMSE: {overall_rmse:.6f}")
    print(f"Overall OOF MAE: {overall_mae:.6f}")
    print(f"Overall OOF R2 Score: {overall_r2:.6f}")
    
    # Save the test predictions for quick submission generation
    print("\nSaving baseline predictions...")
    test_df_out = test_df.copy()
    test_df_out['demand'] = test_predictions
    
    # Clip demand to be non-negative (since traffic demand cannot be less than 0)
    test_df_out['demand'] = test_df_out['demand'].clip(0.0, 1.0)
    
    os.makedirs("predictions", exist_ok=True)
    test_df_out.to_csv("predictions/test_predictions.csv", index=False)
    
    # Generate submission file
    submission = test_df_out[['Index', 'demand']]
    submission.to_csv("submission.csv", index=False)
    print("Baseline submission.csv generated successfully!")
    
    return oof_predictions, test_predictions

if __name__ == '__main__':
    train_model()
