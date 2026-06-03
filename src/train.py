import os
import pickle

import lightgbm as lgb
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

from preprocess import load_and_preprocess_data


def train_model():
    train_df, test_df, features, target = load_and_preprocess_data()

    categorical_features = ['RoadType', 'Weather', 'geohash']

    print("\n--- Starting Training ---")
    print(f"Features used ({len(features)}): {features}")

    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_preds_lgb = np.zeros(len(train_df))
    oof_preds_cat = np.zeros(len(train_df))
    oof_predictions = np.zeros(len(train_df))
    test_predictions = np.zeros(len(test_df))

    lgb_rmses = []
    cat_rmses = []
    ensemble_rmses = []
    lgb_maes = []
    cat_maes = []
    ensemble_maes = []

    os.makedirs("models", exist_ok=True)

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
        print(f"\n--- Fold {fold + 1} / {n_splits} ---")

        X_train = train_df.loc[train_idx, features]
        y_train = train_df.loc[train_idx, target]
        X_val = train_df.loc[val_idx, features]
        y_val = train_df.loc[val_idx, target]

        # LightGBM training
        lgb_train = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_features)
        lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train, categorical_feature=categorical_features)

        lgb_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
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

        lgb_model = lgb.train(
            lgb_params,
            lgb_train,
            valid_sets=[lgb_train, lgb_val],
            callbacks=[
                lgb.early_stopping(stopping_rounds=200, verbose=False),
                lgb.log_evaluation(period=200)
            ],
            num_boost_round=3500,
        )

        val_preds_lgb = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
        test_preds_lgb = lgb_model.predict(test_df[features], num_iteration=lgb_model.best_iteration)

        # CatBoost training
        cat_train = Pool(X_train, y_train, cat_features=categorical_features)
        cat_val = Pool(X_val, y_val, cat_features=categorical_features)

        cat_model = CatBoostRegressor(
            iterations=3500,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=3,
            border_count=128,
            bagging_temperature=0.4,
            random_seed=42 + fold,
            loss_function='RMSE',
            eval_metric='RMSE',
            early_stopping_rounds=200,
            verbose=200,
            thread_count=-1,
        )

        cat_model.fit(cat_train, eval_set=cat_val, use_best_model=True)

        val_preds_cat = cat_model.predict(X_val)
        test_preds_cat = cat_model.predict(test_df[features])

        # Ensemble average of CatBoost and LightGBM
        val_preds_ensemble = 0.5 * val_preds_lgb + 0.5 * val_preds_cat
        test_predictions += (test_preds_lgb + test_preds_cat) / (2 * n_splits)

        oof_preds_lgb[val_idx] = val_preds_lgb
        oof_preds_cat[val_idx] = val_preds_cat
        oof_predictions[val_idx] = val_preds_ensemble

        lgb_rmse = np.sqrt(mean_squared_error(y_val, val_preds_lgb))
        cat_rmse = np.sqrt(mean_squared_error(y_val, val_preds_cat))
        ensemble_rmse = np.sqrt(mean_squared_error(y_val, val_preds_ensemble))

        lgb_mae = mean_absolute_error(y_val, val_preds_lgb)
        cat_mae = mean_absolute_error(y_val, val_preds_cat)
        ensemble_mae = mean_absolute_error(y_val, val_preds_ensemble)

        lgb_rmses.append(lgb_rmse)
        cat_rmses.append(cat_rmse)
        ensemble_rmses.append(ensemble_rmse)
        lgb_maes.append(lgb_mae)
        cat_maes.append(cat_mae)
        ensemble_maes.append(ensemble_mae)

        print(f"Fold {fold + 1} LGB RMSE: {lgb_rmse:.6f} | CAT RMSE: {cat_rmse:.6f} | ENS RMSE: {ensemble_rmse:.6f}")
        print(f"Fold {fold + 1} LGB MAE: {lgb_mae:.6f} | CAT MAE: {cat_mae:.6f} | ENS MAE: {ensemble_mae:.6f}")

        lgb_model_path = f"models/lgb_fold_{fold + 1}.pkl"
        cat_model_path = f"models/catboost_fold_{fold + 1}.pkl"

        with open(lgb_model_path, 'wb') as f:
            pickle.dump(lgb_model, f)

        with open(cat_model_path, 'wb') as f:
            pickle.dump(cat_model, f)

    overall_rmse_lgb = np.sqrt(mean_squared_error(train_df[target], oof_preds_lgb))
    overall_rmse_cat = np.sqrt(mean_squared_error(train_df[target], oof_preds_cat))
    overall_rmse_ens = np.sqrt(mean_squared_error(train_df[target], oof_predictions))
    overall_mae_lgb = mean_absolute_error(train_df[target], oof_preds_lgb)
    overall_mae_cat = mean_absolute_error(train_df[target], oof_preds_cat)
    overall_mae_ens = mean_absolute_error(train_df[target], oof_predictions)
    overall_r2 = r2_score(train_df[target], oof_predictions)

    print("\n--- Out-Of-Fold Summary ---")
    print(f"LightGBM RMSE: {np.mean(lgb_rmses):.6f} ± {np.std(lgb_rmses):.6f}")
    print(f"CatBoost RMSE: {np.mean(cat_rmses):.6f} ± {np.std(cat_rmses):.6f}")
    print(f"Ensemble RMSE: {np.mean(ensemble_rmses):.6f} ± {np.std(ensemble_rmses):.6f}")
    print(f"Overall OOF RMSE: {overall_rmse_ens:.6f}")
    print(f"Overall OOF MAE: {overall_mae_ens:.6f}")
    print(f"Overall OOF R2 Score: {overall_r2:.6f}")

    print("\nSaving ensemble predictions...")
    test_df_out = test_df.copy()
    test_df_out['demand'] = test_predictions.clip(0.0, 1.0)

    os.makedirs("predictions", exist_ok=True)
    test_df_out.to_csv("predictions/test_predictions.csv", index=False)

    submission = test_df_out[['Index', 'demand']]
    submission.to_csv("submission.csv", index=False)
    print("Submission file generated successfully!")

    return oof_predictions, test_predictions


if __name__ == '__main__':
    train_model()
