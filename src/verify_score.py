import pickle
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from preprocess import load_and_preprocess_data


def calculate_r2():
    train_df, _, features, target = load_and_preprocess_data()

    oof_preds_lgb = np.zeros(len(train_df))
    oof_preds_cat = np.zeros(len(train_df))
    oof_predictions = np.zeros(len(train_df))

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (_, val_idx) in enumerate(kf.split(train_df)):
        lgb_path = f"models/lgb_fold_{fold + 1}.pkl"
        cat_path = f"models/catboost_fold_{fold + 1}.pkl"

        with open(lgb_path, 'rb') as f:
            lgb_model = pickle.load(f)

        with open(cat_path, 'rb') as f:
            cat_model = pickle.load(f)

        X_val = train_df.loc[val_idx, features]
        val_preds_lgb = lgb_model.predict(X_val)
        val_preds_cat = cat_model.predict(X_val)

        oof_preds_lgb[val_idx] = val_preds_lgb
        oof_preds_cat[val_idx] = val_preds_cat
        oof_predictions[val_idx] = 0.5 * val_preds_lgb + 0.5 * val_preds_cat

    actual = train_df[target]
    r2 = r2_score(actual, oof_predictions)
    score = max(0, 100 * r2)

    print("--- Official Metric Evaluation ---")
    print(f"Out-of-Fold R2 Score: {r2:.6f}")
    print(f"Calculated Hackathon Score: {score:.4f} / 100.0000")


if __name__ == '__main__':
    calculate_r2()
