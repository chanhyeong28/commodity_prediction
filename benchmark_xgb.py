import os
import warnings
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor


warnings.filterwarnings("ignore")


NUM_TARGET_COLUMNS = 424


def load_train_labels(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    assert 'date_id' in df.columns, "train_labels.csv must contain 'date_id'"
    return df.sort_values('date_id').reset_index(drop=True)


def train_univariate_xgb_models(labels: pd.DataFrame, output_path: str) -> Dict[str, XGBRegressor]:
    target_cols = [c for c in labels.columns if c.startswith('target_')]

    models: Dict[str, XGBRegressor] = {}

    for col in target_cols:
        series = labels[['date_id', col]].copy()
        # Create lag features using the same information available at inference
        for lag in [1, 2, 3, 4]:
            series[f'{col}_lag{lag}'] = series[col].shift(lag)

        # Drop rows with NaNs in lag features
        feature_cols = [f'{col}_lag{lag}' for lag in [1, 2, 3, 4]]
        series = series.dropna(subset=feature_cols)

        if series.empty:
            # Fallback: if not enough rows, skip training and use a dummy constant model later
            continue

        # Time-based split (by date_id) to avoid leakage
        cutoff = series['date_id'].quantile(0.8)
        train_mask = series['date_id'] <= cutoff

        X_train = series.loc[train_mask, feature_cols]
        y_train = series.loc[train_mask, col]
        X_val = series.loc[~train_mask, feature_cols]
        y_val = series.loc[~train_mask, col]

        # Simple, fast baseline config
        model = XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            n_jobs=-1,
            tree_method='hist',
            random_state=42,
        )

        eval_set = [(X_val, y_val)] if len(X_val) else None
        model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            eval_metric='rmse',
            verbose=False,
        )

        models[col] = model

    # Persist models
    joblib.dump(models, output_path)
    return models


def main():
    data_dir = './raw_data'
    labels_path = os.path.join(data_dir, 'train_labels.csv')
    models_path = 'xgb_univariate_models.pkl'

    labels = load_train_labels(labels_path)

    # Ensure expected number of targets when possible
    present_targets = [c for c in labels.columns if c.startswith('target_')]
    if len(present_targets) != NUM_TARGET_COLUMNS:
        warnings.warn(f'Expected {NUM_TARGET_COLUMNS} targets, found {len(present_targets)}')

    train_univariate_xgb_models(labels, models_path)
    print(f'Trained XGB models saved to {models_path}')


if __name__ == '__main__':
    main()