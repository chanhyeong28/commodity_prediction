import os
import warnings

import joblib
import pandas as pd
import polars as pl

import kaggle_evaluation.mitsui_inference_server


NUM_TARGET_COLUMNS = 424
MODELS_PATH = 'xgb_univariate_models.pkl'


def _ensure_columns_order(df: pd.DataFrame | pl.DataFrame) -> pd.DataFrame | pl.DataFrame:
    ordered = [f'target_{i}' for i in range(NUM_TARGET_COLUMNS)]
    if isinstance(df, pl.DataFrame):
        present = [c for c in ordered if c in df.columns]
        return df.select(present)
    else:
        present = [c for c in ordered if c in df.columns]
        return df[present]


def predict(
    test: pl.DataFrame,
    label_lags_1_batch: pl.DataFrame,
    label_lags_2_batch: pl.DataFrame,
    label_lags_3_batch: pl.DataFrame,
    label_lags_4_batch: pl.DataFrame,
) -> pl.DataFrame | pd.DataFrame:
    """Local gateway-compatible predict using univariate XGB models.
    Builds per-target lag features from provided lag batches and infers one row.
    """
    # Load models once (module-level cache fine in Kaggle runtime)
    try:
        models = joblib.load(MODELS_PATH)
    except Exception as e:
        warnings.warn(f'Failed to load models from {MODELS_PATH}: {e}. Returning zeros.')
        return pl.DataFrame({f'target_{i}': 0.0 for i in range(NUM_TARGET_COLUMNS)})

    # Build feature row per target using lag batches; prefer later lag if duplicates
    def extract_value(df: pl.DataFrame, col: str) -> float | None:
        if df.is_empty() or col not in df.columns:
            return None
        # Take the last available row (latest release)
        return df.select(pl.col(col)).to_series().drop_nulls().tail(1).to_list()[0] if col in df.columns else None

    preds = {}
    for i in range(NUM_TARGET_COLUMNS):
        col = f'target_{i}'
        model = models.get(col)
        if model is None:
            preds[col] = 0.0
            continue

        l1 = extract_value(label_lags_1_batch, col)
        l2 = extract_value(label_lags_2_batch, col)
        l3 = extract_value(label_lags_3_batch, col)
        l4 = extract_value(label_lags_4_batch, col)

        features = pd.DataFrame({
            f'{col}_lag1': [l1],
            f'{col}_lag2': [l2],
            f'{col}_lag3': [l3],
            f'{col}_lag4': [l4],
        })
        features = features.fillna(0.0)
        try:
            preds[col] = float(model.predict(features)[0])
        except Exception:
            preds[col] = 0.0

    predictions = pl.DataFrame(preds)
    predictions = _ensure_columns_order(predictions)

    assert isinstance(predictions, (pd.DataFrame, pl.DataFrame))
    assert len(predictions) == 1
    return predictions


# When your notebook is run on the hidden test set, inference_server.serve must be called within 15 minutes of the notebook starting
# or the gateway will throw an error. If you need more than 15 minutes to load your model you can do so during the very
# first `predict` call, which does not have the usual 1 minute response deadline.
inference_server = kaggle_evaluation.mitsui_inference_server.MitsuiInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(('/kaggle/input/mitsui-commodity-prediction-challenge/',))