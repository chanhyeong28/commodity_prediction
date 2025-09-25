import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def SMAPE(pred, true):
    """
    Symmetric Mean Absolute Percentage Error (SMAPE).
    Used for short-term forecasting tasks as per TimeLlaMA paper.
    """
    return np.mean(200 * np.abs(pred - true) / (np.abs(pred) + np.abs(true) + 1e-8))


def MASE(pred, true, insample, freq=1):
    """
    Mean Absolute Scaled Error (MASE).
    Used for short-term forecasting tasks as per TimeLlaMA paper.
    """
    # Calculate naive forecast error (seasonal naive)
    naive_errors = np.mean(np.abs(insample[freq:] - insample[:-freq]))
    if naive_errors == 0:
        return np.mean(np.abs(pred - true))
    
    mae = np.mean(np.abs(pred - true))
    return mae / naive_errors


def OWA(pred, true, naive_pred=None, naive_true=None):
    """
    Overall Weighted Average (OWA).
    Used for short-term forecasting tasks as per TimeLlaMA paper.
    OWA = 0.5 * (SMAPE / SMAPE_naive + MASE / MASE_naive)
    """
    if naive_pred is None or naive_true is None:
        # If no naive forecast provided, return SMAPE as fallback
        return SMAPE(pred, true)
    
    # Calculate SMAPE for our model and naive forecast
    smape_model = SMAPE(pred, true)
    smape_naive = SMAPE(naive_pred, naive_true)
    
    # Calculate MASE for our model and naive forecast
    mase_model = np.mean(np.abs(pred - true))
    mase_naive = np.mean(np.abs(naive_pred - naive_true))
    
    # Avoid division by zero
    if smape_naive == 0 or mase_naive == 0:
        return smape_model
    
    # Calculate OWA
    owa = 0.5 * (smape_model / smape_naive + mase_model / mase_naive)
    return owa


def metric(pred, true):
    """
    Standard metric function for long-term forecasting.
    Returns MAE, MSE, RMSE, MAPE, MSPE as per TimeLlaMA paper.
    """
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe


def short_term_metrics(pred, true, insample=None, freq=1, naive_pred=None, naive_true=None):
    """
    Metrics for short-term forecasting tasks as per TimeLlaMA paper.
    Returns SMAPE, MASE, OWA.
    """
    smape = SMAPE(pred, true)
    
    if insample is not None:
        mase = MASE(pred, true, insample, freq)
    else:
        mase = np.mean(np.abs(pred - true))  # Fallback to MAE
    
    owa = OWA(pred, true, naive_pred, naive_true)
    
    return smape, mase, owa
