import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    d += 1e-12
    return 0.01*(u / d).mean(-1)


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

def Relat_RMSE(pred, true,no_pred):
    return RMSE(pred, true) / RMSE(no_pred, true)


def metric(pred, true, no_pred):
    mae = float(MAE(pred, true))  
    mse = float(MSE(pred, true))
    rmse = float(RMSE(pred, true))
    mape = float(MAPE(pred, true))
    mspe = float(MSPE(pred, true))
    rse = float(RSE(pred, true))
    corr = float(CORR(pred, true))
    relative_rmse = float(Relat_RMSE(pred, true, no_pred))
    return mae, mse, rmse, mape, mspe, rse, corr, relative_rmse