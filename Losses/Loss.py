import numpy as np

def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

def mse_drv(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)

def mae(y_true, y_pred):
    return np.mean(y_true - y_pred)

def mae_drv(y_true, y_pred):
    if y_true > y_pred:
        return -1
    elif y_true < y_pred:
        return 1

def huber(y_true, y_pred, threshold):
    error = y_pred - y_true
    is_small_error = np.abs(error) < threshold
    mae_loss = threshold * (np.abs(error) - 0.5 * threshold)
    mse_loss = np.square(error) / 2
    return np.mean(np.where(is_small_error, mse_loss, mae_loss))