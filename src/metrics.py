
# import numpy as np
# from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# def rmse(y_true, y_pred):
#     return np.sqrt(mean_squared_error(y_true, y_pred))


# def mae(y_true, y_pred):
#     return mean_absolute_error(y_true, y_pred)


# def r2(y_true, y_pred):
#     return r2_score(y_true, y_pred)


# def psi(y_true, y_pred):
#     """
#     Prediction Stability Index (PSI)
#     Lower values indicate more stable predictions.
#     """
#     diff = np.diff(y_pred - y_true)
#     return np.std(diff)

import numpy as np
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)


def r2(y_true, y_pred):
    return r2_score(y_true, y_pred)


def psi(y_true, y_pred):
    """
    Prediction Stability Index (PSI)

    Measures temporal smoothness of prediction errors.
    """
    error = y_pred - y_true
    return np.std(np.diff(error))
