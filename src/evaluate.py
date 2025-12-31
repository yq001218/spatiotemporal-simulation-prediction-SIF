
from src.metrics import rmse, mae, r2, psi

def evaluate(y_true, y_pred):
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "R2": r2(y_true, y_pred),
        "PSI": psi(y_true, y_pred)
    }
