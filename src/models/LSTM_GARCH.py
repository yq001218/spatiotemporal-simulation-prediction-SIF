import torch
from src.models.lstm import LSTMModel
from src.models.garch import GARCHModel

class LSTM_GARCH:
    """
    LSTM + GARCH hybrid model
    Trend (LSTM) + volatility (GARCH)
    """

    def __init__(self, input_dim):
        self.lstm = LSTMModel(input_dim)
        self.garch = GARCHModel()

    def fit(self, X, y):
        with torch.no_grad():
            trend = self.lstm(
                torch.tensor(X, dtype=torch.float32)
            ).numpy().ravel()
        residuals = y - trend
        self.garch.fit(residuals)

    def predict(self, X):
        with torch.no_grad():
            trend = self.lstm(
                torch.tensor(X, dtype=torch.float32)
            ).numpy().ravel()
        return trend + self.garch.predict()
