import torch
from src.models.lstm import LSTMModel
from src.models.rf import RFModel

class LSTM_RF:
    """
    Hybrid LSTM + Random Forest
    LSTM extracts temporal features
    RF performs nonlinear regression
    """

    def __init__(self, input_dim):
        self.lstm = LSTMModel(input_dim)
        self.rf = RFModel()

    def fit(self, X, y):
        with torch.no_grad():
            features = self.lstm(
                torch.tensor(X, dtype=torch.float32)
            ).numpy()
        self.rf.fit(features, y)

    def predict(self, X):
        with torch.no_grad():
            features = self.lstm(
                torch.tensor(X, dtype=torch.float32)
            ).numpy()
        return self.rf.predict(features)
