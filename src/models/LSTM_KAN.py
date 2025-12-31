import torch
import torch.nn as nn
from src.models.kan import KANLayer

class LSTM_KAN(nn.Module):
    """
    LSTM + KAN model
    Temporal dependency + interpretable nonlinearity
    """

    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            batch_first=True
        )

        self.kan = KANLayer(hidden_dim)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.kan(h[-1])
