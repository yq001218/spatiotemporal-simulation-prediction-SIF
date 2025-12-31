import torch
import torch.nn as nn

class LSTMTransformer(nn.Module):
    """
    Hybrid LSTM–Transformer model
    """

    def __init__(self,
                 input_dim,
                 hidden_dim=64,
                 nhead=4,
                 num_layers=2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            batch_first=True
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_out = self.transformer(lstm_out)
        return self.fc(attn_out[:, -1, :])
