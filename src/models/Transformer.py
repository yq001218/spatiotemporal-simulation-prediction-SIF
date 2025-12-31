import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    """
    Transformer encoder for time series regression
    """

    def __init__(self,
                 input_dim,
                 d_model=64,
                 nhead=4,
                 num_layers=2):
        super().__init__()

        self.embedding = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x)
        return self.fc(x[:, -1, :])
