import torch
import torch.nn as nn

class KANLayer(nn.Module):
    """
    Kolmogorov–Arnold Network (simplified)
    """

    def __init__(self, input_dim, hidden_dim=16):
        super().__init__()
        self.subnets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1)
            ) for _ in range(input_dim)
        ])

    def forward(self, x):
        out = 0
        for i, net in enumerate(self.subnets):
            out += net(x[:, i:i+1])
        return out
