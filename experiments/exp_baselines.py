"""
Baseline model experiments:
- Random Forest
- LSTM
- Transformer
"""

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data_utils import load_csv, build_sequences, train_test_split
from src.metrics import evaluate
from src.models.rf import RFModel
from src.models.lstm import LSTMModel
from src.models.transformer import TransformerModel
from src.train import train

# ======================
# Load and prepare data
# ======================
X, y = load_csv("data/raw/sif_drivers.csv", target_col="SIF")
X_seq, y_seq = build_sequences(X, y, time_step=3)
X_tr, X_te, y_tr, y_te = train_test_split(X_seq, y_seq)

# ======================
# Random Forest
# ======================
rf = RFModel()
rf.fit(X_tr, y_tr)
rf_pred = rf.predict(X_te)
print("RF:", evaluate(y_te, rf_pred))

# ======================
# LSTM
# ======================
train_loader = DataLoader(
    TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32)
    ),
    batch_size=32,
    shuffle=True
)

lstm = LSTMModel(input_dim=X_tr.shape[-1])
optimizer = torch.optim.Adam(lstm.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

train(lstm, train_loader, optimizer, loss_fn)

with torch.no_grad():
    lstm_pred = lstm(torch.tensor(X_te, dtype=torch.float32)).numpy().ravel()

print("LSTM:", evaluate(y_te, lstm_pred))

# ======================
# Transformer
# ======================
transformer = TransformerModel(input_dim=X_tr.shape[-1])
optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3)

train(transformer, train_loader, optimizer, loss_fn)

with torch.no_grad():
    trans_pred = transformer(
        torch.tensor(X_te, dtype=torch.float32)
    ).numpy().ravel()

print("Transformer:", evaluate(y_te, trans_pred))

