import torch
from torch.utils.data import DataLoader, TensorDataset
from src.data_utils import load_csv, build_sequences, train_test_split
from src.models.lstm_transformer import LSTMTransformer
from src.train import train

# Load data
X, y = load_csv("data/raw/sif_drivers.csv", target_col="SIF")

# Build sequences
X_seq, y_seq = build_sequences(X, y, time_step=3)

# Split
X_tr, X_te, y_tr, y_te = train_test_split(X_seq, y_seq)

# Torch dataset
train_loader = DataLoader(
    TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32)
    ),
    batch_size=32,
    shuffle=True
)

# Model
model = LSTMTransformer(input_dim=X_tr.shape[-1])
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

train(model, train_loader, optimizer, loss_fn)


