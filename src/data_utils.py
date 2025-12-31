import numpy as np
import pandas as pd

def load_csv(path, target_col):
    """
    Load multivariate time series from CSV file.

    Parameters
    ----------
    path : str
        Path to csv file
    target_col : str
        Name of target variable (e.g., 'SIF')

    Returns
    -------
    X : ndarray (N, P)
    y : ndarray (N,)
    """
    df = pd.read_csv(path)
    y = df[target_col].values
    X = df.drop(columns=[target_col]).values
    return X, y


def build_sequences(X, y, time_step):
    """
    Convert time series into supervised learning samples.

    X: (N, P)
    y: (N,)
    → X_seq: (N-T, T, P), y_seq: (N-T,)
    """
    X_seq, y_seq = [], []
    for i in range(len(X) - time_step):
        X_seq.append(X[i:i + time_step])
        y_seq.append(y[i + time_step])
    return np.array(X_seq), np.array(y_seq)


def train_test_split(X, y, ratio=0.8):
    """
    Chronological train-test split
    """
    n_train = int(len(X) * ratio)
    return X[:n_train], X[n_train:], y[:n_train], y[n_train:]

