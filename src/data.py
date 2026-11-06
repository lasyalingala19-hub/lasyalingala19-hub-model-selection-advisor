from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


def load_dataset(path: str | None = None, target: str = "Class"):
    """Load a local dataset or create a reproducible demo dataset."""
    if path and Path(path).exists():
        frame = pd.read_csv(path)
        if target not in frame.columns:
            raise ValueError(f"Target column '{target}' was not found.")
        return frame, target

    X, y = make_classification(
        n_samples=5000, n_features=12, n_informative=8, n_redundant=2,
        weights=[0.96, 0.04], random_state=42,
    )
    columns = [f"feature_{i}" for i in range(X.shape[1])]
    frame = pd.DataFrame(X, columns=columns)
    frame[target] = y
    return frame, target


def split_features_target(frame: pd.DataFrame, target: str):
    X = frame.drop(columns=[target]).copy()
    y = frame[target].copy()
    X = pd.get_dummies(X, drop_first=False)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    return X, y
