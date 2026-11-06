import os
import tempfile
import time
from pathlib import Path

import joblib
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score


def evaluate_model(model, X_test, y_test):
    start = time.perf_counter()
    probabilities = model.predict_proba(X_test)[:, 1]
    elapsed = time.perf_counter() - start
    predictions = (probabilities >= 0.5).astype(int)

    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as handle:
        temporary_path = handle.name
    joblib.dump(model, temporary_path)
    model_size_mb = os.path.getsize(temporary_path) / (1024 * 1024)
    Path(temporary_path).unlink(missing_ok=True)

    return {
        "roc_auc": roc_auc_score(y_test, probabilities),
        "pr_auc": average_precision_score(y_test, probabilities),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "latency_ms": elapsed * 1000 / max(len(X_test), 1),
        "model_size_mb": model_size_mb,
    }
