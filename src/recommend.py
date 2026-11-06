import pandas as pd


def _normalize(series: pd.Series, higher_is_better: bool = True):
    minimum, maximum = series.min(), series.max()
    if maximum == minimum:
        return pd.Series(1.0, index=series.index)
    values = (series - minimum) / (maximum - minimum)
    return values if higher_is_better else 1 - values


def recommend(results: pd.DataFrame, weights: dict | None = None):
    """Rank models using transparent, configurable weights."""
    weights = weights or {"pr_auc": 0.40, "recall": 0.30, "precision": 0.15, "latency_ms": 0.10, "model_size_mb": 0.05}
    ranked = results.copy()
    score = 0.0
    for metric, weight in weights.items():
        higher = metric not in {"latency_ms", "model_size_mb"}
        score += _normalize(ranked[metric], higher) * weight
    ranked["recommendation_score"] = score
    return ranked.sort_values("recommendation_score", ascending=False).reset_index(drop=True)
