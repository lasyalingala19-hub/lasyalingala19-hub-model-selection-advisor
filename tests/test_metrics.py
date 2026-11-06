import pandas as pd

from src.recommend import recommend


def test_recommendation_returns_best_model_first():
    results = pd.DataFrame([
        {"model": "slow", "pr_auc": 0.90, "recall": 0.90, "precision": 0.90, "latency_ms": 100, "model_size_mb": 10},
        {"model": "fast", "pr_auc": 0.80, "recall": 0.80, "precision": 0.80, "latency_ms": 2, "model_size_mb": 1},
    ])
    ranked = recommend(results)
    assert "recommendation_score" in ranked.columns
    assert ranked.iloc[0]["model"] == "slow"
