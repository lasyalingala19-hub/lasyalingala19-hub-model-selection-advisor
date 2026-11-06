import argparse
from pathlib import Path

import mlflow
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import load_dataset, split_features_target
from .evaluate import evaluate_model
from .recommend import recommend


def build_models():
    models = {
        "logistic_regression": Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=500, class_weight="balanced")),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=150, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=42),
    }
    try:
        from xgboost import XGBClassifier
        models["xgboost"] = XGBClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.08,
            eval_metric="logloss", random_state=42,
        )
    except ImportError:
        print("XGBoost is unavailable; continuing with scikit-learn models.")
    return models


def run(data_path=None, target="Class"):
    frame, target = load_dataset(data_path, target)
    X, y = split_features_target(frame, target)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    mlflow.set_experiment("ai-model-selection-advisor")
    rows = []
    for name, model in build_models().items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            metrics = evaluate_model(model, X_test, y_test)
            mlflow.log_param("model", name)
            mlflow.log_param("feature_count", X.shape[1])
            mlflow.log_metrics(metrics)
            rows.append({"model": name, **metrics})

    results = pd.DataFrame(rows)
    ranked = recommend(results)
    Path("artifacts").mkdir(exist_ok=True)
    ranked.to_csv("artifacts/model_comparison.csv", index=False)
    print(ranked.to_string(index=False))
    print(f"\nRecommended model: {ranked.iloc[0]['model']}")
    return ranked


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None)
    parser.add_argument("--target", default="Class")
    args = parser.parse_args()
    run(args.data, args.target)
