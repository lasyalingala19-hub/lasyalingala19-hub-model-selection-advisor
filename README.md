
# AI Model Selection and Experiment Advisor

An end-to-end machine-learning project that trains several models, tracks experiments, compares performance and operational trade-offs, and recommends the most suitable model for a stated business objective.

## Project objective

Model selection should not depend on accuracy alone. This project compares models using predictive performance, imbalanced-class metrics, latency, model size, fairness metrics when group data is available, and an explainable weighted recommendation score.

## Included models

- Logistic Regression baseline
- Random Forest
- HistGradientBoosting
- XGBoost, when installed

If `data/creditcard.csv` is not present, the project creates a small synthetic imbalanced classification dataset so the pipeline can be tested immediately.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.train
streamlit run src/app.py
```

The training command writes results to `artifacts/model_comparison.csv` and logs runs to MLflow.

Open MLflow in a second terminal:

```bash
mlflow ui --backend-store-uri ./mlruns
```

Then visit `http://127.0.0.1:5000`.

## Use your fraud dataset

Place your file at `data/creditcard.csv`, then run:

```bash
python -m src.train --data data/creditcard.csv --target Class
```

## Project structure

```text
src/
  app.py                 Streamlit dashboard
  train.py               Training and experiment pipeline
  data.py                Dataset loading and preprocessing
  evaluate.py            Metrics, latency, and model-size measurement
  recommend.py           Weighted model recommendation
tests/
  test_metrics.py        Unit tests
```

## Recommendation logic

The default score prioritizes recall and PR-AUC for imbalanced fraud detection:

```text
40% PR-AUC | 30% recall | 15% precision | 10% latency | 5% model size
```

The report shows each component so the user can change the priorities for a different business problem.

## Next improvements

- Add MLflow Model Registry
- Add SHAP error analysis
- Add Fairlearn group metrics
- Add cost estimates for AWS deployment
- Add an LLM explanation layer using structured experiment results
- Add FastAPI endpoints
- Add GitHub Actions CI/CD
- Add Docker Compose with MLflow and PostgreSQL

## GitHub commands

```bash
git init
git add .
git commit -m "Add AI model selection advisor"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/model-selection-advisor.git
git push -u origin main
```

Do not commit datasets, secrets, virtual environments, or API keys.
