from pathlib import Path
import subprocess

import pandas as pd
import streamlit as st

st.set_page_config(page_title="AI Model Selection Advisor", layout="wide")
st.title("AI Model Selection and Experiment Advisor")
st.write("Compare models using predictive performance and deployment trade-offs.")

uploaded = st.file_uploader("Upload a CSV dataset (optional)", type=["csv"])
target = st.text_input("Target column", value="Class")

if st.button("Run model comparison"):
    data_path = None
    if uploaded:
        Path("data").mkdir(exist_ok=True)
        data_path = "data/uploaded.csv"
        Path(data_path).write_bytes(uploaded.getvalue())

    command = ["python", "-m", "src.train", "--target", target]
    if data_path:
        command += ["--data", data_path]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        st.error(result.stderr)
    else:
        st.code(result.stdout)
        results_path = Path("artifacts/model_comparison.csv")
        if results_path.exists():
            results = pd.read_csv(results_path)
            st.subheader("Model comparison")
            st.dataframe(results, use_container_width=True)
            st.subheader("Metric comparison")
            st.bar_chart(results.set_index("model")[["pr_auc", "recall", "precision"]])
            st.success(f"Recommended model: {results.iloc[0]['model']}")
