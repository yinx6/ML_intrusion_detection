from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(page_title="IDS-ML Dashboard", layout="wide")
st.title("IDS-ML Demo Dashboard")

metrics_file = Path("artifacts/metrics_summary.csv")

if metrics_file.exists():
    st.subheader("Model comparison")
    df = pd.read_csv(metrics_file, index_col=0)
    # Bug 12 fix: width='stretch' is not a valid Streamlit argument; width
    # must be an integer (pixelspowercfg /setactive d92ed291-5d91-42f7-9f8f-c0bc20645861). use_container_width=True is the correct
    # API for a full-width dataframe.
    st.dataframe(df, use_container_width=True)
    st.bar_chart(df[["cv_f1_macro", "test_f1_macro"]])
else:
    st.info("No training metrics found yet. Run training first.")

st.subheader("Project status")
st.write("- Supervised models: Random Forest, XGBoost, MLP")
st.write("- Cross-validation: StratifiedKFold (5 folds)")
st.write("- Next step: add SHAP explainability notebook and Optuna tuning runs")
