"""Streamlit dashboard: indoor zone (building + floor) classification from WiFi RSSI."""
import glob
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from common import (TARGET, NO_SIGNAL, add_noise, evaluate, add_signal_features,  # noqa: F401 (needed to unpickle)
                    split_zone, level_accuracies)

st.set_page_config(page_title="WiFi Zone Locator", page_icon="📶", layout="wide")
st.title("📶 Indoor Zone Prediction from WiFi Signal Strength")
st.caption("Five classifiers trained on UJIIndoorLoc RSSI fingerprints. "
           "Upload test data, stress it with signal noise, and compare models. No retraining happens here.")

@st.cache_resource
def load_models():
    out = {}
    for p in sorted(glob.glob("model/*.joblib")):
        label = os.path.basename(p)[:-7].replace("_", " ").title()
        out[label] = joblib.load(p)
    return out

models = load_models()
if not models:
    st.error("No saved models found in model/. Run train_models.py first.")
    st.stop()

with st.sidebar:
    st.header("Controls")
    up = st.file_uploader("Test data (CSV with RSSI columns + 'zone')", type="csv")
    choice = st.selectbox("Model", list(models))
    st.subheader("Signal stress test")
    sigma = st.slider("RSSI noise (dB, std dev)", 0, 12, 0)
    drop = st.slider("AP dropout (%)", 0, 50, 0)

if up is None:
    st.info("Upload test_data.csv from the repo (or any CSV with the same columns) to begin.")
    st.stop()

data = pd.read_csv(up)
if TARGET not in data.columns:
    st.error(f"CSV must contain the target column '{TARGET}'.")
    st.stop()

model = models[choice]
feat_cols = [c for c in data.columns if c != TARGET]
X, y = data[feat_cols], data[TARGET]
if sigma or drop:
    X = pd.DataFrame(add_noise(X, sigma, drop / 100, seed=1), columns=feat_cols)
    st.warning(f"Stress test active: noise σ={sigma} dB, {drop}% AP dropout")

res = evaluate(model, X, y)
st.subheader(f"{choice}: evaluation on {len(data)} rows")
cols = st.columns(6)
for c, k in zip(cols, ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]):
    c.metric(k, f"{res[k]:.4f}")
st.caption(f"Inference speed: {res['Inference ms/1k rows']:.1f} ms per 1000 rows")

pred = model.predict(X)
tab1, tab2, tab3, tab4 = st.tabs(["Confusion matrix", "Classification report", "All models",
                                  "Building & floor"])
with tab1:
    labels = list(model.classes_)
    cm = confusion_matrix(y, pred, labels=labels)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="viridis", xticklabels=labels,
                yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted zone"); ax.set_ylabel("Actual zone")
    st.pyplot(fig)
with tab2:
    st.dataframe(pd.DataFrame(classification_report(y, pred, output_dict=True,
                                                    zero_division=0)).T.round(3))
with tab3:
    comp = pd.DataFrame({n: evaluate(m, X, y) for n, m in models.items()}).T.round(4)
    st.dataframe(comp)
    st.bar_chart(comp[["Accuracy", "F1", "MCC"]])

with tab4:
    st.markdown("**How well does the model find the building, the floor, and the exact zone?**")
    lv = level_accuracies(y, pred)
    c1, c2, c3 = st.columns(3)
    c1.metric("Building correct", f"{lv['Building']:.4f}")
    c2.metric("Floor correct", f"{lv['Floor']:.4f}")
    c3.metric("Exact zone correct", f"{lv['Exact zone']:.4f}")

    st.divider()
    st.markdown("**Try a single fingerprint**")
    idx = st.slider("Row number in the uploaded file", 0, len(X) - 1, 0)
    row = X.iloc[[idx]]
    heard = int((row.values > NO_SIGNAL).sum())
    st.caption(f"This fingerprint hears {heard} access points; strongest signal "
               f"{row.values.max():.0f} dBm.")
    guess = model.predict(row)[0]
    gb, gf = split_zone(guess)
    tb, tf = split_zone(y.iloc[idx])
    a, b, c, d = st.columns(4)
    a.metric("Predicted building", gb, delta=None if gb == tb else f"actual {tb}", delta_color="inverse")
    b.metric("Predicted floor", gf, delta=None if gf == tf else f"actual {tf}", delta_color="inverse")
    c.metric("Actual building", tb)
    d.metric("Actual floor", tf)
    probs = pd.Series(model.predict_proba(row)[0], index=model.classes_).sort_values(ascending=False)
    st.caption("Model confidence for the three most likely zones")
    st.bar_chart(probs.head(3))
