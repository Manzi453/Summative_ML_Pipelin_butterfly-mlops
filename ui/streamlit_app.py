"""
Streamlit UI — three tabs:
  Predict  : upload an image, get species + confidence + top-3
  Retrain  : upload labeled images, trigger warm-start retrain
  Insights : dataset class distribution + sample grid + model status
"""
import os
import io
import requests
import pandas as pd
import streamlit as st
from PIL import Image

API_URL = os.environ.get("API_URL", "http://localhost:8000")
CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "butterflies_and_moths.csv")

st.set_page_config(page_title="Butterfly & Moth Classifier", layout="wide")
st.title("🦋 Butterfly & Moth Classifier — MLOps Pipeline")

tab_predict, tab_retrain, tab_insights = st.tabs(["Predict", "Retrain", "Insights"])

# ---------------------------------------------------------------- Predict --
with tab_predict:
    st.subheader("Predict a species")
    uploaded = st.file_uploader("Upload a butterfly/moth image", type=["jpg", "jpeg", "png"], key="predict_upload")

    if uploaded is not None:
        st.image(uploaded, width=300)
        if st.button("Run prediction"):
            files = {"file": (uploaded.name, uploaded.getvalue())}
            with st.spinner("Calling prediction API..."):
                resp = requests.post(f"{API_URL}/predict", files=files, timeout=60)
            if resp.status_code == 200:
                result = resp.json()
                if "error" in result:
                    st.error(result["error"])
                else:
                    st.success(f"Predicted: **{result['label']}** ({result['confidence']*100:.1f}% confidence)")
                    st.write("Top 3:")
                    for p in result["top_predictions"]:
                        st.write(f"- {p['label']}: {p['confidence']*100:.1f}%")
            else:
                st.error(f"API error: {resp.status_code} — {resp.text}")

# ---------------------------------------------------------------- Retrain --
with tab_retrain:
    st.subheader("Upload labeled images for retraining")

    try:
        df = pd.read_csv(CSV_PATH)
        species_list = sorted(df["labels"].unique().tolist())
    except Exception:
        species_list = []

    label = st.selectbox("Species label for these images", species_list)
    batch = st.file_uploader(
        "Upload one or more images of this species", type=["jpg", "jpeg", "png"],
        accept_multiple_files=True, key="retrain_upload",
    )

    if batch and st.button("Upload batch"):
        progress = st.progress(0)
        for i, f in enumerate(batch):
            files = {"file": (f.name, f.getvalue())}
            data = {"label": label}
            requests.post(f"{API_URL}/upload", files=files, data=data, timeout=60)
            progress.progress((i + 1) / len(batch))
        st.success(f"Uploaded {len(batch)} image(s) for '{label}'.")

    st.divider()
    st.subheader("Trigger retraining")
    epochs = st.slider("Epochs", min_value=1, max_value=15, value=5)
    if st.button("Retrain model now"):
        with st.spinner("Retraining — this can take a few minutes..."):
            resp = requests.post(f"{API_URL}/retrain", params={"epochs": epochs}, timeout=1800)
        if resp.status_code == 200:
            result = resp.json()
            if "error" in result:
                st.warning(result["error"])
            else:
                st.success(f"Retrained on {result['images_used']} images. "
                           f"Final accuracy: {result['final_accuracy']*100:.1f}%")
        else:
            st.error(f"API error: {resp.status_code} — {resp.text}")

# --------------------------------------------------------------- Insights --
with tab_insights:
    st.subheader("Dataset insights")
    try:
        df = pd.read_csv(CSV_PATH)
        counts = df[df["data set"] == "train"]["labels"].value_counts() if "data set" in df.columns else df["labels"].value_counts()

        col1, col2 = st.columns(2)
        with col1:
            st.write("Images per species (top 20)")
            st.bar_chart(counts.head(20))
        with col2:
            st.write("Dataset summary")
            st.metric("Total species", df["labels"].nunique())
            st.metric("Total labeled images", len(df))

    except Exception as e:
        st.warning(f"Could not load dataset CSV for insights: {e}")

    st.divider()
    st.subheader("Model status")
    try:
        resp = requests.get(f"{API_URL}/status", timeout=10)
        if resp.status_code == 200:
            st.json(resp.json())
    except Exception as e:
        st.warning(f"Could not reach API for status: {e}")
