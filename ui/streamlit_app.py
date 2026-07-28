"""
Streamlit UI — three tabs:
  Predict  : upload an image, get species + confidence + top-3
  Retrain  : upload labeled images, trigger warm-start retrain
  Insights : dataset class distribution + sample grid + model status

Calls src/ prediction, model, and database logic directly in-process
(no separate FastAPI backend) so the app runs as a single Streamlit
Community Cloud service. app.py + docker-compose.yml still expose the
same logic over HTTP for local/Docker use.
"""
import os
import sys
import datetime

import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src import database
from src.prediction import predict_image
from src.model import retrain as model_retrain, MODEL_PATH
from src.preprocessing import get_class_list

UPLOADS_DIR = os.path.join(BASE_DIR, "data", "uploads")
CSV_PATH = os.path.join(BASE_DIR, "data", "butterflies_and_moths.csv")

st.set_page_config(page_title="Butterfly & Moth Classifier", layout="wide")


@st.cache_resource(show_spinner=False)
def init_backend():
    database.init_db()
    os.makedirs(UPLOADS_DIR, exist_ok=True)


init_backend()

st.title("🦋 Butterfly & Moth Classifier — MLOps Pipeline")

tab_predict, tab_retrain, tab_insights = st.tabs(["Predict", "Retrain", "Insights"])

# ---------------------------------------------------------------- Predict --
with tab_predict:
    st.subheader("Predict a species")
    uploaded = st.file_uploader("Upload a butterfly/moth image", type=["jpg", "jpeg", "png"], key="predict_upload")

    if uploaded is not None:
        st.image(uploaded, width=300)
        if st.button("Run prediction"):
            with st.spinner("Running prediction..."):
                result = predict_image(uploaded.getvalue(), filename=uploaded.name)
            if "error" in result:
                st.error(result["error"])
            else:
                database.log_prediction(uploaded.name, result["label"], result["confidence"])
                st.success(f"Predicted: **{result['label']}** ({result['confidence']*100:.1f}% confidence)")
                st.write("Top 3:")
                for p in result["top_predictions"]:
                    st.write(f"- {p['label']}: {p['confidence']*100:.1f}%")

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
        classes, _, _ = get_class_list()
        progress = st.progress(0)
        for i, f in enumerate(batch):
            if label not in classes:
                st.error(f"Unknown label '{label}'. Must be one of the existing 100 species.")
                break
            class_dir = os.path.join(UPLOADS_DIR, label)
            os.makedirs(class_dir, exist_ok=True)
            save_path = os.path.join(class_dir, f.name)
            with open(save_path, "wb") as out:
                out.write(f.getvalue())
            database.log_upload(f.name, label, save_path)
            progress.progress((i + 1) / len(batch))
        else:
            st.success(f"Uploaded {len(batch)} image(s) for '{label}'.")

    st.divider()
    st.subheader("Trigger retraining")
    epochs = st.slider("Epochs", min_value=1, max_value=15, value=5)
    if st.button("Retrain model now"):
        pending = database.get_pending_uploads()
        if not pending:
            st.warning("No new uploaded images to retrain on. Upload some first.")
        else:
            with st.spinner("Retraining — this can take a few minutes..."):
                result = model_retrain(new_data_dir=UPLOADS_DIR, epochs=epochs)
                database.mark_uploads_used([row["id"] for row in pending])
            st.success(f"Retrained on {len(pending)} images. "
                       f"Final accuracy: {result['final_accuracy']*100:.1f}%")

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
        classes, _, _ = get_class_list()
        model_exists = os.path.exists(MODEL_PATH)
        last_modified = (
            datetime.datetime.fromtimestamp(os.path.getmtime(MODEL_PATH)).isoformat()
            if model_exists
            else None
        )
        st.json({
            "model_loaded": model_exists,
            "last_retrain": last_modified,
            "num_classes": len(classes),
            **database.get_stats(),
        })
    except Exception as e:
        st.warning(f"Could not load model status: {e}")
