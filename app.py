"""
FastAPI app. Endpoints:
  POST /predict  -> single image prediction
  POST /upload   -> save a labeled image for future retraining
  POST /retrain  -> warm-start retrain on accumulated uploads
  GET  /status   -> model version / dataset size / class list
"""
import os
import shutil
import datetime

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

from src import database
from src.prediction import predict_image
from src.model import retrain as model_retrain, MODEL_PATH
from src.preprocessing import get_class_list

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOADS_DIR = os.path.join(BASE_DIR, "data", "uploads")

app = FastAPI(title="Butterfly & Moth Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    database.init_db()
    os.makedirs(UPLOADS_DIR, exist_ok=True)


@app.get("/status")
def status():
    classes, _, _ = get_class_list()
    stats = database.get_stats()
    model_exists = os.path.exists(MODEL_PATH)
    last_modified = (
        datetime.datetime.fromtimestamp(os.path.getmtime(MODEL_PATH)).isoformat()
        if model_exists
        else None
    )
    return {
        "model_loaded": model_exists,
        "last_retrain": last_modified,
        "num_classes": len(classes),
        **stats,
    }


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    image_bytes = await file.read()
    result = predict_image(image_bytes, filename=file.filename)

    if "error" not in result:
        database.log_prediction(file.filename, result["label"], result["confidence"])

    return result


@app.post("/upload")
async def upload_endpoint(file: UploadFile = File(...), label: str = Form(...)):
    classes, _, _ = get_class_list()
    if label not in classes:
        return {"error": f"Unknown label '{label}'. Must be one of the existing 100 species."}

    class_dir = os.path.join(UPLOADS_DIR, label)
    os.makedirs(class_dir, exist_ok=True)

    save_path = os.path.join(class_dir, file.filename)
    with open(save_path, "wb") as f:
        contents = await file.read()
        f.write(contents)

    database.log_upload(file.filename, label, save_path)

    return {"status": "saved", "path": save_path, "label": label}


@app.post("/retrain")
def retrain_endpoint(epochs: int = 5):
    pending = database.get_pending_uploads()
    if len(pending) == 0:
        return {"error": "No new uploaded images to retrain on. Use /upload first."}

    result = model_retrain(new_data_dir=UPLOADS_DIR, epochs=epochs)

    ids = [row["id"] for row in pending]
    database.mark_uploads_used(ids)

    return {"status": "retrained", **result, "images_used": len(pending)}
