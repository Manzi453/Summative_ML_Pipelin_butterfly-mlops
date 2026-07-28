"""
Thin wrapper tying preprocessing + model.predict together for API use,
with basic input validation.
"""
from src.preprocessing import load_and_preprocess_image
from src.model import predict as model_predict

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def validate_filename(filename: str) -> bool:
    lower = filename.lower()
    return any(lower.endswith(ext) for ext in ALLOWED_EXTENSIONS)


def predict_image(image_bytes: bytes, filename: str = "") -> dict:
    if filename and not validate_filename(filename):
        return {"error": f"Unsupported file type: {filename}. Use jpg/jpeg/png."}

    try:
        arr = load_and_preprocess_image(image_bytes)
    except Exception as e:
        return {"error": f"Could not read image: {e}"}

    result = model_predict(arr)
    return result
