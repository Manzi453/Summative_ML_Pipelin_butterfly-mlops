"""
Shared preprocessing used by both the training notebook and the FastAPI app.
Single source of truth for image size, normalization, and label mapping so
train-time and serve-time preprocessing can never drift apart.
"""
import io
import os
import numpy as np
import pandas as pd
from PIL import Image

IMG_SIZE = (224, 224)
CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "butterflies_and_moths.csv")


def load_labels_df(csv_path: str = CSV_PATH) -> pd.DataFrame:
    """Load the label index CSV (class id, filepaths, labels, data set)."""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


_CLASS_LIST_CACHE = {}


def get_class_list(csv_path: str = CSV_PATH):
    """Return sorted class names and label<->index maps, derived from the CSV
    so the notebook and the API always agree on class ordering. Cached per
    csv_path since the CSV never changes after training, avoiding a full
    pandas read + sort on every API request."""
    if csv_path not in _CLASS_LIST_CACHE:
        df = load_labels_df(csv_path)
        classes = sorted(df["labels"].unique().tolist())
        label_to_idx = {label: i for i, label in enumerate(classes)}
        idx_to_label = {i: label for label, i in label_to_idx.items()}
        _CLASS_LIST_CACHE[csv_path] = (classes, label_to_idx, idx_to_label)
    return _CLASS_LIST_CACHE[csv_path]


def preprocess_array(img_array: np.ndarray) -> np.ndarray:
    """Resize a HWC uint8/float array to IMG_SIZE, kept in raw [0,255] float32.
    Not divided by 255: EfficientNetB0 has its own built-in Rescaling/Normalization
    layers and expects raw pixel values, matching the training pipeline."""
    img = Image.fromarray(img_array.astype("uint8")).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.asarray(img, dtype="float32")
    return arr


def load_and_preprocess_image(source) -> np.ndarray:
    """Accepts a file path (str), raw bytes, or a file-like object.
    Returns a (224, 224, 3) float32 array in raw [0,255] range (not
    normalized: EfficientNetB0 rescales/normalizes internally)."""
    if isinstance(source, (bytes, bytearray)):
        img = Image.open(io.BytesIO(source))
    elif isinstance(source, str):
        img = Image.open(source)
    else:
        img = Image.open(source)

    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(img, dtype="float32")
    return arr


def batch_for_model(arr: np.ndarray) -> np.ndarray:
    """Add batch dimension: (224,224,3) -> (1,224,224,3)."""
    return np.expand_dims(arr, axis=0)


def build_tf_datasets(data_dir: str, batch_size: int = 32, seed: int = 42):
    """Build train/valid/test tf.data datasets from the standard
    data_dir/{train,valid,test}/<class_name>/*.jpg folder layout.
    Kept separate from the API path — only used inside the notebook."""
    import tensorflow as tf

    train_dir = os.path.join(data_dir, "train")
    valid_dir = os.path.join(data_dir, "valid")
    test_dir = os.path.join(data_dir, "test")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, image_size=IMG_SIZE, batch_size=batch_size, seed=seed, label_mode="categorical"
    )
    valid_ds = tf.keras.utils.image_dataset_from_directory(
        valid_dir, image_size=IMG_SIZE, batch_size=batch_size, seed=seed, label_mode="categorical"
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir, image_size=IMG_SIZE, batch_size=batch_size, seed=seed, label_mode="categorical", shuffle=False
    )

    class_names = train_ds.class_names

    # No Rescaling(1/255): EfficientNetB0 has its own built-in Rescaling/
    # Normalization layers and expects raw [0,255] pixel values.
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(AUTOTUNE)
    valid_ds = valid_ds.prefetch(AUTOTUNE)
    test_ds = test_ds.prefetch(AUTOTUNE)

    return train_ds, valid_ds, test_ds, class_names
