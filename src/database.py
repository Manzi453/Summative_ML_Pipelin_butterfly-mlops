"""
SQLite persistence for prediction logs and retraining uploads.
Uses stdlib sqlite3 — no external DB service needed for the deployment.
"""
import os
import sqlite3
import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "app.db")


def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            predicted_label TEXT,
            confidence REAL,
            created_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS uploads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            label TEXT,
            saved_path TEXT,
            used_for_retrain INTEGER DEFAULT 0,
            created_at TEXT
        )
    """)
    conn.commit()
    conn.close()


def log_prediction(filename: str, predicted_label: str, confidence: float):
    conn = get_connection()
    conn.execute(
        "INSERT INTO predictions (filename, predicted_label, confidence, created_at) VALUES (?, ?, ?, ?)",
        (filename, predicted_label, confidence, datetime.datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()


def log_upload(filename: str, label: str, saved_path: str):
    conn = get_connection()
    conn.execute(
        "INSERT INTO uploads (filename, label, saved_path, created_at) VALUES (?, ?, ?, ?)",
        (filename, label, saved_path, datetime.datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()


def mark_uploads_used(ids):
    if not ids:
        return
    conn = get_connection()
    q_marks = ",".join("?" for _ in ids)
    conn.execute(f"UPDATE uploads SET used_for_retrain = 1 WHERE id IN ({q_marks})", ids)
    conn.commit()
    conn.close()


def get_pending_uploads():
    conn = get_connection()
    rows = conn.execute("SELECT * FROM uploads WHERE used_for_retrain = 0").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_stats():
    conn = get_connection()
    pred_count = conn.execute("SELECT COUNT(*) as c FROM predictions").fetchone()["c"]
    upload_count = conn.execute("SELECT COUNT(*) as c FROM uploads").fetchone()["c"]
    pending = conn.execute("SELECT COUNT(*) as c FROM uploads WHERE used_for_retrain = 0").fetchone()["c"]
    conn.close()
    return {"total_predictions": pred_count, "total_uploads": upload_count, "pending_retrain_images": pending}
