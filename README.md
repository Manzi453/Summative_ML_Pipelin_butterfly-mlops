# Butterfly & Moth Classifier — MLOps Pipeline

100-species image classifier (EfficientNetB0 transfer learning) with a full
predict → upload → retrain loop, FastAPI backend, Streamlit UI, Docker, and
Locust load testing.

Dataset: [Butterfly & Moths Image Classification, 100 species](https://www.kaggle.com/datasets/gpiosenka/butterfly-images40-species)

## 1. Get the data

```bash
kaggle datasets download -d gpiosenka/butterfly-images40-species --unzip -p data/
```

You should end up with:
```
data/
├── train/<species>/*.jpg
├── valid/<species>/*.jpg
├── test/<species>/*.jpg
└── butterflies_and_moths.csv   (already included in this repo)
```

## 2. Local setup (no Docker)

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Train the model:
```bash
jupyter notebook notebook/training_notebook.ipynb
# Run all cells — saves models/model.keras at the end
```

Run the API:
```bash
uvicorn app:app --reload --port 8000
```

Run the UI (separate terminal):
```bash
streamlit run ui/streamlit_app.py
```

Open http://localhost:8501

## 3. Docker (recommended for the deployment package)

```bash
docker compose up --build
```
- API: http://localhost:8000/docs (Swagger UI)
- UI: http://localhost:8501

To test with multiple API containers (for the Locust comparison table):
```bash
docker compose up --build --scale api=2
```

## 4. Load testing

```bash
locust -f locustfile.py --host http://localhost:8000
```
Open http://localhost:8089, set users/spawn rate, run against 1 vs 2 API
containers and record requests/sec + p95 latency for the comparison table.

## 5. Deploy

Render (or Railway):
- New Web Service → connect this repo
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
- Deploy a second service for the Streamlit UI with
  `streamlit run ui/streamlit_app.py --server.port $PORT --server.address 0.0.0.0`
  and set `API_URL` env var to the API service's public URL.

## 6. Rubric → file map

| Criterion | Where |
|---|---|
| Video Demo | Record using the deployed/local Streamlit UI |
| Retraining Process | `app.py` `/upload` + `/retrain`, `src/model.py` warm-start logic |
| Prediction Process | `app.py` `/predict`, `ui/streamlit_app.py` Predict tab |
| Evaluation of Models | `notebook/training_notebook.ipynb` |
| Deployment Package | Docker + Render URL + Streamlit Insights tab |

## Notes

- Model uses a **frozen EfficientNetB0 backbone** — fast to train/retrain even on CPU.
- Retraining is **warm-start**: it loads the existing `models/model.keras` and
  fine-tunes further at a lower learning rate, rather than training from scratch —
  this directly satisfies the rubric's "uses a custom model created as a
  pre-trained model" requirement.
- Old models are archived (timestamped) in `models/archive/` on every retrain.
