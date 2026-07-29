# Butterfly & Moth Classifier — MLOps Pipeline

An end-to-end machine learning system that classifies images across 100
butterfly and moth species. The project covers the full lifecycle — data
processing, model training, evaluation, deployment, prediction, and
on-demand retraining — packaged behind a FastAPI backend, a Streamlit UI,
and Docker, with load testing via Locust.

**Live app:** https://summativebutterfly-mlops.streamlit.app/

**Video demo:** https://www.awesomescreenshot.com/video/55071011?key=4a7b70347c46e563419ba396ac6d6cbb

## Overview

The model is an EfficientNetB0 transfer-learning classifier trained on the
[Butterfly & Moths Image Classification dataset](https://www.kaggle.com/datasets/gpiosenka/butterfly-images40-species)
(100 species, ~13,600 labeled images). Around this model sits a small
production system:

- **Predict** — upload a single image and get back the predicted species,
  confidence score, and top-3 alternatives.
- **Upload & retrain** — upload new labeled images, which are persisted to
  disk and logged to a database; a retraining job then fine-tunes the
  existing model on the newly uploaded data and versions the result.
- **Insights** — dataset composition, prediction/upload activity, and
  service uptime, surfaced in the UI.
- **Load testing** — Locust-driven simulation of concurrent prediction
  traffic, comparing latency and throughput across one and two API
  containers.

## Architecture

```
                 ┌─────────────────┐
                 │   Streamlit UI   │  predict / upload / retrain / insights
                 └────────┬─────────┘
                          │
                 ┌────────▼─────────┐
                 │   FastAPI (app)   │  /predict /upload /retrain /status
                 └────────┬─────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
┌───────▼──────┐  ┌───────▼───────┐  ┌──────▼───────┐
│ src/model.py │  │ src/database  │  │src/preprocess│
│ predict /    │  │ SQLite log of │  │ shared image │
│ retrain /    │  │ predictions & │  │ pipeline     │
│ save (warm-  │  │ uploads       │  │              │
│ start)       │  │               │  │              │
└──────────────┘  └───────────────┘  └──────────────┘
```

Retraining is **warm-start**: each retrain job loads the current
`models/model.keras`, fine-tunes it further on newly uploaded images at a
reduced learning rate, and saves the result — the previous version is
archived with a timestamp rather than discarded, so every model version
remains recoverable.

## Project structure

```
├── README.md
├── app.py                    # FastAPI application
├── Dockerfile
├── docker-compose.yml
├── locustfile.py             # Load testing
├── requirements.txt
├── notebook/
│   └── training_notebook.ipynb   # Data prep, training, evaluation
├── src/
│   ├── preprocessing.py      # Image loading & preprocessing
│   ├── model.py              # Build / load / save / predict / retrain
│   ├── prediction.py         # Prediction entry point
│   └── database.py           # SQLite persistence for predictions/uploads
├── ui/
│   └── streamlit_app.py      # Predict, Retrain, and Insights tabs
├── data/
│   ├── train/ valid/ test/   # Training data
│   ├── uploads/              # New data submitted for retraining
│   └── butterflies_and_moths.csv
└── models/
    ├── model.keras           # Current production model
    └── archive/              # Timestamped previous versions
```

## Model performance

Evaluated on the held-out test set (`notebook/training_notebook.ipynb`):

| Metric | Score |
|---|---|
| Accuracy | 0.946 |
| Loss | 0.173 |
| Precision (weighted) | 0.955 |
| Recall (weighted) | 0.946 |
| F1 score (weighted) | 0.945 |

The model uses a frozen, pretrained EfficientNetB0 backbone with a
lightweight classification head (dropout + dense layers), trained with data
augmentation, early stopping, and learning-rate reduction on plateau.

## Getting started

### Prerequisites

- Python 3.11
- (Optional) Docker & Docker Compose
- (Optional) A Kaggle account, if you want to re-download the raw dataset

### 1. Get the data

The repository already includes the training, validation, and test images
and label CSV under `data/`. To re-download from Kaggle instead:

```bash
kaggle datasets download -d gpiosenka/butterfly-images40-species --unzip -p data/
```

### 2. Local setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Train the model (optional — a trained `models/model.keras` is already included):

```bash
jupyter notebook notebook/training_notebook.ipynb
# Run all cells — saves models/model.keras at the end
```

Run the API:

```bash
uvicorn app:app --reload --port 8000
```

Run the UI (in a separate terminal):

```bash
streamlit run ui/streamlit_app.py
```

Open http://localhost:8501 for the UI and http://localhost:8000/docs for
the interactive API docs.

### 3. Docker

```bash
docker compose up --build
```

- API: http://localhost:8000/docs
- UI: http://localhost:8501

## API reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/status` | Model status, last retrain time, class count, uptime, DB stats |
| POST | `/predict` | Predict the species for a single uploaded image |
| POST | `/upload` | Save a labeled image for later use in retraining |
| POST | `/retrain` | Fine-tune the current model on pending uploaded images |

## Load testing

Simulated concurrent traffic against `/predict` using Locust, comparing one
API instance against two. Raw results (CSVs + HTML reports) are committed
under [`results/`](results/): `locust_1container.*` and
`locust_2containers.*`.

```bash
locust -f locustfile.py --host http://localhost:8000
```

### Results (20 users, spawn rate 5/s, 45s, `/predict` + `/status` mix)

| Instances | Total requests | Req/s | Failures | Median latency | p95 latency | p99 latency | Max latency |
|---|---|---|---|---|---|---|---|
| 1 | 381 | 8.62 | 0 (0.00%) | 180 ms | 840 ms | 1500 ms | 3156 ms |
| 2 | 399 | 9.04 | 0 (0.00%) | 100 ms | 370 ms | 4500 ms | 5422 ms |

Going from one to two instances roughly halved median latency (180ms →
100ms) and raised throughput slightly (8.62 → 9.04 req/s), but tail latency
(p99, max) got *worse*, not better. Both instances were run as separate
processes on the same machine for this test, so two concurrent, CPU-bound
EfficientNetB0 inference workloads ended up competing for the same CPU
cores instead of running on genuinely separate resources — the classic
case is worse-case latency actually improves, so this result depends on
available CPU headroom. On real infrastructure (separate Docker containers
scheduled across multiple cores/hosts, e.g. via `docker compose up --scale
api=2` behind a load balancer, or a cloud autoscaler), the extra CPU
headroom removes that contention and tail latency should improve alongside
throughput, matching the horizontal-scaling behavior the median-latency
improvement already hints at here.

<details>
<summary>Reproducing the 1 vs 2 container comparison</summary>

`docker-compose.yml` publishes a fixed host port for `api`, so
`docker compose up --scale api=2` can't bind two containers to the same
host port. Instead, run a second container manually on another port and
point Locust at both via `API_HOSTS` (see `locustfile.py`), which
round-robins requests across hosts client-side:

```bash
# 1 container
docker compose up -d api
locust -f locustfile.py --host http://localhost:8000 --headless -u 20 -r 5 -t 45s \
  --csv=results/locust_1container --html=results/locust_1container.html

# add a 2nd container on port 8001, same image
docker run -d --name butterfly-api-2 -p 8001:8000 \
  -v "$(pwd)/data":/app/data -v "$(pwd)/models":/app/models \
  summative_ml_pipelin_butterfly-mlops-api:latest uvicorn app:app --host 0.0.0.0 --port 8000

# 2 containers, same load profile
API_HOSTS="http://localhost:8000,http://localhost:8001" \
  locust -f locustfile.py --host http://localhost:8000 --headless -u 20 -r 5 -t 45s \
  --csv=results/locust_2containers --html=results/locust_2containers.html
```

The results above were captured the same way, using two local `uvicorn`
processes (ports 8000/8001) rather than `docker run` — same isolation model
(independent server processes), substituted because the Docker daemon
wasn't available in the environment this was reproduced in. Re-run the
commands above with a running Docker daemon for literal container-to-
container numbers if needed.

</details>

## Deployment

The live app is deployed on Streamlit Community Cloud:
https://summativebutterfly-mlops.streamlit.app/

The deployed Streamlit app calls the `src/` prediction, model, and database
logic directly in-process, so a single service handles both the UI and the
inference logic in production. `app.py` and `docker-compose.yml` expose the
same logic over HTTP for local and Docker use, and for the Locust load test
above.

To deploy the FastAPI service separately (e.g. Render or Railway):

- New Web Service → connect this repo
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
- Deploy a second service for the Streamlit UI with
  `streamlit run ui/streamlit_app.py --server.port $PORT --server.address 0.0.0.0`
  and set the `API_URL` environment variable to the API service's public URL.

## Tech stack

TensorFlow / Keras · FastAPI · Streamlit · SQLite · Docker · Locust ·
scikit-learn · pandas
