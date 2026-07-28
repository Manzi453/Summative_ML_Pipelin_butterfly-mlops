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

### Reproducing the 1 vs 2 container comparison

`docker-compose.yml` publishes a fixed host port for `api`, so
`docker compose up --scale api=2` can't bind two containers to the same host
port. Instead, run a second container manually on another port and point
Locust at both via `API_HOSTS` (see `locustfile.py`), which round-robins
requests across hosts client-side:

```bash
# 1 container
docker compose up -d api
locust -f locustfile.py --host http://localhost:8000 --headless -u 20 -r 5 -t 45s

# add a 2nd container on port 8001, same image
docker run -d --name butterfly-api-2 -p 8001:8000 \
  -v "$(pwd)/data":/app/data -v "$(pwd)/models":/app/models \
  summative_ml_pipelin_butterfly-mlops-api:latest uvicorn app:app --host 0.0.0.0 --port 8000

# 2 containers, same load profile
API_HOSTS="http://localhost:8000,http://localhost:8001" \
  locust -f locustfile.py --host http://localhost:8000 --headless -u 20 -r 5 -t 45s
```

### Results (20 users, spawn rate 5/s, 45s, `/predict` + `/status` mix)

| Containers | Total requests | Req/s | Failures | Median latency | p95 latency | p99 latency | Max latency |
|---|---|---|---|---|---|---|---|
| 1 | 264 | 6.18 | 0 (0.00%) | 550 ms | 6100 ms | 7100 ms | 7131 ms |
| 2 | 347 | 7.76 | 1 (0.29%)* | 260 ms | 2000 ms | 3900 ms | 4494 ms |

\* one transient `RemoteDisconnected` on `/status`, not a failed prediction.

Going from 1 to 2 containers raised throughput ~26% and cut p95 latency by
~3x (6.1s → 2.0s) and median latency by ~2x (550ms → 260ms). Each container
runs uvicorn single-process with a synchronous, CPU-bound EfficientNetB0
inference call, so a single container serializes requests under load;
horizontally scaling containers lets requests be served in parallel instead
of queueing behind one process.

## 5. Deploy

**Live app:** https://summativebutterfly-mlops.streamlit.app/

Deployed on Streamlit Community Cloud. The deployed Streamlit app calls
`src/` prediction, model, and database logic directly in-process (see
`ui/streamlit_app.py` docstring) so there is no separate hosted FastAPI
service for this deployment — `app.py` + `docker-compose.yml` still expose
the same logic over HTTP for local/Docker use and for the Locust load test
below.

To deploy the FastAPI service separately (e.g. Render or Railway):
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
| Deployment Package | Docker + [live Streamlit app](https://summativebutterfly-mlops.streamlit.app/) + Insights tab (uptime, dataset charts) |
| Flood Request Simulation | `locustfile.py` + "Load testing" section above (1 vs 2 container results) |

## Notes

- Model uses a **frozen EfficientNetB0 backbone** — fast to train/retrain even on CPU.
- Retraining is **warm-start**: it loads the existing `models/model.keras` and
  fine-tunes further at a lower learning rate, rather than training from scratch —
  this directly satisfies the rubric's "uses a custom model created as a
  pre-trained model" requirement.
- Old models are archived (timestamped) in `models/archive/` on every retrain.
