"""
Locust load test for the /predict endpoint.
Run: locust -f locustfile.py --host http://localhost:8000
Then open http://localhost:8089 to set users/spawn-rate and start the swarm.

For the multi-container comparison required by the rubric: run this against
two different container configs (e.g. 1 API replica vs 2 API replicas behind
docker-compose --scale api=2) and record requests/sec + latency for each in
your README/report.
"""
import os
import glob
import random
from locust import HttpUser, task, between

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "data", "test")


def _sample_images():
    paths = glob.glob(os.path.join(SAMPLE_DIR, "**", "*.jpg"), recursive=True)
    return paths


class PredictUser(HttpUser):
    wait_time = between(1, 3)

    def on_start(self):
        self.images = _sample_images()

    @task(3)
    def predict(self):
        if not self.images:
            return
        path = random.choice(self.images)
        with open(path, "rb") as f:
            self.client.post("/predict", files={"file": (os.path.basename(path), f, "image/jpeg")})

    @task(1)
    def status(self):
        self.client.get("/status")
