"""
Locust load test for the /predict endpoint.
Run: locust -f locustfile.py --host http://localhost:8000
Then open http://localhost:8089 to set users/spawn-rate and start the swarm.

For the multi-container comparison required by the rubric: docker-compose.yml
publishes a fixed host port for the api service, so `docker compose up
--scale api=2` can't bind two containers to the same host port. Instead, run
a second container manually on a different host port and set API_HOSTS to a
comma-separated list of base URLs (e.g.
API_HOSTS=http://localhost:8000,http://localhost:8001) to have Locust send
requests round-robin across containers, client-side. See README section
"Load testing" for the exact commands and recorded results.
"""
import os
import glob
import random
from locust import HttpUser, task, between

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "data", "test")
API_HOSTS = [h.strip() for h in os.environ.get("API_HOSTS", "").split(",") if h.strip()]


def _sample_images():
    paths = glob.glob(os.path.join(SAMPLE_DIR, "**", "*.jpg"), recursive=True)
    return paths


class PredictUser(HttpUser):
    wait_time = between(1, 3)

    def on_start(self):
        self.images = _sample_images()

    def _url(self, path):
        if API_HOSTS:
            return random.choice(API_HOSTS) + path
        return path

    @task(3)
    def predict(self):
        if not self.images:
            return
        path = random.choice(self.images)
        with open(path, "rb") as f:
            self.client.post(self._url("/predict"), files={"file": (os.path.basename(path), f, "image/jpeg")})

    @task(1)
    def status(self):
        self.client.get(self._url("/status"))
