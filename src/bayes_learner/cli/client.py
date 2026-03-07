"""HTTP client for bayes-learner server."""
import httpx
from contextlib import contextmanager

@contextmanager
def get_client(base_url: str = "http://localhost:8001"):
    with httpx.Client(base_url=base_url, timeout=30.0) as client:
        yield client