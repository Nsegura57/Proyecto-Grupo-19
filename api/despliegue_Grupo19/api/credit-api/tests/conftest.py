import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture()
def client():
    with TestClient(app) as _client:
        yield _client
    app.dependency_overrides = {}


@pytest.fixture()
def credito_ejemplo():
    return {"P2097": 1, "P2098": 2, "P2106": 4, "P2099S2": 2018,
            "P2104": 25_000_000, "P2105S1": 450_000, "P2107S1": 60}
