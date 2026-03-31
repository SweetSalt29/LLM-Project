from fastapi.testclient import TestClient
from backend.main import app
import pytest


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def test_user(client):
    user_data = {
        "name": "testuser",
        "password": "testpass"
    }

    client.post("/auth/register", json=user_data)

    response = client.post(
        "/auth/login",
        data={"username": "testuser", "password": "testpass"}
    )

    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}