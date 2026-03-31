def test_register(client):
    response = client.post("/auth/register", json={
        "name": "user1",
        "password": "pass123"
    })
    assert response.status_code == 200


def test_login(client):
    client.post("/auth/register", json={
        "name": "user2",
        "password": "pass123"
    })

    response = client.post(
        "/auth/login",
        data={"username": "user2", "password": "pass123"}
    )

    assert response.status_code == 200
    assert "access_token" in response.json()