def test_query_empty(client, test_user):
    response = client.post(
        "/query/chat",
        json={"query": "", "mode": "rag"},
        headers=test_user
    )

    assert response.status_code == 400