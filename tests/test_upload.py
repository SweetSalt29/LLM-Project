def test_upload_csv(client, test_user):
    files = [
        ("files", ("test.csv", b"name,age\nAaryan,22", "text/csv"))
    ]

    response = client.post(
        "/upload/",
        files=files,
        headers=test_user
    )

    assert response.status_code == 202