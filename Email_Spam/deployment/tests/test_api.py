from fastapi.testclient import TestClient

from Email_Spam.deployment.app import app


def test_health_and_prediction():
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["model_loaded"] is True

        response = client.post(
            "/predict",
            json={"text": "Free money! Click here now to claim your prize."},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["prediction"] in (0, 1)
        assert body["label"] in ("spam", "not_spam")
        assert isinstance(body["spam_score"], float)


def test_empty_email_is_rejected():
    with TestClient(app) as client:
        response = client.post("/predict", json={"text": ""})
        assert response.status_code == 422
