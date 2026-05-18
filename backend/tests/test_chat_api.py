from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_stream_endpoint_accepts_request():
    response = client.post("/api/chat/stream", json={
        "question": "你好",
        "history": [],
        "has_docs": False,
    })
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
