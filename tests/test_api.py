"""API tests isolate scan history in a temporary SQLite database."""

from fastapi.testclient import TestClient

from scanner.api import app


def test_scan_valid_url_stores_result(tmp_path, monkeypatch):
    monkeypatch.setenv("URL_SCANNER_DB_PATH", str(tmp_path / "test.sqlite3"))
    monkeypatch.delenv("URL_SCANNER_API_KEY", raising=False)
    client = TestClient(app)
    response = client.post("/scan", json={"url": "https://example.com/"})
    assert response.status_code == 200
    body = response.json()
    assert body["url"] == "https://example.com/"
    assert 0 <= body["score"] <= 1
    assert body["level"] in {"LOW", "MEDIUM", "HIGH"}
    assert isinstance(body["reasons"], list)
    assert body["model"] in {"heuristic", "trained_model"}
    assert body["scanned_at"]


def test_empty_url_is_rejected(tmp_path, monkeypatch):
    monkeypatch.delenv("URL_SCANNER_API_KEY", raising=False)
    monkeypatch.setenv("URL_SCANNER_DB_PATH", str(tmp_path / "empty.sqlite3"))
    response = TestClient(app).post("/scan", json={"url": ""})
    assert response.status_code == 422


def test_overlong_url_is_rejected(tmp_path, monkeypatch):
    monkeypatch.delenv("URL_SCANNER_API_KEY", raising=False)
    monkeypatch.setenv("URL_SCANNER_DB_PATH", str(tmp_path / "long.sqlite3"))
    response = TestClient(app).post("/scan", json={"url": "a" * 2049})
    assert response.status_code == 422


def test_history_returns_recent_scans(tmp_path, monkeypatch):
    monkeypatch.setenv("URL_SCANNER_DB_PATH", str(tmp_path / "history.sqlite3"))
    monkeypatch.delenv("URL_SCANNER_API_KEY", raising=False)
    client = TestClient(app)
    client.post("/scan", json={"url": "https://first.example/"})
    client.post("/scan", json={"url": "https://second.example/"})
    response = client.get("/history?limit=1")
    assert response.status_code == 200
    assert len(response.json()) == 1
    assert response.json()[0]["url"] == "https://second.example/"


def test_health_endpoint():
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_auth_disabled_when_environment_key_is_unset(monkeypatch, tmp_path):
    monkeypatch.delenv("URL_SCANNER_API_KEY", raising=False)
    monkeypatch.setenv("URL_SCANNER_DB_PATH", str(tmp_path / "open.sqlite3"))
    assert TestClient(app).get("/history").status_code == 200


def test_auth_required_when_environment_key_is_set(monkeypatch, tmp_path):
    monkeypatch.setenv("URL_SCANNER_API_KEY", "test-secret")
    monkeypatch.setenv("URL_SCANNER_DB_PATH", str(tmp_path / "secured.sqlite3"))
    client = TestClient(app)
    assert client.get("/history").status_code == 401
    assert client.post("/scan", json={"url": "https://example.com"}).status_code == 401
    assert client.get("/history", headers={"X-API-Key": "test-secret"}).status_code == 200


def test_health_is_public_with_auth_enabled(monkeypatch):
    monkeypatch.setenv("URL_SCANNER_API_KEY", "test-secret")
    assert TestClient(app).get("/health").status_code == 200


def test_web_dashboard_and_static_assets_are_served():
    client = TestClient(app)
    assert "URL Threat Scanner" in client.get("/").text
    assert client.get("/static/app.js").status_code == 200


def test_public_config_reports_availability_without_secret(monkeypatch):
    monkeypatch.setenv("URL_SCANNER_API_KEY", "keep-this-private")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-private")
    response = TestClient(app).get("/config")
    assert response.status_code == 200
    assert response.json() == {"api_key_required": True, "ai_explanations_enabled": True}
    assert "private" not in response.text


def test_explain_without_provider_key_uses_deterministic_fallback(monkeypatch):
    monkeypatch.delenv("URL_SCANNER_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    response = TestClient(app).post(
        "/explain", json={"score": 0.8, "level": "HIGH", "reasons": ["URL contains an @ marker."]}
    )
    assert response.status_code == 200
    assert response.json()["provider"] == "deterministic"
    assert "@ marker" in response.json()["explanation"]


def test_explain_validates_level_and_reason_limits(monkeypatch):
    monkeypatch.delenv("URL_SCANNER_API_KEY", raising=False)
    client = TestClient(app)
    assert client.post("/explain", json={"score": 0.5, "level": "SAFE", "reasons": []}).status_code == 422
    assert client.post("/explain", json={"score": 0.5, "level": "MEDIUM", "reasons": ["x" * 301]}).status_code == 422


def test_explain_requires_existing_api_key_when_configured(monkeypatch):
    monkeypatch.setenv("URL_SCANNER_API_KEY", "explain-secret")
    response = TestClient(app).post("/explain", json={"score": 0.5, "level": "MEDIUM", "reasons": []})
    assert response.status_code == 401


def test_explain_endpoint_keeps_scan_result_unchanged(monkeypatch):
    monkeypatch.delenv("URL_SCANNER_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    scan_result = {"score": 0.72, "level": "HIGH", "reasons": ["Long URL."]}
    response = TestClient(app).post("/explain", json=scan_result)
    assert response.status_code == 200
    assert response.json()["provider"] == "deterministic"
    assert response.json()["explanation"]
    assert scan_result == {"score": 0.72, "level": "HIGH", "reasons": ["Long URL."]}
