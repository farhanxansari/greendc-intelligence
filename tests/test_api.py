from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_metrics_default_window():
    r = client.get("/api/metrics")
    assert r.status_code == 200
    body = r.json()
    assert body["freq"] == "D" and len(body["points"]) > 0


def test_pue_summary_has_kpis():
    body = client.get("/api/pue/summary").json()
    assert 1.0 < body["kpis"]["last_30d"]["pue"] < 1.5


def test_forecast_next_returns_24_hours():
    body = client.get("/api/forecast/next").json()
    assert len(body["points"]) == 24


def test_anomalies_filter():
    body = client.get("/api/anomalies", params={"min_confidence": "high"}).json()
    assert all(e["confidence"] == "high" for e in body["events"])


def test_recommendations_replay():
    body = client.get("/api/recommendations", params={"as_of": "2024-06-01"}).json()
    rules = {r["rule_id"] for r in body["recommendations"]}
    assert "sustained_overhead_increase" in rules


def test_bad_horizon_rejected():
    assert client.get("/api/forecast/test", params={"horizon": 6}).status_code == 400