from __future__ import annotations

from fastapi.testclient import TestClient

from habitat_monitor.api import app
from habitat_monitor.generate import generate_labeled_stream


def test_health() -> None:
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_detect_requires_fit() -> None:
    client = TestClient(app)
    frames = [r.frame.model_dump() for r in generate_labeled_stream(20, seed=2, inject=False)]
    resp = client.post("/detect", json={"frames": frames})
    assert resp.status_code == 409


def test_fit_detect_optimize_report() -> None:
    client = TestClient(app)
    train = [r.frame.model_dump() for r in generate_labeled_stream(80, seed=4, inject=False)]
    test = [r.frame.model_dump() for r in generate_labeled_stream(30, seed=9, inject=True)]
    fit = client.post("/fit", json={"frames": train})
    assert fit.status_code == 200
    detected = client.post("/detect", json={"frames": test})
    assert detected.status_code == 200
    body = detected.json()
    assert body["anomaly_count"] == sum(1 for d in body["detections"] if d["is_anomaly"])
    opt = client.post("/optimize", json={"anomaly_rate": 0.25, "battery_frac": 0.8})
    assert opt.status_code == 200
    assert opt.json()["interval_ms"] > 0
    report = client.post(
        "/report",
        json={"detections": body["detections"], "sampling": opt.json()},
    )
    assert report.status_code == 200
    assert "executive_summary" in report.json()
