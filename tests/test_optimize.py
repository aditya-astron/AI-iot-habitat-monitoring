from __future__ import annotations

import pytest

from habitat_monitor.optimize import (
    AlertEngine,
    AlertPolicy,
    SamplingPolicy,
    anomaly_rate,
    rank_response,
    recommend_interval,
)
from habitat_monitor.schema import Detection, Sensors


def _det(seq: int, anomaly: bool, risk: float) -> Detection:
    return Detection(
        seq=seq,
        ts_ms=seq * 1000,
        device_id="hab-01",
        is_anomaly=anomaly,
        risk_score=risk,
        detectors=["threshold"] if anomaly else [],
        reasons=["hot"] if anomaly else [],
        sensors=Sensors(
            temperature_c=24.0,
            humidity_pct=50.0,
            soil_moisture_pct=40.0,
            co2_ppm=400.0,
            light_lux=10.0,
            motion=0,
        ),
    )


def test_high_anomaly_rate_shortens_interval() -> None:
    quiet = recommend_interval(0.0)
    busy = recommend_interval(0.5)
    assert busy.interval_ms < quiet.interval_ms
    assert busy.interval_ms >= SamplingPolicy().min_interval_ms


def test_low_battery_lengthens_interval() -> None:
    normal = recommend_interval(0.2, battery_frac=0.9)
    low = recommend_interval(0.2, battery_frac=0.1)
    assert low.interval_ms > normal.interval_ms


def test_interval_rejects_bad_rate() -> None:
    with pytest.raises(ValueError):
        recommend_interval(1.4)


def test_anomaly_rate() -> None:
    dets = [_det(i, i % 2 == 0, 0.6) for i in range(10)]
    assert anomaly_rate(dets) == 0.5


def test_alert_hysteresis_and_cooldown() -> None:
    engine = AlertEngine(AlertPolicy(raise_threshold=0.4, clear_threshold=0.2, cooldown_frames=3))
    first = engine.process(_det(0, True, 0.8))
    assert first.suppressed is False
    second = engine.process(_det(1, True, 0.8))
    assert second.suppressed is True
    assert second.suppress_reason and "cooldown" in second.suppress_reason
    # Drop below clear so latch releases.
    engine.process(_det(2, False, 0.05))
    engine.process(_det(3, False, 0.05))
    engine.process(_det(4, False, 0.05))
    later = engine.process(_det(5, True, 0.85))
    assert later.suppressed is False


def test_rank_response_respects_budget() -> None:
    from habitat_monitor.schema import Alert

    alerts = [
        Alert(seq=i, device_id="hab-01", severity="warning", risk_score=0.5 + i * 0.1,
              reasons=["hot"], suppressed=False)
        for i in range(5)
    ]
    top = rank_response(alerts, budget=2)
    assert len(top) == 2
    assert top[0].risk_score >= top[1].risk_score
