from __future__ import annotations

from habitat_monitor.detect import (
    IsolationForestDetector,
    StuckSensorDetector,
    ThresholdDetector,
    ZScoreDetector,
    default_ensemble,
)
from habitat_monitor.generate import generate_labeled_stream
from habitat_monitor.schema import ReadingFrame, Sensors


def _frame(seq: int, **overrides: float) -> ReadingFrame:
    values = {
        "temperature_c": 24.0,
        "humidity_pct": 55.0,
        "soil_moisture_pct": 40.0,
        "co2_ppm": 420.0,
        "light_lux": 200.0,
        "motion": 0,
    }
    values.update(overrides)
    return ReadingFrame(
        device_id="hab-01",
        seq=seq,
        ts_ms=seq * 1000,
        sensors=Sensors(**values),
    )


def test_threshold_flags_overheat() -> None:
    det = ThresholdDetector()
    det.fit([])
    ok, risk, reason = det.score(_frame(0, temperature_c=45.0))
    assert ok is True
    assert risk > 0.5
    assert "temperature_c" in reason


def test_threshold_accepts_nominal() -> None:
    det = ThresholdDetector()
    det.fit([])
    ok, risk, _ = det.score(_frame(0))
    assert ok is False
    assert risk == 0.0


def test_zscore_flags_co2_event_after_clean_fit() -> None:
    train = [r.frame for r in generate_labeled_stream(120, seed=3, inject=False)]
    model = ZScoreDetector(z_gate=4.0)
    model.fit(train)
    spike = train[-1].model_copy(
        update={"sensors": train[-1].sensors.model_copy(update={"co2_ppm": 1600.0})}
    )
    ok, risk, reason = model.score(spike)
    assert ok is True
    assert risk > 0.5
    assert "co2_ppm" in reason


def test_iforest_scores_after_fit() -> None:
    train = [r.frame for r in generate_labeled_stream(80, seed=3, inject=False)]
    model = IsolationForestDetector(random_state=3)
    model.fit(train)
    ok, risk, _reason = model.score(train[-1])
    assert ok in (True, False)
    assert risk >= 0.0


def test_stuck_sensor_flags_flatline() -> None:
    det = StuckSensorDetector(window=8)
    frames = [_frame(i, temperature_c=24.00, humidity_pct=50.00) for i in range(12)]
    det.fit(frames[:8])
    flagged = False
    for frame in frames[8:]:
        ok, _risk, reason = det.score(frame)
        if ok:
            flagged = True
            assert "stuck" in reason
    assert flagged


def test_ensemble_detect_many_length() -> None:
    train = [r.frame for r in generate_labeled_stream(80, seed=1, inject=False)]
    test = [r.frame for r in generate_labeled_stream(40, seed=2, inject=True)]
    ens = default_ensemble(random_state=1)
    ens.fit(train)
    detections = ens.detect_many(test)
    assert len(detections) == len(test)
    assert any(d.is_anomaly for d in detections)
