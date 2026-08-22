from __future__ import annotations

from habitat_monitor.features import (
    FeatureWindow,
    reading_features,
    vapor_pressure_deficit_kpa,
)
from habitat_monitor.schema import ReadingFrame, Sensors


def _frame(seq: int, temp: float, rh: float = 50.0) -> ReadingFrame:
    return ReadingFrame(
        device_id="hab-01",
        seq=seq,
        ts_ms=seq * 1000,
        sensors=Sensors(
            temperature_c=temp,
            humidity_pct=rh,
            soil_moisture_pct=40.0,
            co2_ppm=420.0,
            light_lux=100.0,
            motion=0,
        ),
    )


def test_vpd_zero_at_saturation() -> None:
    assert vapor_pressure_deficit_kpa(25.0, 100.0) == 0.0


def test_vpd_positive_when_dry() -> None:
    dry = vapor_pressure_deficit_kpa(30.0, 20.0)
    wet = vapor_pressure_deficit_kpa(30.0, 80.0)
    assert dry > wet > 0.0


def test_deltas_use_previous_frame() -> None:
    a = _frame(0, 20.0)
    b = _frame(1, 22.5)
    feats = reading_features(b, a)
    assert feats["temp_delta"] == 2.5
    assert feats["co2_delta"] == 0.0


def test_feature_window_zscore_flags_outlier() -> None:
    window = FeatureWindow(window=16)
    for i in range(16):
        window.push(_frame(i, 24.0 + 0.05 * i))
    z = window.channel_zscore("temperature_c", 40.0)
    assert z > 5.0
