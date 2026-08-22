"""Feature construction for habitat sensor streams."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from habitat_monitor.schema import SENSOR_KEYS, ReadingFrame, Sensors

RAW_FEATURE_KEYS: tuple[str, ...] = SENSOR_KEYS
DERIVED_KEYS: tuple[str, ...] = ("vpd_kpa", "temp_delta", "co2_delta")
ALL_FEATURE_KEYS: tuple[str, ...] = RAW_FEATURE_KEYS + DERIVED_KEYS


def vapor_pressure_deficit_kpa(temp_c: float, humidity_pct: float) -> float:
    """Tetens approximation of VPD in kPa. Used in greenhouse / habitat control."""
    saturation = 0.6108 * math.exp((17.27 * temp_c) / (temp_c + 237.3))
    actual = saturation * (max(0.0, min(humidity_pct, 100.0)) / 100.0)
    return max(0.0, saturation - actual)


def sensors_to_vector(sensors: Sensors) -> dict[str, float]:
    return {key: float(getattr(sensors, key)) for key in RAW_FEATURE_KEYS}


def reading_features(frame: ReadingFrame, previous: ReadingFrame | None = None) -> dict[str, float]:
    """Raw sensors plus VPD and first differences."""
    values = sensors_to_vector(frame.sensors)
    values["vpd_kpa"] = vapor_pressure_deficit_kpa(
        frame.sensors.temperature_c, frame.sensors.humidity_pct
    )
    if previous is None:
        values["temp_delta"] = 0.0
        values["co2_delta"] = 0.0
    else:
        values["temp_delta"] = frame.sensors.temperature_c - previous.sensors.temperature_c
        values["co2_delta"] = frame.sensors.co2_ppm - previous.sensors.co2_ppm
    return values


@dataclass
class RollingStats:
    """Welford mean/variance over a fixed window of scalar values."""

    window: int
    _values: deque[float]

    def __init__(self, window: int = 32) -> None:
        if window < 2:
            raise ValueError("window must be >= 2")
        self.window = window
        self._values = deque(maxlen=window)

    def update(self, value: float) -> None:
        self._values.append(float(value))

    @property
    def count(self) -> int:
        return len(self._values)

    @property
    def mean(self) -> float:
        if not self._values:
            return 0.0
        return sum(self._values) / len(self._values)

    @property
    def variance(self) -> float:
        n = len(self._values)
        if n < 2:
            return 0.0
        mean = self.mean
        return sum((x - mean) ** 2 for x in self._values) / (n - 1)

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)

    def zscore(self, value: float) -> float:
        std = self.std
        if std < 1e-9:
            return 0.0
        return (value - self.mean) / std


class FeatureWindow:
    """Maintains per-channel rolling stats for streaming detectors."""

    def __init__(self, window: int = 32) -> None:
        self.window = window
        self._stats = {key: RollingStats(window) for key in RAW_FEATURE_KEYS}
        self._previous: ReadingFrame | None = None

    def push(self, frame: ReadingFrame) -> dict[str, float]:
        feats = reading_features(frame, self._previous)
        for key in RAW_FEATURE_KEYS:
            self._stats[key].update(feats[key])
        self._previous = frame
        return feats

    def channel_variance(self, key: str) -> float:
        return self._stats[key].variance

    def channel_zscore(self, key: str, value: float) -> float:
        return self._stats[key].zscore(value)

    def ready(self, min_count: int = 8) -> bool:
        first = self._stats[RAW_FEATURE_KEYS[0]]
        return first.count >= min_count
