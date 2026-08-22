"""Synthetic habitat streams with optional ground-truth anomaly labels.

The generator models a 24-hour analog-mission / enclosed-habitat cycle
(temperature and light follow a sine day, humidity anti-correlates with
temperature, soil moisture drifts, CO2 has a weak occupancy bump).
Anomalies are injected as contiguous events so evaluation can measure
point-level and event-level metrics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from habitat_monitor.schema import LabeledRecord, ReadingFrame, Sensors

SECONDS_PER_DAY = 86_400
DEFAULT_INTERVAL_S = 30


@dataclass(frozen=True)
class EventSpec:
    name: str
    start: int
    length: int


def _base_state(minute_of_day: float, rng: np.random.Generator) -> dict[str, float]:
    phase = 2.0 * math.pi * (minute_of_day / 1440.0)
    # Peak temperature mid-afternoon (phase shift).
    temperature = 24.0 + 5.5 * math.sin(phase - 0.6) + rng.normal(0.0, 0.35)
    humidity = 62.0 - 8.0 * math.sin(phase - 0.6) + rng.normal(0.0, 1.2)
    humidity = float(np.clip(humidity, 15.0, 95.0))
    soil = 42.0 + 4.0 * math.sin(phase / 2.0) + rng.normal(0.0, 0.8)
    soil = float(np.clip(soil, 12.0, 85.0))
    daylight = max(0.0, math.sin(phase))
    light = 50.0 + 850.0 * daylight + rng.normal(0.0, 12.0)
    light = max(0.0, light)
    occupancy = 0.55 + 0.45 * max(0.0, math.sin(phase))
    co2 = 410.0 + 55.0 * occupancy + rng.normal(0.0, 8.0)
    motion = int(rng.poisson(0.4 + 1.6 * occupancy))
    return {
        "temperature_c": float(temperature),
        "humidity_pct": humidity,
        "soil_moisture_pct": soil,
        "co2_ppm": float(co2),
        "light_lux": float(light),
        "motion": motion,
    }


def _apply_event(
    values: dict[str, float], name: str, offset: int, length: int, rng: np.random.Generator
) -> dict[str, float]:
    out = dict(values)
    progress = offset / max(length - 1, 1)
    if name == "heat_spike":
        out["temperature_c"] += 9.0 + 4.0 * progress
        out["humidity_pct"] = max(8.0, out["humidity_pct"] - 12.0)
    elif name == "co2_event":
        out["co2_ppm"] += 380.0 + 80.0 * progress
    elif name == "dry_out":
        out["soil_moisture_pct"] = max(3.0, out["soil_moisture_pct"] - 22.0 - 8.0 * progress)
    elif name == "sensor_stuck":
        out["temperature_c"] = 24.00
        out["humidity_pct"] = 50.00
    elif name == "night_light":
        out["light_lux"] = 1400.0 + float(rng.normal(0.0, 20.0))
    else:
        raise ValueError(f"unknown event type: {name}")
    return out


def _choose_events(n: int, rng: np.random.Generator) -> list[EventSpec]:
    """Place non-overlapping events covering roughly 6–10% of samples.

    When the window is long enough, include one of each kind so every
    detector has something to catch on the default evaluation split.
    """
    kinds = ("heat_spike", "co2_event", "dry_out", "sensor_stuck", "night_light")
    events: list[EventSpec] = []
    occupied: set[int] = set()

    def _place(kind: str) -> bool:
        length = int(rng.integers(10, 15)) if kind == "sensor_stuck" else int(rng.integers(4, 10))
        lo = int(n * 0.12)
        hi = n - length - 5
        if hi <= lo:
            return False
        for _ in range(24):
            start = int(rng.integers(lo, hi))
            span = set(range(start, start + length))
            if span & occupied:
                continue
            occupied.update(span)
            events.append(EventSpec(kind, start, length))
            return True
        return False

    for kind in kinds:
        _place(kind)
    extras = max(0, max(3, n // 80) - len(events))
    attempts = 0
    while extras > 0 and attempts < 40:
        attempts += 1
        if _place(str(rng.choice(kinds))):
            extras -= 1
    return events


def generate_labeled_stream(
    n: int = 400,
    *,
    seed: int = 42,
    interval_s: int = DEFAULT_INTERVAL_S,
    device_id: str = "hab-01",
    inject: bool = True,
) -> list[LabeledRecord]:
    """Return ``n`` labeled readings. Training sets should pass ``inject=False``."""
    if n < 16:
        raise ValueError("n must be >= 16")
    rng = np.random.default_rng(seed)
    events = _choose_events(n, rng) if inject else []
    event_at: dict[int, EventSpec] = {}
    for event in events:
        for offset in range(event.length):
            event_at[event.start + offset] = event

    records: list[LabeledRecord] = []
    start_ms = 1_700_000_000_000
    for i in range(n):
        minute_of_day = ((i * interval_s) / 60.0) % 1440.0
        values = _base_state(minute_of_day, rng)
        event = event_at.get(i)
        label: int = 0
        event_name: str | None = None
        if event is not None:
            values = _apply_event(values, event.name, i - event.start, event.length, rng)
            label = 1
            event_name = event.name
        frame = ReadingFrame(
            device_id=device_id,
            seq=i,
            ts_ms=start_ms + i * interval_s * 1000,
            sensors=Sensors(**values),
            status="ok",
        )
        records.append(LabeledRecord(frame=frame, label=label, event=event_name))
    return records


def generate_split(
    n_train: int = 360,
    n_test: int = 240,
    *,
    seed: int = 42,
) -> tuple[list[LabeledRecord], list[LabeledRecord]]:
    """Clean training window, held-out test window with injected events."""
    train = generate_labeled_stream(n_train, seed=seed, inject=False)
    test = generate_labeled_stream(n_test, seed=seed + 1, inject=True)
    # Re-number test seq so a concatenated replay is monotonic if desired.
    shifted: list[LabeledRecord] = []
    for rec in test:
        frame = rec.frame.model_copy(update={"seq": rec.frame.seq + n_train})
        shifted.append(LabeledRecord(frame=frame, label=rec.label, event=rec.event))
    return train, shifted


def records_to_jsonl(records: list[LabeledRecord]) -> str:
    return "\n".join(rec.model_dump_json() for rec in records) + "\n"
