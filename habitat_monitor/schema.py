"""Pydantic models for frames, detections, alerts, and reports."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

SENSOR_KEYS: tuple[str, ...] = (
    "temperature_c",
    "humidity_pct",
    "soil_moisture_pct",
    "co2_ppm",
    "light_lux",
    "motion",
)


class Sensors(BaseModel):
    """Physical readings. Names match firmware JSON exactly."""

    temperature_c: float
    humidity_pct: float = Field(ge=0.0, le=100.0)
    soil_moisture_pct: float = Field(ge=0.0, le=100.0)
    co2_ppm: float = Field(ge=0.0)
    light_lux: float = Field(ge=0.0)
    motion: int = Field(ge=0)

    @field_validator("motion", mode="before")
    @classmethod
    def _coerce_motion(cls, value: object) -> int:
        return int(value)


class ReadingFrame(BaseModel):
    v: Literal[1] = 1
    type: Literal["reading"] = "reading"
    device_id: str = Field(min_length=1, max_length=64)
    seq: int = Field(ge=0)
    ts_ms: int = Field(ge=0)
    sensors: Sensors
    status: Literal["ok", "degraded", "fault"] = "ok"


class CommandFrame(BaseModel):
    v: Literal[1] = 1
    type: Literal["cmd"] = "cmd"
    cmd: Literal["ping", "set_interval"]
    interval_ms: int | None = Field(default=None, ge=1000, le=3_600_000)


class AckFrame(BaseModel):
    v: Literal[1] = 1
    type: Literal["ack"] = "ack"
    device_id: str
    ack: Literal["pong", "set_interval"]
    uptime_ms: int | None = None
    interval_ms: int | None = None


class Detection(BaseModel):
    seq: int
    ts_ms: int
    device_id: str
    is_anomaly: bool
    risk_score: float
    detectors: list[str]
    reasons: list[str]
    sensors: Sensors


class Alert(BaseModel):
    seq: int
    device_id: str
    severity: Literal["info", "warning", "critical"]
    risk_score: float
    reasons: list[str]
    suppressed: bool = False
    suppress_reason: str | None = None


class SamplingDecision(BaseModel):
    interval_ms: int
    anomaly_rate: float
    battery_frac: float | None = None
    rationale: str


class HabitatReport(BaseModel):
    executive_summary: str
    health_score: float = Field(ge=0.0, le=100.0)
    key_insights: list[str]
    recommended_actions: list[str]
    confidence: float = Field(ge=0.0, le=1.0)
    anomaly_count: int = Field(ge=0)
    recommended_interval_ms: int = Field(ge=1000)


class LabeledRecord(BaseModel):
    """Offline evaluation wrapper. Firmware never emits this."""

    frame: ReadingFrame
    label: Literal[0, 1]
    event: str | None = None
