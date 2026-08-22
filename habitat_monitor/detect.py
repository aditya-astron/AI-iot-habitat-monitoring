"""Anomaly detectors for habitat sensor streams.

Detectors are fit on a clean (or mostly clean) training window and then
score new frames. Isolation Forest is trained on raw sensor channels plus
VPD. A range/threshold detector encodes operational habitat bounds. A
stuck-sensor detector flags near-zero variance on channels that should
move.

None of these claim a fixed accuracy number. Use ``habitat_monitor.evaluate``
to measure precision/recall/F1 on a labeled split.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from habitat_monitor.features import RAW_FEATURE_KEYS, reading_features
from habitat_monitor.schema import Detection, ReadingFrame, Sensors

# Operational envelopes for an enclosed habitat / analog-mission bay.
# Values outside are physically implausible or immediately actionable.
DEFAULT_BOUNDS: dict[str, tuple[float, float]] = {
    "temperature_c": (12.0, 38.0),
    "humidity_pct": (15.0, 92.0),
    "soil_moisture_pct": (8.0, 88.0),
    "co2_ppm": (280.0, 1800.0),
    "light_lux": (0.0, 80_000.0),
    "motion": (0.0, 25.0),
}

VECTOR_KEYS: tuple[str, ...] = RAW_FEATURE_KEYS + ("vpd_kpa",)


def _vector(frame: ReadingFrame, previous: ReadingFrame | None = None) -> np.ndarray:
    feats = reading_features(frame, previous)
    return np.array([feats[k] for k in VECTOR_KEYS], dtype=float)


class Detector(Protocol):
    name: str

    def fit(self, frames: list[ReadingFrame]) -> None: ...

    def score(self, frame: ReadingFrame, previous: ReadingFrame | None = None) -> tuple[bool, float, str]:
        """Return (is_anomaly, risk_score, reason). risk_score is in [0, 1+]."""


@dataclass
class ThresholdDetector:
    """Flag a reading if any raw channel leaves the operational envelope."""

    name: str = "threshold"
    bounds: dict[str, tuple[float, float]] = field(default_factory=lambda: dict(DEFAULT_BOUNDS))

    def fit(self, frames: list[ReadingFrame]) -> None:
        # Bounds are physical, not estimated. Fit is a no-op so the interface matches.
        return None

    def score(self, frame: ReadingFrame, previous: ReadingFrame | None = None) -> tuple[bool, float, str]:
        reasons: list[str] = []
        worst = 0.0
        for key, (lo, hi) in self.bounds.items():
            value = float(getattr(frame.sensors, key))
            if value < lo:
                over = (lo - value) / max(abs(lo), 1.0)
                reasons.append(f"{key}={value:.2f} below {lo}")
                worst = max(worst, over)
            elif value > hi:
                over = (value - hi) / max(abs(hi), 1.0)
                reasons.append(f"{key}={value:.2f} above {hi}")
                worst = max(worst, over)
        if not reasons:
            return False, 0.0, ""
        return True, min(1.5, 0.55 + worst), "; ".join(reasons)


@dataclass
class IsolationForestDetector:
    """Multivariate detector trained on a clean window.

    The decision threshold is the ``contamination`` quantile of training
    scores (more negative = more anomalous in sklearn). This is measured,
    not hardcoded as an accuracy claim.
    """

    name: str = "isolation_forest"
    contamination: float = 0.03
    random_state: int = 42
    n_estimators: int = 200
    _model: IsolationForest | None = None
    _scaler: StandardScaler | None = None
    _threshold: float = 0.0

    def fit(self, frames: list[ReadingFrame]) -> None:
        if len(frames) < 16:
            raise ValueError("IsolationForestDetector needs at least 16 training frames")
        matrix = np.vstack([_vector(f) for f in frames])
        self._scaler = StandardScaler()
        scaled = self._scaler.fit_transform(matrix)
        self._model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
        )
        self._model.fit(scaled)
        scores = self._model.decision_function(scaled)
        self._threshold = float(np.quantile(scores, self.contamination))

    def score(self, frame: ReadingFrame, previous: ReadingFrame | None = None) -> tuple[bool, float, str]:
        if self._model is None or self._scaler is None:
            raise RuntimeError("IsolationForestDetector.fit() must be called first")
        value = self._scaler.transform(_vector(frame, previous).reshape(1, -1))
        raw = float(self._model.decision_function(value)[0])
        # Map more-negative scores to a positive risk in roughly [0, 1].
        risk = max(0.0, self._threshold - raw)
        # Scale so a score at 2x the training tail maps near 1.
        scale = max(abs(self._threshold), 0.05)
        risk = risk / scale
        is_anom = raw < self._threshold
        reason = f"iforest_score={raw:.4f} threshold={self._threshold:.4f}" if is_anom else ""
        return is_anom, risk, reason


@dataclass
class StuckSensorDetector:
    """Flag channels whose recent variance collapsed (stuck / frozen ADC)."""

    name: str = "stuck_sensor"
    window: int = 8
    min_variance: dict[str, float] = field(
        default_factory=lambda: {
            "temperature_c": 0.02,
            "humidity_pct": 0.15,
            "soil_moisture_pct": 0.08,
        }
    )
    _history: dict[str, list[float]] = field(default_factory=dict)

    def fit(self, frames: list[ReadingFrame]) -> None:
        self._history = {key: [] for key in self.min_variance}
        for frame in frames[-self.window :]:
            self._push(frame.sensors)

    def _push(self, sensors: Sensors) -> None:
        for key in self.min_variance:
            bucket = self._history.setdefault(key, [])
            bucket.append(float(getattr(sensors, key)))
            if len(bucket) > self.window:
                del bucket[0 : len(bucket) - self.window]

    def score(self, frame: ReadingFrame, previous: ReadingFrame | None = None) -> tuple[bool, float, str]:
        self._push(frame.sensors)
        stuck: list[str] = []
        for key, floor in self.min_variance.items():
            series = self._history.get(key, [])
            if len(series) < self.window:
                continue
            var = float(np.var(series))
            if var < floor:
                stuck.append(f"{key} var={var:.5f}<{floor}")
        if not stuck:
            return False, 0.0, ""
        return True, 0.7, "stuck: " + "; ".join(stuck)


@dataclass
class ZScoreDetector:
    """Flag a channel whose |z| versus the training window exceeds ``z_gate``.

    Isolation Forest is weak on single-axis extremes (a 100+ sigma CO2 spike
    can still score as an inlier). This detector covers that case.
    """

    name: str = "zscore"
    z_gate: float = 4.0
    _mean: dict[str, float] | None = None
    _std: dict[str, float] | None = None

    def fit(self, frames: list[ReadingFrame]) -> None:
        if len(frames) < 8:
            raise ValueError("ZScoreDetector needs at least 8 training frames")
        keys = RAW_FEATURE_KEYS
        cols = {k: np.array([float(getattr(f.sensors, k)) for f in frames], dtype=float) for k in keys}
        self._mean = {k: float(np.mean(v)) for k, v in cols.items()}
        self._std = {k: float(max(np.std(v, ddof=1), 1e-6)) for k, v in cols.items()}

    def score(self, frame: ReadingFrame, previous: ReadingFrame | None = None) -> tuple[bool, float, str]:
        if self._mean is None or self._std is None:
            raise RuntimeError("ZScoreDetector.fit() must be called first")
        hits: list[str] = []
        worst = 0.0
        for key in RAW_FEATURE_KEYS:
            z = abs((float(getattr(frame.sensors, key)) - self._mean[key]) / self._std[key])
            if z >= self.z_gate:
                hits.append(f"{key} z={z:.1f}")
                worst = max(worst, z)
        if not hits:
            return False, 0.0, ""
        risk = min(1.8, 0.4 + 0.1 * worst)
        return True, risk, "; ".join(hits)


@dataclass
class EnsembleDetector:
    """OR-ensemble: a frame is anomalous if any member flags it.

    Risk is the max member risk. ``fit`` is forwarded to every member.
    """

    members: list[Detector]
    name: str = "ensemble"

    def fit(self, frames: list[ReadingFrame]) -> None:
        for member in self.members:
            member.fit(frames)

    def score(self, frame: ReadingFrame, previous: ReadingFrame | None = None) -> tuple[bool, float, list[str], list[str]]:
        flagged: list[str] = []
        reasons: list[str] = []
        risk = 0.0
        for member in self.members:
            is_anom, member_risk, reason = member.score(frame, previous)
            risk = max(risk, member_risk)
            if is_anom:
                flagged.append(member.name)
                if reason:
                    reasons.append(f"{member.name}: {reason}")
        return bool(flagged), risk, flagged, reasons

    def detect_one(self, frame: ReadingFrame, previous: ReadingFrame | None = None) -> Detection:
        is_anom, risk, flagged, reasons = self.score(frame, previous)
        return Detection(
            seq=frame.seq,
            ts_ms=frame.ts_ms,
            device_id=frame.device_id,
            is_anomaly=is_anom,
            risk_score=float(risk),
            detectors=flagged,
            reasons=reasons,
            sensors=frame.sensors,
        )

    def detect_many(self, frames: list[ReadingFrame]) -> list[Detection]:
        out: list[Detection] = []
        previous: ReadingFrame | None = None
        for frame in frames:
            out.append(self.detect_one(frame, previous))
            previous = frame
        return out


def default_ensemble(random_state: int = 42) -> EnsembleDetector:
    return EnsembleDetector(
        members=[
            ThresholdDetector(),
            ZScoreDetector(),
            IsolationForestDetector(random_state=random_state),
            StuckSensorDetector(),
        ]
    )
