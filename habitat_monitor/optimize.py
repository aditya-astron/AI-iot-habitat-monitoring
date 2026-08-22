"""Sampling-interval and alert-policy optimization.

This module does not claim an accuracy or time-saved percentage. It
implements two deterministic policies that are tested against synthetic
streams:

1. Adaptive sampling: shorten the interval when recent anomaly rate is
   high; lengthen it when the rate is low or battery is scarce.
2. Alert policy: emit an alert only when risk exceeds a raise threshold,
   then require risk to fall below a clear threshold (hysteresis) and
   wait out a cooldown before repeating.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from habitat_monitor.schema import Alert, Detection, SamplingDecision


@dataclass(frozen=True)
class SamplingPolicy:
    min_interval_ms: int = 5_000
    max_interval_ms: int = 300_000
    nominal_interval_ms: int = 30_000
    high_rate: float = 0.12
    low_rate: float = 0.02
    battery_stretch_below: float = 0.25
    battery_stretch: float = 1.6


def recommend_interval(
    anomaly_rate: float,
    battery_frac: float | None = None,
    policy: SamplingPolicy = SamplingPolicy(),
) -> SamplingDecision:
    """Map anomaly rate (and optional battery) to a sampling interval.

    Linear interpolation between ``max`` (quiet) and ``min`` (busy).
    Low battery stretches the interval so the node lasts longer.
    """
    if anomaly_rate < 0.0 or anomaly_rate > 1.0:
        raise ValueError("anomaly_rate must be in [0, 1]")
    if battery_frac is not None and (battery_frac < 0.0 or battery_frac > 1.0):
        raise ValueError("battery_frac must be in [0, 1]")

    span = policy.high_rate - policy.low_rate
    if span <= 0:
        raise ValueError("high_rate must be greater than low_rate")
    t = (anomaly_rate - policy.low_rate) / span
    t = min(1.0, max(0.0, t))
    interval = policy.max_interval_ms + t * (policy.min_interval_ms - policy.max_interval_ms)

    rationale_parts = [f"anomaly_rate={anomaly_rate:.3f} -> mix={t:.2f}"]
    if battery_frac is not None and battery_frac < policy.battery_stretch_below:
        interval *= policy.battery_stretch
        rationale_parts.append(
            f"battery={battery_frac:.2f}<{policy.battery_stretch_below} stretch={policy.battery_stretch}"
        )

    interval = int(round(interval))
    interval = min(policy.max_interval_ms, max(policy.min_interval_ms, interval))
    rationale_parts.append(f"interval_ms={interval}")
    return SamplingDecision(
        interval_ms=interval,
        anomaly_rate=anomaly_rate,
        battery_frac=battery_frac,
        rationale="; ".join(rationale_parts),
    )


def anomaly_rate(detections: list[Detection], window: int | None = None) -> float:
    if not detections:
        return 0.0
    subset = detections if window is None else detections[-window:]
    if not subset:
        return 0.0
    return sum(1 for d in subset if d.is_anomaly) / len(subset)


@dataclass
class AlertPolicy:
    raise_threshold: float = 0.45
    clear_threshold: float = 0.20
    cooldown_frames: int = 6
    critical_risk: float = 0.9


@dataclass
class AlertEngine:
    """Stateful hysteresis + cooldown filter over detections."""

    policy: AlertPolicy = field(default_factory=AlertPolicy)
    _latched: bool = False
    _since_alert: int = 10**9
    _device_id: str = "hab-01"

    def _severity(self, risk: float) -> str:
        if risk >= self.policy.critical_risk:
            return "critical"
        if risk >= self.policy.raise_threshold:
            return "warning"
        return "info"

    def process(self, detection: Detection) -> Alert:
        self._device_id = detection.device_id
        self._since_alert += 1
        risk = detection.risk_score
        suppressed = False
        suppress_reason: str | None = None

        if detection.is_anomaly and risk >= self.policy.raise_threshold:
            if self._latched and self._since_alert < self.policy.cooldown_frames:
                suppressed = True
                suppress_reason = f"cooldown {self._since_alert}/{self.policy.cooldown_frames}"
            else:
                self._latched = True
                self._since_alert = 0
        elif self._latched and risk <= self.policy.clear_threshold:
            self._latched = False

        return Alert(
            seq=detection.seq,
            device_id=detection.device_id,
            severity=self._severity(risk) if detection.is_anomaly else "info",
            risk_score=risk,
            reasons=list(detection.reasons),
            suppressed=suppressed,
            suppress_reason=suppress_reason,
        )

    def process_many(self, detections: list[Detection]) -> list[Alert]:
        return [self.process(d) for d in detections]


def rank_response(alerts: list[Alert], budget: int) -> list[Alert]:
    """Pick up to ``budget`` unsuppressed alerts with the highest risk."""
    if budget < 0:
        raise ValueError("budget must be >= 0")
    active = [a for a in alerts if not a.suppressed and a.reasons]
    ranked = sorted(active, key=lambda a: a.risk_score, reverse=True)
    return ranked[:budget]
