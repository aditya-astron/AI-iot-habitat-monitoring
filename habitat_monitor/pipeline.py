"""End-to-end path: frames -> detect -> sample/alert decisions."""

from __future__ import annotations

from dataclasses import dataclass, field

from habitat_monitor.detect import EnsembleDetector, default_ensemble
from habitat_monitor.optimize import (
    Alert,
    AlertEngine,
    SamplingDecision,
    anomaly_rate,
    recommend_interval,
)
from habitat_monitor.schema import Detection, ReadingFrame


@dataclass
class PipelineResult:
    detections: list[Detection]
    alerts: list[Alert]
    sampling: SamplingDecision
    anomaly_rate: float


@dataclass
class Pipeline:
    ensemble: EnsembleDetector = field(default_factory=default_ensemble)
    alerts: AlertEngine = field(default_factory=AlertEngine)
    fitted: bool = False

    def fit(self, frames: list[ReadingFrame]) -> None:
        self.ensemble.fit(frames)
        self.fitted = True

    def run(self, frames: list[ReadingFrame], battery_frac: float | None = None) -> PipelineResult:
        if not self.fitted:
            raise RuntimeError("Pipeline.fit() must be called before run()")
        detections = self.ensemble.detect_many(frames)
        alerts = self.alerts.process_many(detections)
        rate = anomaly_rate(detections)
        sampling = recommend_interval(rate, battery_frac=battery_frac)
        return PipelineResult(
            detections=detections,
            alerts=alerts,
            sampling=sampling,
            anomaly_rate=rate,
        )
