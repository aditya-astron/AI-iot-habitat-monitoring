"""Small FastAPI service around the habitat pipeline."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from habitat_monitor import __version__
from habitat_monitor.detect import default_ensemble
from habitat_monitor.genai import generate_report
from habitat_monitor.optimize import recommend_interval
from habitat_monitor.pipeline import Pipeline
from habitat_monitor.schema import Detection, HabitatReport, ReadingFrame, SamplingDecision

app = FastAPI(
    title="Habitat Monitor",
    version=__version__,
    description="Ingest habitat sensor frames, detect anomalies, tune sampling.",
)

_pipeline = Pipeline()
_fitted = False


class FitRequest(BaseModel):
    frames: list[ReadingFrame] = Field(min_length=16)


class DetectRequest(BaseModel):
    frames: list[ReadingFrame] = Field(min_length=1)


class OptimizeRequest(BaseModel):
    anomaly_rate: float = Field(ge=0.0, le=1.0)
    battery_frac: float | None = Field(default=None, ge=0.0, le=1.0)


class ReportRequest(BaseModel):
    detections: list[Detection]
    sampling: SamplingDecision | None = None


class DetectResponse(BaseModel):
    detections: list[Detection]
    anomaly_count: int
    anomaly_rate: float


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "version": __version__, "fitted": _fitted}


@app.post("/fit")
def fit(req: FitRequest) -> dict[str, Any]:
    global _pipeline, _fitted
    _pipeline = Pipeline(ensemble=default_ensemble())
    _pipeline.fit(req.frames)
    _fitted = True
    return {"fitted": True, "n_train": len(req.frames)}


@app.post("/detect", response_model=DetectResponse)
def detect(req: DetectRequest) -> DetectResponse:
    if not _fitted:
        raise HTTPException(status_code=409, detail="Call POST /fit before /detect")
    result = _pipeline.run(req.frames)
    return DetectResponse(
        detections=result.detections,
        anomaly_count=sum(1 for d in result.detections if d.is_anomaly),
        anomaly_rate=result.anomaly_rate,
    )


@app.post("/optimize", response_model=SamplingDecision)
def optimize(req: OptimizeRequest) -> SamplingDecision:
    return recommend_interval(req.anomaly_rate, battery_frac=req.battery_frac)


@app.post("/report", response_model=HabitatReport)
def report(req: ReportRequest) -> HabitatReport:
    return generate_report(req.detections, req.sampling)
