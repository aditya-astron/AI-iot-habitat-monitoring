from __future__ import annotations

import pytest

from habitat_monitor.pipeline import Pipeline


def test_pipeline_requires_fit() -> None:
    pipe = Pipeline()
    with pytest.raises(RuntimeError):
        pipe.run([])


def test_pipeline_run(labeled_split) -> None:
    train, test = labeled_split
    pipe = Pipeline()
    pipe.fit([r.frame for r in train])
    result = pipe.run([r.frame for r in test], battery_frac=0.5)
    assert len(result.detections) == len(test)
    assert 0.0 <= result.anomaly_rate <= 1.0
    assert result.sampling.interval_ms >= 5000
    assert len(result.alerts) == len(result.detections)
