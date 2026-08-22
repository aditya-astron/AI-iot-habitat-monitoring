from __future__ import annotations

from habitat_monitor.detect import ThresholdDetector
from habitat_monitor.evaluate import evaluate_detector, evaluate_ensemble, score_predictions
from habitat_monitor.generate import generate_split


def test_score_predictions_perfect() -> None:
    metrics = score_predictions("x", [0, 1, 1, 0], [0, 1, 1, 0])
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0
    assert metrics.event_f1 == 1.0


def test_event_adjusted_counts_partial_hit() -> None:
    y_true = [0, 1, 1, 1, 0]
    y_pred = [0, 0, 1, 0, 0]
    metrics = score_predictions("partial", y_true, y_pred)
    assert metrics.recall < 1.0
    assert metrics.event_f1 == 1.0


def test_evaluate_split_returns_all_detectors(labeled_split) -> None:
    train, test = labeled_split
    metrics = evaluate_ensemble(train, test)
    names = {m.name for m in metrics}
    assert {"threshold", "zscore", "isolation_forest", "stuck_sensor", "ensemble"} <= names
    ensemble = next(m for m in metrics if m.name == "ensemble")
    # Ensemble should catch at least one injected event on this seed.
    assert ensemble.true_positives + ensemble.false_negatives == ensemble.n_positive
    assert ensemble.n_positive > 0
    assert ensemble.recall > 0.0


def test_threshold_only_on_split() -> None:
    train, test = generate_split(180, 120, seed=11)
    metrics = evaluate_detector("threshold", ThresholdDetector(), train, test)
    assert metrics.n_test == 120
    assert 0.0 <= metrics.f1 <= 1.0
