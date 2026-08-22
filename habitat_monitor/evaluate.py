"""Labeled evaluation for anomaly detectors.

Method
------
1. Generate (or load) a clean training window and a held-out test window
   with injected contiguous events (see ``generate.generate_split``).
2. Fit each detector on the training frames only.
3. Score every test frame. Compare ``is_anomaly`` to the binary label.
4. Report precision, recall, F1, and event-adjusted F1.

Event-adjusted F1 (common in time-series AD): if any point inside a
labeled event is detected, every point of that event counts as a true
positive. False positives outside events still count. This avoids
punishing a detector that catches an event two samples late.

No accuracy or wall-clock claim is hardcoded. Numbers come from the
labeled split that ships with the repo (or a caller-supplied one).
"""

from __future__ import annotations

from dataclasses import dataclass

from habitat_monitor.detect import Detector, EnsembleDetector, default_ensemble
from habitat_monitor.schema import LabeledRecord, ReadingFrame


@dataclass(frozen=True)
class MetricSet:
    name: str
    precision: float
    recall: float
    f1: float
    event_f1: float
    true_positives: int
    false_positives: int
    false_negatives: int
    n_test: int
    n_positive: int

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "name": self.name,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "event_f1": self.event_f1,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "n_test": self.n_test,
            "n_positive": self.n_positive,
        }


def _safe_div(num: float, den: float) -> float:
    if den == 0.0:
        return 0.0
    return num / den


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return precision, recall, f1


def _event_adjusted(y_true: list[int], y_pred: list[int]) -> tuple[int, int, int]:
    """Point-adjust: hit any index in a contiguous positive run => all TP."""
    n = len(y_true)
    adjusted = list(y_pred)
    i = 0
    while i < n:
        if y_true[i] != 1:
            i += 1
            continue
        j = i
        while j < n and y_true[j] == 1:
            j += 1
        if any(y_pred[k] == 1 for k in range(i, j)):
            for k in range(i, j):
                adjusted[k] = 1
        i = j
    tp = sum(1 for t, p in zip(y_true, adjusted) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, adjusted) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, adjusted) if t == 1 and p == 0)
    return tp, fp, fn


def score_predictions(name: str, y_true: list[int], y_pred: list[int]) -> MetricSet:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred length mismatch")
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    precision, recall, f1 = _prf(tp, fp, fn)
    etp, efp, efn = _event_adjusted(y_true, y_pred)
    _, _, event_f1 = _prf(etp, efp, efn)
    return MetricSet(
        name=name,
        precision=precision,
        recall=recall,
        f1=f1,
        event_f1=event_f1,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        n_test=len(y_true),
        n_positive=sum(y_true),
    )


def _predict(detector: Detector, frames: list[ReadingFrame]) -> list[int]:
    preds: list[int] = []
    previous: ReadingFrame | None = None
    for frame in frames:
        is_anom, _risk, _reason = detector.score(frame, previous)
        preds.append(1 if is_anom else 0)
        previous = frame
    return preds


def evaluate_detector(
    name: str,
    detector: Detector,
    train: list[LabeledRecord],
    test: list[LabeledRecord],
) -> MetricSet:
    train_frames = [r.frame for r in train]
    test_frames = [r.frame for r in test]
    detector.fit(train_frames)
    y_pred = _predict(detector, test_frames)
    y_true = [int(r.label) for r in test]
    return score_predictions(name, y_true, y_pred)


def evaluate_ensemble(
    train: list[LabeledRecord],
    test: list[LabeledRecord],
    ensemble: EnsembleDetector | None = None,
) -> list[MetricSet]:
    ens = ensemble or default_ensemble()
    train_frames = [r.frame for r in train]
    ens.fit(train_frames)
    results = [evaluate_detector(m.name, m, train, test) for m in ens.members]
    # Ensemble itself (already fitted).
    y_true = [int(r.label) for r in test]
    y_pred: list[int] = []
    previous = None
    for rec in test:
        is_anom, _risk, _flagged, _reasons = ens.score(rec.frame, previous)
        y_pred.append(1 if is_anom else 0)
        previous = rec.frame
    results.append(score_predictions(ens.name, y_true, y_pred))
    return results


def format_report(metrics: list[MetricSet]) -> str:
    header = (
        f"{'detector':<20} {'P':>6} {'R':>6} {'F1':>6} {'eF1':>6} "
        f"{'TP':>4} {'FP':>4} {'FN':>4}"
    )
    lines = [
        "Anomaly detection on labeled synthetic hold-out",
        "P/R/F1 are point-wise. eF1 is event-adjusted (see module docstring).",
        header,
        "-" * len(header),
    ]
    for m in metrics:
        lines.append(
            f"{m.name:<20} {m.precision:6.3f} {m.recall:6.3f} {m.f1:6.3f} "
            f"{m.event_f1:6.3f} {m.true_positives:4d} {m.false_positives:4d} {m.false_negatives:4d}"
        )
    return "\n".join(lines) + "\n"
