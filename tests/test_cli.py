from __future__ import annotations

import json
from pathlib import Path

from habitat_monitor.cli import main


def test_generate_and_evaluate(tmp_path: Path, capsys) -> None:
    out = tmp_path / "stream.jsonl"
    assert main(["generate", "--split", "--n-train", "80", "--n-test", "40", "--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(text) == 120
    first = json.loads(text[0])
    assert "frame" in first and "label" in first
    assert main(["evaluate", str(out), "--fit", "80"]) == 0
    captured = capsys.readouterr()
    assert "isolation_forest" in captured.out
    assert "ensemble" in captured.out


def test_replay_and_detect(tmp_path: Path, capsys) -> None:
    out = tmp_path / "s.jsonl"
    assert main(["generate", "--n", "80", "--seed", "1", "--out", str(out)]) == 0
    assert main(["replay", str(out)]) == 0
    replayed = capsys.readouterr().out
    assert '"type":"reading"' in replayed
    assert main(["detect", str(out), "--fit", "40", "--holdout", "--anomalies-only"]) == 0


def test_report_mock(tmp_path: Path, capsys) -> None:
    out = tmp_path / "s.jsonl"
    assert main(["generate", "--n", "60", "--out", str(out)]) == 0
    assert main(["report", str(out), "--fit", "30"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "executive_summary" in payload
    assert 0.0 <= payload["health_score"] <= 100.0
