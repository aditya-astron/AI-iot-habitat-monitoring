from __future__ import annotations

from pathlib import Path

from habitat_monitor.generate import generate_labeled_stream, records_to_jsonl
from habitat_monitor.ingest import FileLineSource, iter_frames, load_jsonl, load_labeled_jsonl
from habitat_monitor.protocol import encode_frame


def test_load_labeled_and_raw(tmp_path: Path) -> None:
    records = generate_labeled_stream(20, seed=2, inject=True)
    labeled = tmp_path / "lab.jsonl"
    labeled.write_text(records_to_jsonl(records), encoding="utf-8")
    loaded = load_labeled_jsonl(labeled)
    assert len(loaded) == 20
    assert sum(r.label for r in loaded) == sum(r.label for r in records)

    raw = tmp_path / "raw.jsonl"
    raw.write_text("\n".join(encode_frame(r.frame) for r in records) + "\n", encoding="utf-8")
    frames = load_jsonl(raw)
    assert len(frames) == 20
    assert frames[0].device_id == "hab-01"


def test_iter_frames_skips_acks(tmp_path: Path) -> None:
    records = generate_labeled_stream(16, seed=1, inject=False)
    path = tmp_path / "mix.jsonl"
    lines = [encode_frame(records[0].frame), '{"v":1,"type":"ack","device_id":"hab-01","ack":"pong","uptime_ms":1}', encode_frame(records[1].frame)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    got = list(iter_frames(FileLineSource(path)))
    assert len(got) == 2
