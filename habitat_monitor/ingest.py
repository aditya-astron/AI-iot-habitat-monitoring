"""Ingest readings from JSONL files, in-memory buffers, or a serial port."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

from habitat_monitor.protocol import ProtocolError, parse_line
from habitat_monitor.schema import LabeledRecord, ReadingFrame


class LineSource(Protocol):
    def readline(self) -> str: ...

    def close(self) -> None: ...


class FileLineSource:
    def __init__(self, path: Path) -> None:
        self._fh = path.open("r", encoding="utf-8")

    def readline(self) -> str:
        return self._fh.readline()

    def close(self) -> None:
        self._fh.close()


class SerialLineSource:
    """Thin wrapper around pyserial. Imported lazily so tests need no hardware."""

    def __init__(self, port: str, baud: int = 115200, timeout: float = 1.0) -> None:
        try:
            import serial  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("pyserial is required for serial ingest") from exc
        self._ser = serial.Serial(port, baudrate=baud, timeout=timeout)

    def readline(self) -> str:
        raw = self._ser.readline()
        if not raw:
            return ""
        return raw.decode("utf-8", errors="replace")

    def close(self) -> None:
        self._ser.close()


def iter_frames(source: LineSource) -> Iterator[ReadingFrame]:
    """Yield reading frames. Non-reading frames and blank lines are skipped."""
    try:
        while True:
            line = source.readline()
            if line == "":
                break
            if not line.strip():
                continue
            try:
                frame = parse_line(line)
            except ProtocolError:
                continue
            if isinstance(frame, ReadingFrame):
                yield frame
    finally:
        source.close()


def load_jsonl(path: Path) -> list[ReadingFrame]:
    """Load either raw reading frames or LabeledRecord wrappers."""
    text = path.read_text(encoding="utf-8")
    frames: list[ReadingFrame] = []
    for raw_line in text.splitlines():
        if not raw_line.strip():
            continue
        rec = _parse_record_or_frame(raw_line)
        frames.append(rec if isinstance(rec, ReadingFrame) else rec.frame)
    return frames


def load_labeled_jsonl(path: Path) -> list[LabeledRecord]:
    records: list[LabeledRecord] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        rec = _parse_record_or_frame(raw_line)
        if isinstance(rec, LabeledRecord):
            records.append(rec)
        else:
            records.append(LabeledRecord(frame=rec, label=0, event=None))
    return records


def _parse_record_or_frame(line: str) -> ReadingFrame | LabeledRecord:
    stripped = line.strip()
    if '"frame"' in stripped:
        return LabeledRecord.model_validate_json(stripped)
    parsed = parse_line(stripped)
    if not isinstance(parsed, ReadingFrame):
        raise ProtocolError("expected a reading frame or labeled record")
    return parsed
