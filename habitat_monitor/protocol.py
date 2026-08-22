"""Line-delimited JSON protocol shared with firmware/habitat_sensor.ino.

Wire format
-----------
Baud: 115200 8N1. One UTF-8 JSON object per line, terminated by ``\\n``.
Maximum line length: 1024 bytes (firmware buffer and host parser agree).

Reading (device -> host)::

    {"v":1,"type":"reading","device_id":"hab-01","seq":12,"ts_ms":18432000,
     "sensors":{"temperature_c":24.10,"humidity_pct":61.20,"soil_moisture_pct":38.00,
                "co2_ppm":450.0,"light_lux":720.0,"motion":0},"status":"ok"}

Command (host -> device)::

    {"v":1,"type":"cmd","cmd":"ping"}
    {"v":1,"type":"cmd","cmd":"set_interval","interval_ms":15000}

Ack (device -> host)::

    {"v":1,"type":"ack","device_id":"hab-01","ack":"pong","uptime_ms":18432000}
    {"v":1,"type":"ack","device_id":"hab-01","ack":"set_interval","interval_ms":15000}

See firmware/PROTOCOL.md for pinout and timing.
"""

from __future__ import annotations

import json

from pydantic import ValidationError

from habitat_monitor.schema import AckFrame, CommandFrame, ReadingFrame

PROTOCOL_VERSION = 1
BAUD_RATE = 115200
MAX_LINE_BYTES = 1024
LINE_TERMINATOR = "\n"

Frame = ReadingFrame | CommandFrame | AckFrame


class ProtocolError(ValueError):
    """A line could not be parsed as a protocol frame."""


def encode_frame(frame: Frame) -> str:
    """Serialize a frame to a single line without the trailing newline."""
    payload = frame.model_dump_json(exclude_none=True)
    if len(payload.encode("utf-8")) > MAX_LINE_BYTES:
        raise ProtocolError(f"encoded frame exceeds {MAX_LINE_BYTES} bytes")
    return payload


def parse_line(line: str | bytes) -> Frame:
    """Parse one protocol line. Blank lines raise ProtocolError."""
    if isinstance(line, bytes):
        if len(line) > MAX_LINE_BYTES:
            raise ProtocolError(f"line exceeds {MAX_LINE_BYTES} bytes")
        try:
            text = line.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ProtocolError("line is not valid UTF-8") from exc
    else:
        if len(line.encode("utf-8")) > MAX_LINE_BYTES:
            raise ProtocolError(f"line exceeds {MAX_LINE_BYTES} bytes")
        text = line

    text = text.strip()
    if not text:
        raise ProtocolError("empty line")

    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"invalid JSON: {exc.msg}") from exc

    if not isinstance(raw, dict):
        raise ProtocolError("frame must be a JSON object")

    version = raw.get("v")
    if version != PROTOCOL_VERSION:
        raise ProtocolError(f"unsupported protocol version: {version!r}")

    frame_type = raw.get("type")
    model: type[Frame]
    if frame_type == "reading":
        model = ReadingFrame
    elif frame_type == "cmd":
        model = CommandFrame
    elif frame_type == "ack":
        model = AckFrame
    else:
        raise ProtocolError(f"unknown frame type: {frame_type!r}")

    try:
        return model.model_validate(raw)
    except ValidationError as exc:
        raise ProtocolError(f"invalid {frame_type} frame: {exc.error_count()} error(s)") from exc


def parse_stream(text: str) -> list[Frame]:
    """Parse a multi-line buffer, skipping blank lines."""
    frames: list[Frame] = []
    for raw_line in text.splitlines():
        if not raw_line.strip():
            continue
        frames.append(parse_line(raw_line))
    return frames
