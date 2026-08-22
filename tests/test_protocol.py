from __future__ import annotations

import pytest

from habitat_monitor.protocol import (
    BAUD_RATE,
    MAX_LINE_BYTES,
    PROTOCOL_VERSION,
    ProtocolError,
    encode_frame,
    parse_line,
    parse_stream,
)
from habitat_monitor.schema import CommandFrame, ReadingFrame

FIRMWARE_READING = (
    '{"v":1,"type":"reading","device_id":"hab-01","seq":12,"ts_ms":18432000,'
    '"sensors":{"temperature_c":24.10,"humidity_pct":61.20,"soil_moisture_pct":38.00,'
    '"co2_ppm":450.0,"light_lux":720.0,"motion":0},"status":"ok"}'
)
FIRMWARE_PONG = (
    '{"v":1,"type":"ack","device_id":"hab-01","ack":"pong","uptime_ms":18432000}'
)
FIRMWARE_CMD = '{"v":1,"type":"cmd","cmd":"set_interval","interval_ms":15000}'


def test_firmware_reading_round_trip() -> None:
    frame = parse_line(FIRMWARE_READING)
    assert isinstance(frame, ReadingFrame)
    assert frame.sensors.temperature_c == pytest.approx(24.10)
    assert frame.sensors.motion == 0
    again = parse_line(encode_frame(frame))
    assert again == frame


def test_firmware_ack_and_cmd() -> None:
    ack = parse_line(FIRMWARE_PONG)
    assert ack.ack == "pong"  # type: ignore[union-attr]
    cmd = parse_line(FIRMWARE_CMD)
    assert isinstance(cmd, CommandFrame)
    assert cmd.interval_ms == 15000


def test_parse_stream_skips_blanks() -> None:
    frames = parse_stream("\n" + FIRMWARE_READING + "\n\n" + FIRMWARE_PONG + "\n")
    assert len(frames) == 2


def test_rejects_bad_version() -> None:
    with pytest.raises(ProtocolError, match="version"):
        parse_line('{"v":2,"type":"cmd","cmd":"ping"}')


def test_rejects_unknown_type() -> None:
    with pytest.raises(ProtocolError, match="unknown"):
        parse_line('{"v":1,"type":"telepathy"}')


def test_rejects_invalid_json() -> None:
    with pytest.raises(ProtocolError, match="JSON"):
        parse_line("{not json")


def test_rejects_oversized_line() -> None:
    blob = "{" + ("a" * (MAX_LINE_BYTES + 10))
    with pytest.raises(ProtocolError, match="exceeds"):
        parse_line(blob)


def test_rejects_missing_sensor() -> None:
    with pytest.raises(ProtocolError):
        parse_line(
            '{"v":1,"type":"reading","device_id":"x","seq":1,"ts_ms":1,'
            '"sensors":{"temperature_c":1,"humidity_pct":1,"soil_moisture_pct":1,'
            '"co2_ppm":1,"light_lux":1},"status":"ok"}'
        )


def test_constants_match_firmware() -> None:
    assert PROTOCOL_VERSION == 1
    assert BAUD_RATE == 115200
    assert MAX_LINE_BYTES == 1024
