from __future__ import annotations

from pathlib import Path

from habitat_monitor.protocol import parse_line
from habitat_monitor.schema import SENSOR_KEYS

INO = Path(__file__).resolve().parents[1] / "firmware" / "habitat_sensor.ino"


def test_firmware_emits_protocol_field_names() -> None:
    source = INO.read_text(encoding="utf-8")
    # C string literals escape quotes: \"type\":\"reading\"
    assert "type" in source and "reading" in source
    assert "set_interval" in source
    assert "ping" in source
    for key in SENSOR_KEYS:
        assert key in source
    assert "115200" in source
    assert "1024" in source


def test_firmware_comment_example_parses() -> None:
    line = (
        '{"v":1,"type":"reading","device_id":"hab-01","seq":0,"ts_ms":0,'
        '"sensors":{"temperature_c":24.10,"humidity_pct":61.20,"soil_moisture_pct":40.00,'
        '"co2_ppm":430.0,"light_lux":700.0,"motion":0},"status":"ok"}'
    )
    frame = parse_line(line)
    assert frame.type == "reading"
