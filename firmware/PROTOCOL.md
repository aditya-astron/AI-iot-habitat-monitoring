# Habitat serial protocol

Firmware (`habitat_sensor.ino`) and the Python parser (`habitat_monitor/protocol.py`)
share one line-delimited JSON contract. If you change a field name, change both.

## Physical layer

| Item        | Value                          |
|-------------|--------------------------------|
| Link        | UART / USB-serial              |
| Baud        | 115200 8N1                     |
| Framing     | one UTF-8 JSON object per line |
| Max line    | 1024 bytes                     |
| Terminator  | `\\n` (LF). CR is stripped     |

## Frame types

`v` is always `1`. Unknown `type` values are a hard parse error on the host.

### `reading` (device → host)

```json
{"v":1,"type":"reading","device_id":"hab-01","seq":12,"ts_ms":18432000,"sensors":{"temperature_c":24.10,"humidity_pct":61.20,"soil_moisture_pct":38.00,"co2_ppm":450.0,"light_lux":720.0,"motion":0},"status":"ok"}
```

| Field                         | Meaning                                      |
|-------------------------------|----------------------------------------------|
| `device_id`                   | Stable node name, max 64 chars               |
| `seq`                         | Monotonic uint, wraps at 2^32                |
| `ts_ms`                       | Device uptime milliseconds                   |
| `sensors.temperature_c`       | DHT22 (or simulated), °C                     |
| `sensors.humidity_pct`        | 0–100 %RH                                    |
| `sensors.soil_moisture_pct`   | 0–100, mapped from analog soil probe         |
| `sensors.co2_ppm`             | MQ-135 mapped ppm (approximate)              |
| `sensors.light_lux`           | BH1750 lux, or analog LDR mapped             |
| `sensors.motion`              | PIR counts since last frame (integer ≥ 0)    |
| `status`                      | `ok` \| `degraded` \| `fault`                |

### `cmd` (host → device)

```json
{"v":1,"type":"cmd","cmd":"ping"}
{"v":1,"type":"cmd","cmd":"set_interval","interval_ms":15000}
```

`interval_ms` must be between 1000 and 3600000.

### `ack` (device → host)

```json
{"v":1,"type":"ack","device_id":"hab-01","ack":"pong","uptime_ms":18432000}
{"v":1,"type":"ack","device_id":"hab-01","ack":"set_interval","interval_ms":15000}
```

## Host ingest

```bash
python -m habitat_monitor replay data/sample_stream.jsonl
# or, with a board on /dev/ttyUSB0
HABITAT_SERIAL_PORT=/dev/ttyUSB0 python -m habitat_monitor detect /dev/stdin
```

The Python parser rejects unknown versions, missing sensor keys, and lines
over 1024 bytes. Extra JSON keys on a valid frame are ignored by Pydantic.
