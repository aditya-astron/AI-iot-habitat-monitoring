# Habitat monitoring

Python service and ESP32 firmware for enclosed-habitat / analog-mission
environmental monitoring. Sensor frames arrive over a documented serial JSON
protocol. The host builds features, scores anomalies, and adjusts sampling and
alerts. An optional GenAI step turns detections into a structured report.

This is a rewrite of a thin demo. It does **not** claim a fixed “+45% accuracy”
or “−32% analysis time.” Those numbers were not measured. Use `evaluate` on the
included labeled generator if you need metrics.

## Architecture

```
firmware UART ──► protocol.parse ──► features ──► detectors ──► alerts / sampling
   or JSONL file                                          │
                                                          ▼
                                                    optional LLM report
```

| Stage | Module | Role |
|---|---|---|
| Ingest | `habitat_monitor.ingest` | Serial port, JSONL replay, or synthetic stream |
| Features | `habitat_monitor.features` | Raw channels, VPD, first differences, rolling stats |
| Detect | `habitat_monitor.detect` | Operational bounds, z-score vs train, Isolation Forest, stuck-sensor, OR-ensemble |
| Optimize | `habitat_monitor.optimize` | Adaptive interval + hysteresis/cooldown alerts |
| Report | `habitat_monitor.genai` | Prompt templates + `LLMClient` (mock or OpenAI-compatible) |
| Evaluate | `habitat_monitor.evaluate` | Precision / recall / F1 / event-adjusted F1 |

Detectors are fit on a **clean** training window. Isolation Forest’s threshold is
the training-score quantile (`contamination`). Z-score uses per-channel mean/std
from that same window. Neither number is a hardcoded accuracy claim.

## Protocol

Firmware and host share one contract. Field names are identical.

- Baud 115200 8N1, one UTF-8 JSON object per line, max 1024 bytes
- Frame types: `reading`, `cmd`, `ack`
- Sensor keys: `temperature_c`, `humidity_pct`, `soil_moisture_pct`, `co2_ppm`, `light_lux`, `motion`

See [firmware/PROTOCOL.md](firmware/PROTOCOL.md). The parser lives in
`habitat_monitor/protocol.py`. Tests load the same example lines the firmware
`snprintf` produces.

## Requirements

- Python 3.11+
- Optional: ESP32 (or any board that can print the JSON line) for live serial

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env
```

`.env.example` has no secrets. Leave `GENAI_MODE=mock` unless you supply your
own key locally.

### Replay sample data

`data/sample_stream.jsonl` is a labeled split (clean train prefix, then a test
window with injected events). Each line is `{"frame": {...}, "label": 0|1, "event": ...}`.
Firmware never emits `label`; it is evaluation-only.

```bash
python -m habitat_monitor replay data/sample_stream.jsonl
python -m habitat_monitor detect data/sample_stream.jsonl --fit 360 --holdout --anomalies-only
python -m habitat_monitor evaluate data/sample_stream.jsonl --fit 360
python -m habitat_monitor report data/sample_stream.jsonl --fit 360 --optimize
```

Or generate a fresh split:

```bash
python -m habitat_monitor generate --split --n-train 360 --n-test 240 --out /tmp/stream.jsonl
```

### HTTP API

```bash
python -m habitat_monitor serve --port 8000
# GET  /health
# POST /fit      {"frames":[...]}
# POST /detect   {"frames":[...]}
# POST /optimize {"anomaly_rate":0.2,"battery_frac":0.8}
# POST /report   {"detections":[...],"sampling":{...}}
```

`/detect` returns 409 until `/fit` has been called.

### Tests and lint

```bash
ruff check habitat_monitor tests
pytest
```

CI runs both on Python 3.11 and 3.12.

## Firmware connection

1. Flash `firmware/habitat_sensor.ino`. Default `HABITAT_SIMULATE 1` emits
   valid frames without sensors (bench / CI of the protocol).
2. Set `HABITAT_SIMULATE 0` and wire DHT22 / soil / MQ-135 / PIR as commented
   in the sketch when you have hardware. MQ-135 ppm is a linear ADC map, not a
   calibrated gas model.
3. Host: `115200` 8N1 on `HABITAT_SERIAL_PORT`.
4. Optional commands:

```json
{"v":1,"type":"cmd","cmd":"ping"}
{"v":1,"type":"cmd","cmd":"set_interval","interval_ms":15000}
```

The adaptive sampler’s `interval_ms` is what you would send back with
`set_interval`.

## Evaluation method

`habitat_monitor.generate` builds a 24-hour habitat cycle (temperature and light
follow a day sine, humidity anti-correlates, soil drifts, CO2 has a weak
occupancy bump) plus contiguous injected events: `heat_spike`, `co2_event`,
`dry_out`, `sensor_stuck`, `night_light`.

Metrics (see `habitat_monitor/evaluate.py`):

- Point-wise precision, recall, F1
- Event-adjusted F1: if any sample inside a labeled run is flagged, the whole
  run counts as caught. Isolated false positives still count.

Reproduce:

```bash
python -m habitat_monitor evaluate --n-train 360 --n-test 240 --seed 42
```

Measured on that command (synthetic hold-out, seed 42, 360 train / 240 test,
41 labeled positive points). Re-run if you change the generator.

```
detector             P      R     F1    eF1   TP  FP  FN
threshold        0.000  0.000  0.000  0.000    0   0  41
zscore           0.976  1.000  0.988  0.988   41   1   0
isolation_forest 0.862  0.610  0.714  0.861   25   4  16
stuck_sensor     1.000  0.171  0.292  0.509    7   0  34
ensemble         0.891  1.000  0.943  0.943   41   5   0
```

Threshold stays at zero because injected events remain inside the operational
envelopes (e.g. CO2 800 ppm is abnormal for this cycle but below the 1800 ppm
fault bound). Z-score is strong here because events are large deviations from
the clean training window — that is a property of this generator, not a field
guarantee. Isolation Forest is weaker on single-axis spikes. Stuck-sensor only
fires on flatlined channels, so its recall is low by design. The ensemble
keeps full recall and picks up a few extra false positives.

Do not treat these as habitat-deployment accuracy.

## GenAI reports

Prompts live in `prompts/`. The model interface is `complete(prompt) -> str`.
Tests inject `MockLLM` and a mocked `httpx` transport for
`OpenAICompatibleClient`. Default `GENAI_MODE=mock` never calls a network.

To use a real endpoint, set `GENAI_MODE=openai` and `OPENAI_API_KEY` in a local
`.env` that is gitignored.

## Limitations

- No real multi-week habitat dataset is included. All metrics are on the
  synthetic cycle described above.
- Isolation Forest is unsupervised. Performance depends on how well the
  training window matches later “normal.”
- MQ-135 and analog soil mappings in firmware are uncalibrated.
- Alert/sampling policies are deterministic heuristics, not a solved control
  problem and not a drone-dispatch optimizer.
- The HTTP API keeps one in-memory pipeline (single process). It is a local
  tool, not a multi-tenant service.
- This is not a life-support or certified ECLSS stack.

## License

MIT. See [LICENSE](LICENSE).
