# Habitat window report

You are reviewing environmental telemetry from an enclosed habitat / analog-mission bay.
Use only the facts in this prompt. Do not invent sensor values, locations, or percentages
that are not implied by the table. If the window is quiet, say so.

## Window
- Readings reviewed: {{n_readings}}
- Anomaly count: {{n_anomalies}}
- Anomaly rate: {{anomaly_rate}}
- Recommended sampling interval (ms): {{recommended_interval_ms}}
- Sampling rationale: {{sampling_rationale}}
- Notes: {{extra_notes}}

## Flagged readings (at most 12)
{{anomaly_table}}

## Output
Return a single JSON object, no markdown fences, matching:

```
{
  "executive_summary": "string, 1-3 sentences",
  "health_score": 0-100 number,
  "key_insights": ["string", "..."],
  "recommended_actions": ["string", "..."],
  "confidence": 0-1 number,
  "anomaly_count": integer (must equal the anomaly count above),
  "recommended_interval_ms": integer (must equal the interval above)
}
```

health_score should fall as anomaly_count rises. confidence should be lower when
the table is empty or reasons conflict. recommended_actions must be operational
(inspect a channel, change interval, check a sensor), not marketing language.
