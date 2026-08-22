from __future__ import annotations

import json

import httpx
import pytest

from habitat_monitor.detect import default_ensemble
from habitat_monitor.genai import (
    LLMError,
    MockLLM,
    OpenAICompatibleClient,
    build_context,
    extract_json_object,
    generate_report,
    load_template,
    render_prompt,
)
from habitat_monitor.generate import generate_labeled_stream
from habitat_monitor.optimize import recommend_interval
from habitat_monitor.schema import HabitatReport


def test_render_prompt_replaces_placeholders() -> None:
    out = render_prompt("count={{n_anomalies}} leftover={{missing}}", {"n_anomalies": "3"})
    assert out == "count=3 leftover={{missing}}"


def test_templates_exist_and_have_placeholders() -> None:
    report = load_template("habitat_report.md")
    brief = load_template("alert_brief.md")
    assert "{{anomaly_table}}" in report
    assert "{{n_anomalies}}" in report
    assert "{{anomaly_table}}" in brief


def test_extract_json_from_fenced_text() -> None:
    raw = 'intro\n```json\n{"a": 1}\n```\n'
    assert extract_json_object(raw) == '{"a": 1}'


def test_extract_json_raises_on_plain_text() -> None:
    with pytest.raises(LLMError):
        extract_json_object("no object here")


def test_mock_report_uses_prompt_counts() -> None:
    frames = [r.frame for r in generate_labeled_stream(80, seed=5, inject=False)]
    ens = default_ensemble(random_state=5)
    ens.fit(frames)
    noisy = [r.frame for r in generate_labeled_stream(40, seed=6, inject=True)]
    detections = ens.detect_many(noisy)
    sampling = recommend_interval(0.2)
    report = generate_report(detections, sampling, client=MockLLM())
    assert isinstance(report, HabitatReport)
    assert report.anomaly_count == sum(1 for d in detections if d.is_anomaly)
    assert report.recommended_interval_ms == sampling.interval_ms
    ctx = build_context(detections, sampling)
    assert ctx["n_readings"] == str(len(detections))


def test_openai_client_uses_injected_http() -> None:
    payload = {
        "choices": [
            {
                "message": {
                    "content": HabitatReport(
                        executive_summary="ok",
                        health_score=90.0,
                        key_insights=["stable"],
                        recommended_actions=["hold interval"],
                        confidence=0.8,
                        anomaly_count=0,
                        recommended_interval_ms=30000,
                    ).model_dump_json()
                }
            }
        ]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["model"] == "test-model"
        assert request.headers["Authorization"] == "Bearer test-key"
        return httpx.Response(200, json=payload)

    transport = httpx.MockTransport(handler)
    http = httpx.Client(transport=transport)
    client = OpenAICompatibleClient(
        api_key="test-key",
        model="test-model",
        base_url="https://example.test/v1",
        http=http,
    )
    frames = [r.frame for r in generate_labeled_stream(40, seed=1, inject=False)]
    ens = default_ensemble(random_state=1)
    ens.fit(frames)
    detections = ens.detect_many(frames[:8])
    report = generate_report(detections, client=client)
    assert report.executive_summary == "ok"
    http.close()
