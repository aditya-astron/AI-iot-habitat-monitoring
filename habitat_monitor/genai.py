"""Structured habitat reports from prompt templates + an LLM client.

The client is a narrow protocol (``complete(prompt) -> str``). Tests inject
``MockLLM``. Live calls go through ``OpenAICompatibleClient`` (HTTP chat
completions). Nothing here is a one-line OpenAI wrapper: prompts live in
``prompts/``, context is assembled from detections, and the response is
validated as ``HabitatReport``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Protocol

import httpx

from habitat_monitor.config import Settings, get_settings
from habitat_monitor.schema import Detection, HabitatReport, SamplingDecision

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


class LLMClient(Protocol):
    def complete(self, prompt: str) -> str: ...


class LLMError(RuntimeError):
    """The model returned nothing we can parse as HabitatReport."""


def load_template(name: str) -> str:
    path = PROMPTS_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"prompt template not found: {path}")
    return path.read_text(encoding="utf-8")


def render_prompt(template: str, variables: dict[str, str]) -> str:
    """Replace ``{{key}}`` placeholders. Unknown keys are left intact."""

    def repl(match: re.Match[str]) -> str:
        key = match.group(1)
        return variables.get(key, match.group(0))

    return re.sub(r"\{\{(\w+)\}\}", repl, template)


def extract_json_object(text: str) -> str:
    """Pull the first JSON object from a model response (fences allowed)."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end < start:
        raise LLMError("response does not contain a JSON object")
    return stripped[start : end + 1]


def build_context(
    detections: list[Detection],
    sampling: SamplingDecision | None = None,
    extra_notes: str = "",
) -> dict[str, str]:
    anomalies = [d for d in detections if d.is_anomaly]
    n = len(detections)
    rate = (len(anomalies) / n) if n else 0.0
    # Compact table for the prompt — avoid dumping raw frames.
    lines = []
    for det in anomalies[:12]:
        s = det.sensors
        lines.append(
            f"seq={det.seq} risk={det.risk_score:.2f} "
            f"T={s.temperature_c:.1f}C RH={s.humidity_pct:.0f}% "
            f"soil={s.soil_moisture_pct:.0f}% CO2={s.co2_ppm:.0f} "
            f"lux={s.light_lux:.0f} detectors={','.join(det.detectors) or '-'} "
            f"reasons={'; '.join(det.reasons) or '-'}"
        )
    if not lines:
        lines.append("(no anomalies flagged)")
    interval = sampling.interval_ms if sampling is not None else 30_000
    rationale = sampling.rationale if sampling is not None else "nominal"
    return {
        "n_readings": str(n),
        "n_anomalies": str(len(anomalies)),
        "anomaly_rate": f"{rate:.3f}",
        "anomaly_table": "\n".join(lines),
        "recommended_interval_ms": str(interval),
        "sampling_rationale": rationale,
        "extra_notes": extra_notes or "none",
    }


class MockLLM:
    """Deterministic stand-in used by tests and default GENAI_MODE=mock."""

    def complete(self, prompt: str) -> str:
        n_anom = 0
        match = re.search(r"Anomaly count:\s*(\d+)", prompt)
        if match:
            n_anom = int(match.group(1))
        interval = 30000
        im = re.search(r"Recommended sampling interval \(ms\):\s*(\d+)", prompt)
        if im:
            interval = int(im.group(1))
        health = max(20.0, 96.0 - 8.0 * n_anom)
        if n_anom == 0:
            summary = "No anomalies were flagged in the reviewed window."
            insights = ["Sensor channels stayed inside the expected habitat cycle."]
            actions = ["Keep the nominal sampling interval."]
        else:
            summary = (
                f"{n_anom} anomalous reading(s) were flagged. "
                "Review heat, CO2, soil, stuck-sensor, and lighting events first."
            )
            insights = [
                "Flagged points deviate from the fitted habitat baseline.",
                "Event clusters matter more than isolated false positives.",
            ]
            actions = [
                "Inspect the listed detector reasons in order of risk.",
                f"Consider sampling at {interval} ms while the rate stays elevated.",
            ]
        report = HabitatReport(
            executive_summary=summary,
            health_score=round(health, 1),
            key_insights=insights,
            recommended_actions=actions,
            confidence=0.7 if n_anom else 0.85,
            anomaly_count=n_anom,
            recommended_interval_ms=interval,
        )
        return report.model_dump_json()


class OpenAICompatibleClient:
    """Chat Completions client (OpenAI or any compatible base URL)."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        timeout_s: float = 30.0,
        http: httpx.Client | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("api_key is required for OpenAICompatibleClient")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self._http = http

    def complete(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "temperature": 0.1,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a habitat-systems analyst. Return only valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        url = f"{self.base_url}/chat/completions"
        if self._http is not None:
            response = self._http.post(url, json=payload, headers=headers, timeout=self.timeout_s)
        else:
            response = httpx.post(url, json=payload, headers=headers, timeout=self.timeout_s)
        response.raise_for_status()
        data = response.json()
        try:
            return str(data["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("unexpected chat completion shape") from exc


def client_from_settings(settings: Settings | None = None) -> LLMClient:
    cfg = settings or get_settings()
    if cfg.genai_mode == "mock" or not cfg.openai_api_key:
        return MockLLM()
    return OpenAICompatibleClient(
        api_key=cfg.openai_api_key,
        model=cfg.openai_model,
        base_url=cfg.openai_base_url,
    )


def generate_report(
    detections: list[Detection],
    sampling: SamplingDecision | None = None,
    *,
    client: LLMClient | None = None,
    extra_notes: str = "",
) -> HabitatReport:
    llm = client or client_from_settings()
    template = load_template("habitat_report.md")
    variables = build_context(detections, sampling, extra_notes=extra_notes)
    prompt = render_prompt(template, variables)
    raw = llm.complete(prompt)
    try:
        payload = extract_json_object(raw)
        report = HabitatReport.model_validate_json(payload)
    except Exception as exc:  # noqa: BLE001 — surface parse failures as LLMError
        raise LLMError(f"failed to parse HabitatReport: {exc}") from exc
    return report


def generate_alert_brief(
    detections: list[Detection],
    *,
    client: LLMClient | None = None,
) -> str:
    llm = client or client_from_settings()
    template = load_template("alert_brief.md")
    variables = build_context(detections)
    prompt = render_prompt(template, variables)
    return llm.complete(prompt).strip()
