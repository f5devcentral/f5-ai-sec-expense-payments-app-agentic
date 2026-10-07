from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

DEFAULT_F5_AI_SECURITY_BASE_URL = "https://us1.calypsoai.app/"


@dataclass
class GuardrailVerdict:
    allowed: bool
    reason: str
    provider: str = "none"
    raw: dict[str, Any] | None = None


def guardrail_check(message: str, context: dict[str, Any]) -> GuardrailVerdict:
    """Swappable inter-agent guardrail insertion point.

    This runs at inference time before one agent's output is accepted as the next
    agent's input. By default it uses the F5 AI Security OpenAI-compatible
    endpoint when F5_AI_SECURITY_BASE_URL and a token are configured.
    """
    config_error = _f5_config_error()
    if not config_error:
        return _f5_guardrail_check(message, context)
    return GuardrailVerdict(
        allowed=False,
        reason=config_error,
        provider="not_configured",
    )


def _has_f5_config() -> bool:
    return _f5_config_error() == ""


def f5_is_configured() -> bool:
    return _has_f5_config()


def f5_openai_base_url() -> str:
    return _f5_openai_base_url()


def f5_project_token() -> str:
    return (os.getenv("F5_AI_SECURITY_PROJECT_TOKEN") or os.getenv("CALYPSOAI_TOKEN") or "").strip()


def _f5_config_error() -> str:
    base_url = _f5_base_url()
    token = (
        os.getenv("F5_AI_SECURITY_PROJECT_TOKEN")
        or os.getenv("CALYPSOAI_TOKEN")
        or ""
    ).strip()
    if not token:
        return (
            "F5 guardrail is ON but F5_AI_SECURITY_PROJECT_TOKEN is not configured. "
            "Turn guardrails off for an unprotected run."
        )
    if "CONNECTION-NAME" in base_url:
        return (
            "F5_AI_SECURITY_BASE_URL still contains CONNECTION-NAME. Set "
            "F5_AI_SECURITY_CONNECTION_NAME or provide a full /openai/<connection-name> base URL."
        )
    if _needs_connection_name(base_url) and not _f5_connection_name():
        return (
            "F5_AI_SECURITY_BASE_URL points to the F5 host root. Set "
            "F5_AI_SECURITY_CONNECTION_NAME so the app can call /openai/<connection-name>."
        )
    return ""


def _f5_guardrail_check(message: str, context: dict[str, Any]) -> GuardrailVerdict:
    base_url = _f5_openai_base_url()
    token = (os.getenv("F5_AI_SECURITY_PROJECT_TOKEN") or os.getenv("CALYPSOAI_TOKEN") or "").strip()
    model = os.getenv("F5_GUARDRAIL_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    trace_id = str(context.get("trace_id") or "expense-demo")

    client = OpenAI(base_url=base_url, api_key=token)
    boundary = str(context.get("boundary") or "inter_agent_boundary")
    source_agent, target_agent = _boundary_agents(boundary)
    guardrail_probe = _boundary_inspection_message(
        message,
        boundary=boundary,
        source_agent=source_agent,
        target_agent=target_agent,
        scenario=str(context.get("scenario") or ""),
    )

    try:
        completion = client.chat.completions.create(
            model=model,
            temperature=0,
            max_tokens=24,
            messages=_f5_boundary_messages(guardrail_probe),
            extra_headers=_f5_metadata_headers(
                trace_id=trace_id,
                boundary=boundary,
                source_agent=source_agent,
                target_agent=target_agent,
                scenario=str(context.get("scenario") or ""),
            ),
        )
    except Exception as exc:  # noqa: BLE001
        parsed = _extract_guardrail_result_from_exception(exc)
        if parsed and parsed.get("outcome") == "blocked":
            return GuardrailVerdict(
                allowed=False,
                reason=parsed.get("message") or _scanner_reason(parsed) or "F5 Guardrails blocked the message.",
                provider="f5",
                raw=parsed,
            )
        raise

    dumped_completion = _model_dump(completion)
    parsed_completion = _extract_guardrail_result_from_payload(dumped_completion)
    if parsed_completion and parsed_completion.get("outcome") in {"blocked", "flagged", "redacted"}:
        return GuardrailVerdict(
            allowed=False,
            reason=(
                parsed_completion.get("message")
                or _scanner_reason(parsed_completion)
                or f"F5 Guardrails returned outcome: {parsed_completion.get('outcome')}."
            ),
            provider="f5",
            raw=parsed_completion,
        )

    choices = getattr(completion, "choices", None)
    if choices is None and dumped_completion:
        return GuardrailVerdict(
            allowed=False,
            reason=(
                "F5 returned a non-OpenAI completion payload for the guardrail check. "
                "Verify F5_AI_SECURITY_PROJECT_TOKEN and connection settings."
            ),
            provider="f5",
            raw=dumped_completion,
        )

    f5_outcome = _extract_guardrail_outcome_from_node(dumped_completion)
    if f5_outcome in {"blocked", "flagged", "redacted"}:
        return GuardrailVerdict(
            allowed=False,
            reason=f"F5 Guardrails returned outcome: {f5_outcome}.",
            provider="f5",
            raw={"outcome": f5_outcome},
        )
    return GuardrailVerdict(
        allowed=True,
        reason="Allowed by F5 guardrail path.",
        provider="f5",
        raw={"outcome": f5_outcome or "clear"},
    )


def _safe_json(content: str) -> dict[str, Any]:
    try:
        parsed = json.loads(content)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {"allowed": True, "reason": content}


def _model_dump(node: Any) -> dict[str, Any]:
    try:
        dumped = node.model_dump(mode="python")
        return dumped if isinstance(dumped, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def _extract_error_payload(exc: Exception) -> dict[str, Any]:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        return body
    if isinstance(body, str):
        return _safe_json(body)
    return {}


def _extract_guardrail_result_from_exception(exc: Exception) -> dict[str, Any] | None:
    return _extract_guardrail_result_from_payload(_extract_error_payload(exc))


def _extract_guardrail_result_from_payload(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    error_node = payload.get("error") if isinstance(payload.get("error"), dict) else payload
    if not isinstance(error_node, dict):
        return None
    cai_error = error_node.get("cai_error")
    if not isinstance(cai_error, dict):
        return None
    outcome = str(cai_error.get("outcome") or "").strip().lower()
    if outcome not in {"blocked", "flagged", "redacted", "cleared", "clear"}:
        return None
    return {
        "outcome": outcome,
        "message": str(error_node.get("message") or ""),
        "scanner_results": cai_error.get("scanner_results") or cai_error.get("scannerResults") or [],
        "analysis": cai_error.get("analysis"),
        "response": cai_error.get("response"),
    }


def _extract_guardrail_outcome_from_node(node: Any) -> str | None:
    if not isinstance(node, dict):
        return None
    parsed = _extract_guardrail_result_from_payload(node)
    if parsed:
        return parsed["outcome"]
    for key in ("cai_error", "guardrails", "guardrail", "cai", "metadata", "_meta"):
        child = node.get(key)
        if isinstance(child, dict):
            outcome = str(child.get("outcome") or "").strip().lower()
            if outcome in {"blocked", "flagged", "redacted", "cleared", "clear"}:
                return outcome
            nested = _extract_guardrail_outcome_from_node(child)
            if nested:
                return nested
    return None


def _scanner_reason(parsed: dict[str, Any]) -> str:
    scanners = parsed.get("scanner_results")
    if not isinstance(scanners, list):
        return ""
    names = [
        str(item.get("scanner") or item.get("name") or item.get("type") or "")
        for item in scanners
        if isinstance(item, dict)
    ]
    names = [name for name in names if name]
    return f"Blocked by scanner(s): {', '.join(names)}" if names else ""


def _boundary_inspection_message(
    message: str,
    *,
    boundary: str,
    source_agent: str,
    target_agent: str,
    scenario: str = "",
) -> str:
    scenario_line = f"Scenario: {scenario}\n" if scenario else ""
    return (
        f"Boundary inspection: {source_agent} -> {target_agent}\n"
        f"Boundary key: {boundary}\n"
        f"{scenario_line}"
        f"Trust assumption: {target_agent} treats this as trusted output from {source_agent}.\n\n"
        "Dynamic handoff content:\n"
        "---\n"
        f"{message}\n"
        "---"
    )


def _f5_boundary_messages(guardrail_probe: str) -> list[dict[str, str]]:
    return [
        {
            "role": "user",
            "content": (
                "Record this multi-agent boundary inspection and reply exactly: "
                "Boundary inspection recorded.\n\n"
                f"{guardrail_probe}"
            ),
        }
    ]


def _f5_metadata_headers(
    *,
    trace_id: str,
    boundary: str,
    source_agent: str,
    target_agent: str,
    scenario: str = "",
) -> dict[str, str]:
    headers = {
        "x-cai-metadata-session-id": _metadata_value(trace_id),
        "x-cai-metadata-boundary": _metadata_value(boundary),
        "x-cai-metadata-source-agent": _metadata_value(source_agent),
        "x-cai-metadata-target-agent": _metadata_value(target_agent),
    }
    if scenario:
        headers["x-cai-metadata-demo-scenario"] = _metadata_value(scenario)
    return headers


def _metadata_value(value: str) -> str:
    normalized = value.replace("\n", " ").strip()
    return normalized[:255]


def _boundary_agents(boundary: str) -> tuple[str, str]:
    labels = {
        "intake_agent_to_compliance_agent": ("Intake Agent", "Compliance Agent"),
        "compliance_agent_to_payment_agent": ("Compliance Agent", "Payment Agent"),
    }
    if boundary in labels:
        return labels[boundary]
    if "_to_" not in boundary:
        return "Upstream Agent", "Downstream Agent"
    source, target = boundary.split("_to_", 1)
    return _agent_label(source), _agent_label(target)


def _agent_label(value: str) -> str:
    label = value.replace("_", " ").replace("-", " ").strip().title()
    return label if label.endswith("Agent") else f"{label} Agent"


def _f5_base_url() -> str:
    return (os.getenv("F5_AI_SECURITY_BASE_URL") or DEFAULT_F5_AI_SECURITY_BASE_URL).strip()


def _f5_connection_name() -> str:
    return (os.getenv("F5_AI_SECURITY_CONNECTION_NAME") or "").strip().strip("/")


def _needs_connection_name(base_url: str) -> bool:
    normalized = base_url.rstrip("/")
    return "/openai/" not in normalized and not normalized.endswith("/openai")


def _f5_openai_base_url() -> str:
    base_url = _f5_base_url().rstrip("/")
    if not _needs_connection_name(base_url):
        return base_url
    return f"{base_url}/openai/{_f5_connection_name()}"
