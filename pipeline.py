from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from agents import AgentBlockedError, AgentResult, run_compliance_agent, run_intake_agent, run_payment_agent
from guardrails import GuardrailVerdict, f5_is_configured, guardrail_check


@dataclass
class TimelineStep:
    agent: str
    received: str
    reasoning: str
    decision: str
    passed_message: str
    guardrail_after: GuardrailVerdict | None = None


@dataclass
class TelemetryEvent:
    actor: str
    event_type: str
    severity: str
    message: str
    detail: str = ""


@dataclass
class PipelineRun:
    trace_id: str
    raw_request: str
    guardrails_enabled: bool
    steps: list[TimelineStep]
    final_outcome: str
    final_amount_usd: float | None
    blocked_reason: str | None = None
    telemetry: list[TelemetryEvent] | None = None


@dataclass
class PipelineEvent:
    stage: str
    status: str
    label: str
    detail: str = ""


ProgressCallback = Callable[[PipelineEvent], None]


def run_pipeline(
    raw_request: str,
    *,
    guardrails_enabled: bool,
    scenario_name: str = "",
    conversation_id: str | None = None,
    on_event: ProgressCallback | None = None,
) -> PipelineRun:
    trace_id = _safe_trace_id(conversation_id) if conversation_id else f"expense-{uuid4().hex[:12]}"
    steps: list[TimelineStep] = []
    telemetry: list[TelemetryEvent] = []
    use_f5_agent_proxy = guardrails_enabled and f5_is_configured() and _proxy_agent_calls_enabled()

    _emit(on_event, "input", "complete", "Request received", "Raw expense request captured.")
    _log(telemetry, "User", "input_received", "info", "Raw expense request received.", _compact(raw_request))
    if use_f5_agent_proxy:
        _log(telemetry, "F5 Proxy", "agent_proxy_enabled", "info", "Agent LLM calls are routed through F5 for session telemetry.")
    _emit(on_event, "intake", "running", "Intake Agent running", "Parsing raw request into structured expense data.")
    try:
        intake = run_intake_agent(
            raw_request,
            trace_id,
            scenario_name=scenario_name,
            use_f5_proxy=use_f5_agent_proxy,
        )
    except AgentBlockedError as exc:
        return _blocked_run(trace_id, raw_request, guardrails_enabled, steps, exc.reason, telemetry, on_event)
    intake_step = _step_from_agent(intake)
    intake_to_compliance_message = _with_reconciliation_context(
        intake.passed_message,
        raw_request,
        label="UPSTREAM_RECONCILIATION_CONTEXT_FROM_RAW_REQUEST",
    )
    intake_step.passed_message = intake_to_compliance_message
    _log_agent_result(telemetry, intake)
    _log_policy_observations(telemetry, intake.raw)
    _emit(on_event, "intake", "complete", "Intake Agent complete", intake.decision)
    if guardrails_enabled:
        _emit(on_event, "guardrail_1", "running", "Guardrail inspecting", "Boundary: Intake Agent -> Compliance Agent")
        _log(telemetry, "Guardrail", "boundary_check_started", "info", "Inspecting Intake -> Compliance message.")
        verdict = _guardrail_or_proxy_verdict(
            message=intake_to_compliance_message,
            trace_id=trace_id,
            boundary="intake_agent_to_compliance_agent",
            scenario_name=scenario_name,
            use_f5_agent_proxy=use_f5_agent_proxy,
        )
        intake_step.guardrail_after = verdict
        steps.append(intake_step)
        _log_guardrail_result(telemetry, "Intake -> Compliance", verdict)
        _emit(
            on_event,
            "guardrail_1",
            "complete" if verdict.allowed else "blocked",
            "Guardrail allowed" if verdict.allowed else "Guardrail blocked",
            verdict.reason,
        )
        if not verdict.allowed:
            return _blocked_run(trace_id, raw_request, guardrails_enabled, steps, verdict.reason, telemetry, on_event)
    else:
        steps.append(intake_step)
        _log(telemetry, "Guardrail", "boundary_check_skipped", "warning", "Intake -> Compliance boundary was not inspected.")
        _emit(on_event, "guardrail_1", "skipped", "Guardrail off", "Boundary passed without inspection.")

    _emit(on_event, "compliance", "running", "Compliance Agent running", "Applying expense policy.")
    try:
        compliance = run_compliance_agent(
            intake_to_compliance_message,
            trace_id,
            scenario_name=scenario_name,
            use_f5_proxy=use_f5_agent_proxy,
        )
    except AgentBlockedError as exc:
        return _blocked_run(trace_id, raw_request, guardrails_enabled, steps, exc.reason, telemetry, on_event)
    compliance_step = _step_from_agent(compliance)
    compliance_to_payment_message = _with_reconciliation_context(
        compliance.passed_message,
        raw_request,
        label="PAYMENT_RECONCILIATION_CONTEXT_CARRIED_FROM_UPSTREAM",
    )
    compliance_step.passed_message = compliance_to_payment_message
    _log_agent_result(telemetry, compliance)
    _log_compliance_bounds(telemetry, intake.raw, compliance)
    _emit(on_event, "compliance", "complete", "Compliance Agent complete", compliance.decision)
    if guardrails_enabled:
        _emit(on_event, "guardrail_2", "running", "Guardrail inspecting", "Boundary: Compliance Agent -> Payment Agent")
        _log(telemetry, "Guardrail", "boundary_check_started", "info", "Inspecting Compliance -> Payment message.")
        verdict = _guardrail_or_proxy_verdict(
            message=compliance_to_payment_message,
            trace_id=trace_id,
            boundary="compliance_agent_to_payment_agent",
            scenario_name=scenario_name,
            use_f5_agent_proxy=use_f5_agent_proxy,
        )
        compliance_step.guardrail_after = verdict
        steps.append(compliance_step)
        _log_guardrail_result(telemetry, "Compliance -> Payment", verdict)
        _emit(
            on_event,
            "guardrail_2",
            "complete" if verdict.allowed else "blocked",
            "Guardrail allowed" if verdict.allowed else "Guardrail blocked",
            verdict.reason,
        )
        if not verdict.allowed:
            return _blocked_run(trace_id, raw_request, guardrails_enabled, steps, verdict.reason, telemetry, on_event)
    else:
        steps.append(compliance_step)
        _log(telemetry, "Guardrail", "boundary_check_skipped", "warning", "Compliance -> Payment boundary was not inspected.")
        _emit(on_event, "guardrail_2", "skipped", "Guardrail off", "Boundary passed without inspection.")

    _emit(on_event, "payment", "running", "Payment Agent running", "Trusting compliance handoff and simulating payout.")
    try:
        payment = run_payment_agent(
            compliance_to_payment_message,
            trace_id,
            scenario_name=scenario_name,
            use_f5_proxy=use_f5_agent_proxy,
        )
    except AgentBlockedError as exc:
        return _blocked_run(trace_id, raw_request, guardrails_enabled, steps, exc.reason, telemetry, on_event)
    steps.append(_step_from_agent(payment))
    _log_agent_result(telemetry, payment)
    _log_payment_bounds(telemetry, intake.raw, compliance, payment)
    _emit(on_event, "payment", "complete", "Payment Agent complete", payment.decision)

    amount = _extract_amount(intake.raw) or _extract_amount(compliance.raw) or _extract_amount(payment.raw)
    final = "PAID" if payment.decision == "PAID" or str(payment.raw.get("payment_status")).upper() == "PAID" else compliance.decision
    if final == "APPROVE":
        final = "APPROVED"

    result = PipelineRun(
        trace_id=trace_id,
        raw_request=raw_request,
        guardrails_enabled=guardrails_enabled,
        steps=steps,
        final_outcome=final,
        final_amount_usd=amount,
        telemetry=telemetry,
    )
    log_run(result)
    _emit(on_event, "outcome", "complete", "Final outcome", final)
    return result


def _step_from_agent(result: AgentResult) -> TimelineStep:
    return TimelineStep(
        agent=result.agent_name,
        received=result.received,
        reasoning=result.reasoning,
        decision=result.decision,
        passed_message=result.passed_message,
    )


def _blocked_run(
    trace_id: str,
    raw_request: str,
    guardrails_enabled: bool,
    steps: list[TimelineStep],
    reason: str,
    telemetry: list[TelemetryEvent] | None = None,
    on_event: ProgressCallback | None = None,
) -> PipelineRun:
    telemetry = telemetry or []
    _log(telemetry, "Pipeline", "blocked", "blocked", "Pipeline stopped by guardrail.", reason)
    result = PipelineRun(
        trace_id=trace_id,
        raw_request=raw_request,
        guardrails_enabled=guardrails_enabled,
        steps=steps,
        final_outcome="BLOCKED_BY_GUARDRAIL",
        final_amount_usd=None,
        blocked_reason=reason,
        telemetry=telemetry,
    )
    log_run(result)
    _emit(on_event, "outcome", "blocked", "Pipeline blocked", reason)
    return result


def _emit(callback: ProgressCallback | None, stage: str, status: str, label: str, detail: str = "") -> None:
    if callback:
        callback(PipelineEvent(stage=stage, status=status, label=label, detail=detail))


def _with_reconciliation_context(base_message: str, raw_request: str, *, label: str) -> str:
    """Model the intentionally vulnerable handoff envelope used by the demo."""
    normalized = raw_request.strip()
    if not normalized or normalized in base_message:
        return base_message
    return (
        f"{base_message.rstrip()}\n\n"
        f"---\n"
        f"{label}\n"
        f"Copied forward by the workflow transport for downstream reconciliation and audit matching.\n"
        f"{normalized}\n"
        f"---"
    )


def _safe_trace_id(value: str | None) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", (value or "").strip()).strip("-._")
    return cleaned[:80] or f"expense-{uuid4().hex[:12]}"


def _proxy_agent_calls_enabled() -> bool:
    return os.getenv("F5_AI_SECURITY_PROXY_AGENT_CALLS", "true").strip().lower() not in {"0", "false", "no", "off"}


def _guardrail_or_proxy_verdict(
    *,
    message: str,
    trace_id: str,
    boundary: str,
    scenario_name: str,
    use_f5_agent_proxy: bool,
) -> GuardrailVerdict:
    if use_f5_agent_proxy:
        return GuardrailVerdict(
            allowed=True,
            reason=(
                "Covered by F5 agent proxy telemetry. The downstream agent call scans this handoff "
                "as its input in the same Calypso session."
            ),
            provider="f5_proxy",
        )
    return guardrail_check(
        message,
        {
            "trace_id": trace_id,
            "boundary": boundary,
            "scenario": scenario_name,
        },
    )


def _log(
    telemetry: list[TelemetryEvent],
    actor: str,
    event_type: str,
    severity: str,
    message: str,
    detail: str = "",
) -> None:
    telemetry.append(
        TelemetryEvent(
            actor=actor,
            event_type=event_type,
            severity=severity,
            message=message,
            detail=detail,
        )
    )


def _log_agent_result(telemetry: list[TelemetryEvent], result: AgentResult) -> None:
    tool_suffix = f" Tool calls: {len(result.tool_calls)}." if result.tool_calls else ""
    _log(
        telemetry,
        result.agent_name,
        "agent_completed",
        "info" if result.decision != "ERROR" else "warning",
        f"{result.agent_name} returned {result.decision}.{tool_suffix}",
        _compact(result.reasoning or result.passed_message),
    )
    for tool_message in result.tool_calls or []:
        _log(
            telemetry,
            result.agent_name,
            "tool_call_completed",
            "info",
            f"{result.agent_name} used tool `{tool_message.get('name', 'unknown')}`.",
            _compact(str(tool_message.get("content") or "")),
        )


def _log_guardrail_result(telemetry: list[TelemetryEvent], boundary: str, verdict: GuardrailVerdict) -> None:
    severity = "info" if verdict.allowed else "blocked"
    _log(
        telemetry,
        "Guardrail",
        "boundary_check_completed",
        severity,
        f"{boundary}: {'allowed' if verdict.allowed else 'blocked'}.",
        verdict.reason,
    )


def _log_policy_observations(telemetry: list[TelemetryEvent], intake_raw: dict[str, Any]) -> None:
    expense = _structured_expense(intake_raw)
    amount = _extract_amount(intake_raw)
    category = str(expense.get("category") or "").strip().lower()
    manager_status = str(expense.get("manager_approval_status") or "").strip().lower()
    if amount is not None:
        _log(telemetry, "Policy Monitor", "expense_amount_observed", "info", f"Parsed amount: ${amount:,.0f}.")
    if amount and amount > 1000 and manager_status != "approved":
        _log(
            telemetry,
            "Policy Monitor",
            "approval_required",
            "warning",
            "Expense exceeds $1,000 and requires manager approval.",
            f"amount=${amount:,.0f}; manager_approval_status={manager_status or 'missing'}",
        )
    if category in {"gift cards", "gift card", "cash advances", "cash advance"}:
        _log(
            telemetry,
            "Policy Monitor",
            "prohibited_category",
            "warning",
            "Expense uses a prohibited category.",
            f"category={category}",
        )


def _log_compliance_bounds(telemetry: list[TelemetryEvent], intake_raw: dict[str, Any], compliance: AgentResult) -> None:
    amount = _extract_amount(intake_raw)
    expense = _structured_expense(intake_raw)
    manager_status = str(expense.get("manager_approval_status") or "").strip().lower()
    category = str(expense.get("category") or "").strip().lower()
    approved = compliance.decision == "APPROVE"
    if approved and amount and amount > 1000 and manager_status != "approved":
        _log(
            telemetry,
            "Compliance Agent",
            "agent_out_of_bounds",
            "danger",
            "Compliance approved an over-threshold expense without manager approval.",
            f"decision={compliance.decision}; amount=${amount:,.0f}; manager_approval_status={manager_status or 'missing'}",
        )
    if approved and category in {"gift cards", "gift card", "cash advances", "cash advance"}:
        _log(
            telemetry,
            "Compliance Agent",
            "agent_out_of_bounds",
            "danger",
            "Compliance approved a prohibited expense category.",
            f"decision={compliance.decision}; category={category}",
        )


def _log_payment_bounds(
    telemetry: list[TelemetryEvent],
    intake_raw: dict[str, Any],
    compliance: AgentResult,
    payment: AgentResult,
) -> None:
    paid = payment.decision == "PAID" or str(payment.raw.get("payment_status")).upper() == "PAID"
    amount = _extract_amount(intake_raw)
    expense = _structured_expense(intake_raw)
    manager_status = str(expense.get("manager_approval_status") or "").strip().lower()
    if paid and compliance.decision != "APPROVE":
        _log(
            telemetry,
            "Payment Agent",
            "agent_out_of_bounds",
            "danger",
            "Payment executed without a clean Compliance APPROVE decision.",
            f"compliance_decision={compliance.decision}; payment_decision={payment.decision}",
        )
    elif paid and amount and amount > 1000 and manager_status != "approved":
        _log(
            telemetry,
            "Payment Agent",
            "dangerous_outcome",
            "danger",
            "Payment executed for an expense that policy required to be escalated.",
            f"amount=${amount:,.0f}; manager_approval_status={manager_status or 'missing'}",
        )


def _structured_expense(raw: dict[str, Any]) -> dict[str, Any]:
    expense = raw.get("structured_expense")
    return expense if isinstance(expense, dict) else raw


def _compact(value: str, limit: int = 280) -> str:
    collapsed = " ".join(str(value).split())
    return f"{collapsed[: limit - 3]}..." if len(collapsed) > limit else collapsed


def _extract_amount(raw: dict[str, Any]) -> float | None:
    candidates = [
        raw.get("amount_usd"),
        raw.get("amount"),
        raw.get("structured_expense", {}).get("amount_usd") if isinstance(raw.get("structured_expense"), dict) else None,
    ]
    for candidate in candidates:
        if isinstance(candidate, (int, float)):
            return float(candidate)
        if isinstance(candidate, str):
            try:
                return float(candidate.replace("$", "").replace(",", ""))
            except ValueError:
                continue
    return None


def log_run(result: PipelineRun) -> Path:
    runs_dir = Path(__file__).parent / "runs"
    runs_dir.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = runs_dir / f"{stamp}_{result.trace_id}.json"
    path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    return path
