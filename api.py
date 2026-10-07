from __future__ import annotations

import os
from dataclasses import asdict
from uuid import uuid4

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from pipeline import run_pipeline
from scenarios import SCENARIOS, scenario_by_key


app = FastAPI(
    title="LedgerLoop Expense Payments Demo API",
    version="0.1.0",
    description="Red-teamable API for the multi-agent expense/payment prompt-injection demo.",
)


class AskRequest(BaseModel):
    message: str | None = Field(
        default=None,
        description="Raw expense/payment request to send into the workflow.",
    )
    scenario_key: str | None = Field(
        default=None,
        description="Optional predefined scenario key. Ignored when message is provided.",
    )
    conversation_id: str | None = Field(
        default=None,
        description="Stable id reused across red-team probes and F5 session telemetry.",
    )
    guardrails_enabled: bool = Field(default=True)


class ResetRequest(BaseModel):
    conversation_id: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/scenarios")
def list_scenarios() -> list[dict[str, str]]:
    return [{"key": scenario.key, "label": scenario.label} for scenario in SCENARIOS]


@app.post("/ask")
def ask(
    payload: AskRequest,
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None),
) -> dict[str, object]:
    _require_api_token(authorization=authorization, x_api_token=x_api_token)
    raw_request, scenario_name = _request_text(payload)
    conversation_id = payload.conversation_id or f"rt-{uuid4().hex[:12]}"
    result = run_pipeline(
        raw_request,
        guardrails_enabled=payload.guardrails_enabled,
        scenario_name=scenario_name,
        conversation_id=conversation_id,
    )
    body = asdict(result)
    body["conversation_id"] = conversation_id
    body["answer"] = _answer(result.final_outcome, result.final_amount_usd, result.blocked_reason)
    return body


@app.post("/api/orchestrator/run")
def run_orchestrator(
    payload: AskRequest,
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None),
) -> dict[str, object]:
    return ask(payload, authorization=authorization, x_api_token=x_api_token)


@app.post("/api/conversations/reset")
def reset_conversation(payload: ResetRequest) -> dict[str, object]:
    return {
        "conversation_id": payload.conversation_id,
        "cleared": True,
        "note": "This workflow is stateless; reuse or discard the conversation_id as needed.",
    }


def _request_text(payload: AskRequest) -> tuple[str, str]:
    if payload.message and payload.message.strip():
        return payload.message.strip(), "Custom API request"
    if payload.scenario_key:
        scenario = scenario_by_key(payload.scenario_key)
        return scenario.request, scenario.label
    scenario = scenario_by_key("benign")
    return scenario.request, scenario.label


def _require_api_token(*, authorization: str | None, x_api_token: str | None) -> None:
    expected = (os.getenv("ORCHESTRATOR_API_TOKEN") or "").strip()
    if not expected:
        return
    bearer = ""
    if authorization and authorization.lower().startswith("bearer "):
        bearer = authorization.split(" ", 1)[1].strip()
    if expected not in {bearer, (x_api_token or "").strip()}:
        raise HTTPException(status_code=401, detail="Missing or invalid API token.")


def _answer(outcome: str, amount: float | None, blocked_reason: str | None) -> str:
    amount_label = f"${amount:,.0f}" if amount is not None else "the request"
    if outcome == "PAID":
        return f"Payment executed for {amount_label}."
    if outcome in {"APPROVED", "APPROVE"}:
        return f"Expense approved for {amount_label}."
    if outcome == "BLOCKED_BY_GUARDRAIL":
        return f"Blocked by guardrail: {blocked_reason or 'policy violation detected'}"
    if outcome == "ESCALATE":
        return f"Escalated for manager approval: {amount_label}."
    if outcome == "DENY":
        return f"Denied: {amount_label}."
    return f"Workflow completed with outcome {outcome}: {amount_label}."
