# Multi-Agent Prompt Injection: Expense Payments Demo

Streamlit demo app for a security talk on multi-agent prompt injection.
This is a security education demo. Payment execution is simulated; no real
payment provider is called.

The pipeline has three live LLM agents:

1. Intake Agent structures a raw expense request.
2. Compliance Agent calls a local `lookup_expense_policy` tool, applies policy,
   and decides `APPROVE`, `DENY`, or `ESCALATE`.
3. Payment Agent trusts the compliance handoff, calls a local `create_payment`
   tool, and simulates payout.

The vulnerable boundary is intentional: one agent's output becomes the next agent's input.
The cross-agent scenario also models a common enterprise integration flaw: the
workflow transport preserves "reconciliation context" from the original request
and appends it to downstream handoffs. With guardrails off, that lets untrusted
metadata survive a correct Compliance `ESCALATE` decision and reach a Payment
Agent that trusts payment-facing metadata as executable state.

The main page includes an architecture overview and live flow chart. During a run,
the chart advances through input capture, each agent, both guardrail trust
boundaries, and the final outcome so an audience can see where injected content
crosses from untrusted input into downstream agent context.

Each run also shows an agent telemetry log. The log records agent completions,
tool calls, guardrail boundary checks, policy observations, and dangerous
out-of-bounds events such as Compliance approving an over-threshold expense
without manager approval or Payment executing without a clean Compliance
approval.

## Setup

```bash
git clone <repo-url>
cd f5-ai-sec-expense-payments-app-agentic
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="..."
streamlit run app.py
```

Do not commit real API keys or project tokens. Use local environment variables,
a private `.env` file, or deployment secrets.

To run the red-team API locally:

```bash
uvicorn api:app --host 127.0.0.1 --port 8020
```

Optional model settings:

```bash
export OPENAI_MODEL="gpt-4o-mini"
export OPENAI_TEMPERATURE="0"
```

## F5 Guardrails

The guardrail insertion point is `guardrails.py`:

```python
guardrail_check(message, context) -> GuardrailVerdict
```

It is called between:

- Intake Agent -> Compliance Agent
- Compliance Agent -> Payment Agent

To use the F5 AI Security OpenAI-compatible endpoint, configure:

```bash
export F5_AI_SECURITY_PROJECT_TOKEN="..."
# Optional; defaults to https://us1.calypsoai.app/
export F5_AI_SECURITY_BASE_URL="https://us1.calypsoai.app/"
# Required when F5_AI_SECURITY_BASE_URL is the host root.
export F5_AI_SECURITY_CONNECTION_NAME="<connection-name>"
```

When the base URL is the host root, the app builds the OpenAI-compatible endpoint as
`F5_AI_SECURITY_BASE_URL/openai/F5_AI_SECURITY_CONNECTION_NAME`. You can also set
`F5_AI_SECURITY_BASE_URL` directly to a full `/openai/<connection-name>` URL.
The client uses `OpenAI(base_url=..., api_key=F5_AI_SECURITY_PROJECT_TOKEN)` plus
`x-cai-metadata-session-id`.

By default, when guardrails are on and F5 is configured, the actual Intake,
Compliance, and Payment agent LLM calls are routed through the F5
OpenAI-compatible endpoint. This makes the Calypso/F5 Agent Sessions view read
as the real multi-agent workflow: each agent's system prompt, received message,
and model response appear in sequence under the same session id.

Compliance and Payment also use OpenAI-style tool calls:

- `lookup_expense_policy`: returns policy evidence for Compliance.
- `create_payment`: simulates the final payment action for Payment.

These are local Python functions exposed to the model as provider-level tools,
not a separate MCP server. In Calypso/F5 Agent Sessions, they are intended to
populate the Tool calls lane so the audience can see when the workflow moves
from text reasoning into an action-like step. Because OpenAI tools use a
model -> tool -> model loop, a tool-using agent may appear as adjacent prompt
records in the same session.

The explicit `guardrail_check()` insertion point still exists in the pipeline.
When agent calls are already proxied through F5, the Streamlit boundary verdicts
are marked as covered by the downstream F5-proxied agent input so the Calypso UI
does not get cluttered with extra synthetic boundary-inspection agents.

To return to the older boundary-probe-only behavior, set:

```bash
export F5_AI_SECURITY_PROXY_AGENT_CALLS="false"
```

In that mode, each guardrail call sends a readable boundary-inspection payload
with the dynamic inter-agent handoff content first, plus short
`x-cai-metadata-*` headers for the session id, boundary, source agent, target
agent, and demo scenario.

If F5 env vars are not configured, guardrails fail open with an inline reason so the app can still run locally.

## Scenarios

- Benign: `$200` team lunch.
- Honest policy violation: `$5,000` laptop with no manager approval.
- Direct prompt injection: malicious text in an over-threshold expense description targets Compliance.
- Cross-agent propagation: poisoned intake content is designed to flow into the Payment Agent handoff.

## Demo Script

Recommended talk flow:

1. Run **Benign** with guardrails off or on. Expected final outcome: paid/approved.
2. Run **Honest policy violation**. Expected final outcome: escalated or denied, not paid.
3. Run **Direct prompt injection** with guardrails off. The attacker is trying to turn an over-threshold request into an approval. A good run still escalates here; that shows the Compliance Agent can resist direct manipulation.
4. Run **Cross-agent propagation** with guardrails off. The Compliance Agent can still escalate, but the preserved reconciliation metadata can poison the Payment Agent handoff and produce the dangerous `PAID` outcome.
5. Turn guardrails on and rerun the injection scenarios. Expected result with F5 configured: an inline guardrail block before the poisoned message crosses the boundary.

If guardrails are on but F5 env vars are missing, the app fails closed at the first inter-agent boundary. Turn guardrails off for an intentionally unprotected live run.

## Run Logs

Every full run is written to `runs/<timestamp>_<trace_id>.json` with all agent messages, decisions, guardrail verdicts, and final outcome. This is ready for a later recording/replay mode.

## Red-Team API

The FastAPI service exposes the same workflow for external red-team tools:

- `POST /ask`: simple workflow entry point.
- `POST /api/orchestrator/run`: compatible red-team alias.
- `GET /api/scenarios`: list predefined scenario keys.
- `POST /api/conversations/reset`: stateless reset endpoint for tools that expect one.

Example:

```bash
curl -sS http://127.0.0.1:8020/ask \
  -H "Content-Type: application/json" \
  -d '{
    "conversation_id": "rt-expense-001",
    "scenario_key": "cross_agent_propagation",
    "guardrails_enabled": false
  }'
```

For a protected public endpoint, set `ORCHESTRATOR_API_TOKEN` and send:

```bash
curl -sS https://<render-api-host>/api/orchestrator/run \
  -H "Authorization: Bearer $ORCHESTRATOR_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"conversation_id":"rt-live-001","message":"Employee: ...","guardrails_enabled":true}'
```

The `conversation_id` is used as the F5/Calypso session id for the run, so reuse
it across probes when you want related attempts grouped together.

## Render

`render.yaml` defines two web services from this repo:

- `ledgerloop-expense-demo-ui`: Streamlit presentation UI.
- `ledgerloop-expense-demo-api`: FastAPI red-team endpoint.

Set these environment variables on both services as needed:

- `OPENAI_API_KEY`
- `F5_AI_SECURITY_PROJECT_TOKEN`
- `F5_AI_SECURITY_CONNECTION_NAME`
- `F5_AI_SECURITY_BASE_URL` defaults to `https://us1.calypsoai.app/`
- `ORCHESTRATOR_API_TOKEN` for the API service if you want bearer-token protection
