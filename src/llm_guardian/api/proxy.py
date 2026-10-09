"""An OpenAI-compatible ``/chat/completions`` proxy with the guardian in the path.

Point a client's ``base_url`` at ``http://<guardian>/openai/v1`` and every request is scanned on
the way in, forwarded to ``proxy.upstream_url`` (llm-d, vLLM or any OpenAI-compatible server), and
-- for non-streaming responses -- scanned again on the way out. In enforce mode a ``block`` is
returned as an OpenAI-shaped error (HTTP 403); in shadow mode everything is forwarded and the
verdict travels in ``X-Guardian-*`` response headers.

What it trusts, stated plainly: message roles are client-supplied, and the design says a role
label is not provenance. The proxy is meant for deployments where the *agent runtime* builds the
message list (so a ``tool`` message really is a tool result), and it never maps a role to more
trust than a user turn has: ``system``/``developer``/``user`` -> ``user``, ``tool``/``function`` ->
``tool_result``, ``assistant`` text -> ``model_output``, ``assistant`` tool-call arguments ->
``tool_call``. A gateway that knows better should call ``/v1/scan`` itself.

Only the *tail* of the conversation is scanned on the way in -- the messages after the last
assistant turn, i.e. what is new since the model last spoke. Earlier turns were scanned when they
were new (and assistant turns on the way out), and rescanning the whole history on every request
would make one weak signal count once per request in the per-conversation risk.

Streaming responses are scanned incrementally by :class:`~.stream_guard.StreamGuard`
(``X-Guardian-Output: scanned_incrementally``): events are held back until a checkpoint scan
passes, so no unscanned text reaches the client, and an enforced block ends the stream with an
OpenAI-shaped error event. ``proxy.scan_streams=false`` restores plain pass-through.
"""

from __future__ import annotations

import contextlib
import json
import uuid
from collections.abc import Callable, Sequence
from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Header, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool

from ..deps import (
    ClockDep,
    MetricsDep,
    PolicyEngineDep,
    ScanServiceDep,
    SettingsDep,
    UpstreamClientDep,
)
from ..models import (
    EvidenceRef,
    PolicyAction,
    PolicyCheck,
    PolicyCheckRequest,
    ScanRequest,
    ScanResponse,
    SegmentInput,
    VerdictDecision,
)
from ..policy import PolicyEngine
from ..provenance import Provenance
from .stream_guard import StreamGuard

router = APIRouter(tags=["proxy"])

#: Client role -> the provenance the proxy assigns. Nothing maps above a user turn.
ROLE_PROVENANCE: dict[str, Provenance] = {
    "system": Provenance.USER,
    "developer": Provenance.USER,
    "user": Provenance.USER,
    "tool": Provenance.TOOL_RESULT,
    "function": Provenance.TOOL_RESULT,
    "assistant": Provenance.MODEL_OUTPUT,
}


@router.post("/chat/completions", response_model=None)
async def chat_completions(
    request: Request,
    service: ScanServiceDep,
    settings: SettingsDep,
    client: UpstreamClientDep,
    engine: PolicyEngineDep,
    clock: ClockDep,
    metrics: MetricsDep,
    x_guardian_conversation: Annotated[str | None, Header()] = None,
    x_guardian_tenant: Annotated[str | None, Header()] = None,
    x_guardian_principal: Annotated[str | None, Header()] = None,
) -> Response:
    """Scan, forward, scan the answer, return it -- or an OpenAI-shaped error on a block."""
    try:
        body = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        return _error(400, "invalid_request_error", "request body is not JSON", "invalid_json")
    if not isinstance(body, dict) or not isinstance(body.get("messages"), list):
        return _error(400, "invalid_request_error", "'messages' must be a list", "invalid_body")

    conversation = x_guardian_conversation
    tenant = x_guardian_tenant or settings.default_tenant
    # One id for the whole turn: the output scan is the same turn as the input scan, so the
    # per-conversation risk counts it once instead of decaying through a second, clean "turn".
    turn_id = uuid.uuid4().hex
    headers: dict[str, str] = {"X-Guardian-Request-Id": turn_id}

    principal = settings.proxy.principal
    if settings.proxy.trust_principal_header and x_guardian_principal:
        principal = x_guardian_principal
    evidence: tuple[EvidenceRef, ...] = ()

    inbound = segments_from_messages(new_messages(body["messages"]))
    if inbound:
        verdict = await run_in_threadpool(
            service.scan,
            ScanRequest(
                request_id=turn_id,
                conversation_id=conversation,
                tenant=tenant,
                segments=tuple(inbound),
            ),
        )
        headers.update(_verdict_headers("Input", verdict))
        if _enforced_block(verdict):
            metrics.count_event("proxy_input_blocked")
            return _error(
                403,
                "guardian_blocked",
                "The request was blocked by the guardian: " + _reasons(verdict),
                "input_blocked",
                headers,
            )
        evidence = evidence_from(verdict)

    def authorize_now(calls: Sequence[tuple[str, Any]]) -> PolicyCheck | None:
        return first_denial(engine, principal or "", tenant, calls, evidence, clock)

    async def authorize(calls: Sequence[tuple[str, Any]]) -> str | None:
        denied = authorize_now(calls)
        return None if denied is None else _denial_message(denied)

    upstream_headers = {"content-type": "application/json"}
    if settings.proxy.upstream_api_key is not None:
        upstream_headers["authorization"] = (
            f"Bearer {settings.proxy.upstream_api_key.get_secret_value()}"
        )
    elif "authorization" in request.headers:
        upstream_headers["authorization"] = request.headers["authorization"]

    async def scan_output(segments: Sequence[SegmentInput]) -> ScanResponse:
        return await run_in_threadpool(
            service.scan,
            ScanRequest(
                request_id=turn_id,
                conversation_id=conversation,
                tenant=tenant,
                segments=tuple(segments),
            ),
        )

    if body.get("stream"):
        upstream = await client.send(
            client.build_request("POST", "chat/completions", json=body, headers=upstream_headers),
            stream=True,
        )
        media_type = upstream.headers.get("content-type", "text/event-stream")
        if upstream.status_code >= 400 or not settings.proxy.scan_streams:
            headers["X-Guardian-Output"] = "not_scanned"
            stream = upstream.aiter_raw()
        else:
            headers["X-Guardian-Output"] = "scanned_incrementally"
            guard = StreamGuard(
                scan_output,
                checkpoint_chars=settings.proxy.stream_checkpoint_chars,
                window_chars=settings.proxy.stream_window_chars,
                authorize=authorize if settings.proxy.authorize_tool_calls else None,
                on_event=metrics.count_event,
            )
            stream = guard.run(upstream.aiter_raw())
        return StreamingResponse(
            stream,
            status_code=upstream.status_code,
            media_type=media_type,
            headers=headers,
            background=BackgroundTask(upstream.aclose),
        )

    upstream = await client.post("chat/completions", json=body, headers=upstream_headers)
    try:
        data = upstream.json()
    except ValueError:
        data = None
    if upstream.status_code >= 400 or not isinstance(data, dict):
        headers["X-Guardian-Output"] = "not_scanned"
        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type"),
            headers=headers,
        )

    outbound = segments_from_choices(data.get("choices"))
    if outbound:
        verdict = await scan_output(outbound)
        headers.update(_verdict_headers("Output", verdict))
        if _enforced_block(verdict):
            metrics.count_event("proxy_output_blocked")
            return _error(
                403,
                "guardian_blocked",
                "The model's response was withheld by the guardian: " + _reasons(verdict),
                "output_blocked",
                headers,
            )
        calls = tool_calls_from_choices(data.get("choices"))
        if settings.proxy.authorize_tool_calls and calls:
            denied = authorize_now(calls)
            headers["X-Guardian-Tool-Policy"] = (
                "allow" if denied is None else f"deny:{denied.denied_by}"
            )
            if denied is not None and not verdict.shadow:
                metrics.count_event("proxy_tool_call_denied")
                return _error(
                    403, "guardian_blocked", _denial_message(denied), "tool_call_denied", headers
                )
    return JSONResponse(data, status_code=upstream.status_code, headers=headers)


# -- tool-call authorization --------------------------------------------------------------------
def evidence_from(verdict: ScanResponse) -> tuple[EvidenceRef, ...]:
    """Every finding of this turn's input scan, as the evidence a tool proposal carries."""
    refs: dict[tuple[str, str], EvidenceRef] = {}
    for segment in verdict.segments:
        for outcome in segment.scanners:
            for finding in outcome.findings:
                refs.setdefault(
                    (segment.segment_id, finding.kind),
                    EvidenceRef(
                        segment_id=segment.segment_id,
                        finding_kind=finding.kind,
                        severity=finding.severity,
                        provenance=segment.provenance,
                    ),
                )
    return tuple(refs.values())


def first_denial(
    engine: PolicyEngine,
    principal: str,
    tenant: str | None,
    calls: Sequence[tuple[str, Any]],
    evidence: Sequence[EvidenceRef],
    clock: Callable[[], datetime],
) -> PolicyCheck | None:
    """Authorize each proposed call; the first denial, or ``None`` when every call is allowed."""
    for name, arguments in calls:
        check = engine.check(
            PolicyCheckRequest(
                principal=principal,
                tenant=tenant,
                action=PolicyAction.TOOL_CALL,
                tool=name or None,
                tool_arguments=_as_arguments(arguments),
                evidence=tuple(evidence),
            ),
            now=clock,
        )
        if not check.allowed:
            return check.model_copy(update={"reason_codes": (f"tool:{name}", *check.reason_codes)})
    return None


def tool_calls_from_choices(choices: Any) -> list[tuple[str, Any]]:
    calls: list[tuple[str, Any]] = []
    if not isinstance(choices, list):
        return calls
    for choice in choices:
        message = choice.get("message") if isinstance(choice, dict) else None
        raw = message.get("tool_calls") if isinstance(message, dict) else None
        for call in raw if isinstance(raw, list) else []:
            function = call.get("function") if isinstance(call, dict) else None
            if isinstance(function, dict):
                calls.append((str(function.get("name", "")), _parsed(function.get("arguments"))))
    return calls


def _parsed(arguments: Any) -> Any:
    if isinstance(arguments, str):
        with contextlib.suppress(json.JSONDecodeError):
            return json.loads(arguments)
    return arguments


def _as_arguments(arguments: Any) -> dict[str, Any]:
    """Policy argument checks need a mapping; anything else is checked as one opaque value."""
    if isinstance(arguments, dict):
        return arguments
    if arguments in (None, ""):
        return {}
    return {"arguments": arguments}


def _denial_message(check: PolicyCheck) -> str:
    tool = next((code[5:] for code in check.reason_codes if code.startswith("tool:")), "")
    detail = ", ".join(check.argument_failures or check.evidence_used) or check.denied_by or ""
    return f"The proposed tool call {tool!r} was denied by policy: {detail}"


# -- message mapping ----------------------------------------------------------------------------
def new_messages(messages: Sequence[Any]) -> list[Any]:
    """The messages after the last assistant turn: what is new since the model last spoke."""
    for position in range(len(messages) - 1, -1, -1):
        message = messages[position]
        if isinstance(message, dict) and message.get("role") == "assistant":
            return list(messages[position + 1 :])
    return list(messages)


def segments_from_messages(messages: Sequence[Any]) -> list[SegmentInput]:
    segments: list[SegmentInput] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "user"))
        provenance = ROLE_PROVENANCE.get(role, Provenance.TOOL_RESULT)
        text = _text_of(message.get("content"))
        if text:
            segments.append(SegmentInput(text=text, provenance=provenance))
        segments.extend(_tool_call_segments(message))
    return segments


def segments_from_choices(choices: Any) -> list[SegmentInput]:
    if not isinstance(choices, list):
        return []
    segments: list[SegmentInput] = []
    for choice in choices:
        message = choice.get("message") if isinstance(choice, dict) else None
        if not isinstance(message, dict):
            continue
        text = _text_of(message.get("content"))
        if text:
            segments.append(SegmentInput(text=text, provenance=Provenance.MODEL_OUTPUT))
        segments.extend(_tool_call_segments(message))
    return segments


def _tool_call_segments(message: dict[str, Any]) -> list[SegmentInput]:
    calls = message.get("tool_calls")
    if not isinstance(calls, list):
        return []
    segments: list[SegmentInput] = []
    for call in calls:
        function = call.get("function") if isinstance(call, dict) else None
        if not isinstance(function, dict):
            continue
        # Rendered as a JSON object -- the shape the rules recognise as structure, so the JSON
        # quoting around an argument is not mistaken for "the author is quoting an attack".
        arguments = _parsed(function.get("arguments", ""))
        text = json.dumps(
            {"name": function.get("name", ""), "arguments": arguments}, ensure_ascii=False
        )
        segments.append(SegmentInput(text=text, provenance=Provenance.TOOL_CALL))
    return segments


def _text_of(content: Any) -> str:
    """Plain text of a message ``content``: a string, or the text parts of a part list."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") in {"text", "input_text"}
        ]
        return "\n".join(part for part in parts if part)
    return ""


# -- responses ----------------------------------------------------------------------------------
def _enforced_block(verdict: ScanResponse) -> bool:
    return verdict.decision is VerdictDecision.BLOCK and not verdict.shadow


def _verdict_headers(stage: str, verdict: ScanResponse) -> dict[str, str]:
    return {
        f"X-Guardian-{stage}-Decision": verdict.decision.value,
        "X-Guardian-Shadow": str(verdict.shadow).lower(),
    }


def _reasons(verdict: ScanResponse) -> str:
    kinds = sorted({kind for segment in verdict.segments for kind in segment.signals.matched_rules})
    return ", ".join(kinds) or verdict.decision.value


def _error(
    status: int,
    kind: str,
    message: str,
    code: str,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    return JSONResponse(
        {"error": {"message": message, "type": kind, "param": None, "code": code}},
        status_code=status,
        headers=headers,
    )
