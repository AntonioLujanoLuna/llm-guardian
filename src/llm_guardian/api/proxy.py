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
from collections.abc import Sequence
from typing import Annotated, Any

from fastapi import APIRouter, Header, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool

from ..deps import ScanServiceDep, SettingsDep, UpstreamClientDep
from ..models import ScanRequest, ScanResponse, SegmentInput, VerdictDecision
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
    x_guardian_conversation: Annotated[str | None, Header()] = None,
    x_guardian_tenant: Annotated[str | None, Header()] = None,
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
            return _error(
                403,
                "guardian_blocked",
                "The request was blocked by the guardian: " + _reasons(verdict),
                "input_blocked",
                headers,
            )

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
            return _error(
                403,
                "guardian_blocked",
                "The model's response was withheld by the guardian: " + _reasons(verdict),
                "output_blocked",
                headers,
            )
    return JSONResponse(data, status_code=upstream.status_code, headers=headers)


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
        arguments: Any = function.get("arguments", "")
        if isinstance(arguments, str):
            with contextlib.suppress(json.JSONDecodeError):
                arguments = json.loads(arguments)
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
