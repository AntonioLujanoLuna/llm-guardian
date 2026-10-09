"""Scan a streamed (server-sent events) chat completion without letting unscanned text out.

The guard sits between the upstream byte stream and the client. It parses each SSE event, adds
any ``delta.content`` to the running text of its choice, and **holds the events back** until a
checkpoint scan has passed:

* every ``checkpoint_chars`` of new text, the tail of each choice (``window_chars``, which always
  covers everything not yet released, plus context before it) is scanned as ``model_output``;
  a pass releases the held events, an enforced block ends the stream with an error event;
* tool-call fragments (``delta.tool_calls``) are assembled per call and judged only once complete,
  so from the first fragment on nothing more is released until the end-of-stream scan, which
  covers every choice's full text and every assembled call (as ``tool_call``).

So the client never receives a byte the guardian has not scanned, at the cost of a release lag of
at most ``checkpoint_chars``. In shadow mode nothing is withheld: after the first non-allow
checkpoint the rest streams straight through and only the final scan runs, so an advisory
response is neither delayed nor recorded once per window.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..models import ScanResponse, SegmentInput, VerdictDecision
from ..provenance import Provenance

ScanFn = Callable[[Sequence[SegmentInput]], Awaitable[ScanResponse]]
#: Authorizes the assembled tool calls; returns a denial message, or ``None`` when all pass.
AuthorizeFn = Callable[[Sequence[tuple[str, Any]]], Awaitable[str | None]]


@dataclass
class _ToolCall:
    name: str = ""
    arguments: str = ""


@dataclass
class _State:
    texts: dict[int, str] = field(default_factory=dict)
    calls: dict[tuple[int, int], _ToolCall] = field(default_factory=dict)
    unscanned: int = 0
    holding_for_tools: bool = False
    passthrough: bool = False


class StreamGuard:
    """Turns an upstream SSE byte stream into a scanned, hold-back stream for the client."""

    def __init__(
        self,
        scan: ScanFn,
        *,
        checkpoint_chars: int,
        window_chars: int,
        authorize: AuthorizeFn | None = None,
        on_event: Callable[[str], None] | None = None,
    ) -> None:
        self._scan = scan
        self._authorize = authorize
        self._on_event = on_event or (lambda event: None)
        self._checkpoint = checkpoint_chars
        self._window = window_chars
        self._state = _State()
        self.final: ScanResponse | None = None
        self.blocked: ScanResponse | None = None

    async def run(self, upstream: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
        pending: list[bytes] = []
        buffer = b""
        async for chunk in upstream:
            buffer += chunk.replace(b"\r\n", b"\n")
            while b"\n\n" in buffer:
                event, buffer = buffer.split(b"\n\n", 1)
                pending.append(event + b"\n\n")
                self._absorb(event)
                if self._state.passthrough:
                    for held in pending:
                        yield held
                    pending.clear()
                elif self._due():
                    verdict = await self._checkpoint_scan()
                    if verdict is not None and _enforced_block(verdict):
                        self.blocked = verdict
                        self._on_event("proxy_stream_blocked")
                        yield _error_event(verdict)
                        return
                    for held in pending:
                        yield held
                    pending.clear()
        if buffer.strip():
            pending.append(buffer if buffer.endswith(b"\n\n") else buffer + b"\n\n")
            self._absorb(buffer)

        verdict = await self._final_scan()
        self.final = verdict
        if verdict is not None and _enforced_block(verdict):
            self.blocked = verdict
            self._on_event("proxy_stream_blocked")
            yield _error_event(verdict)
            return
        if self._authorize is not None and self._state.calls:
            denial = await self._authorize(self.tool_calls())
            enforcing = verdict is not None and not verdict.shadow
            if denial is not None and enforcing:
                self._on_event("proxy_tool_call_denied")
                yield _denial_event(denial)
                return
        for held in pending:
            yield held

    # -- parsing ------------------------------------------------------------------------------
    def _absorb(self, event: bytes) -> None:
        for line in event.split(b"\n"):
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if not payload or payload == b"[DONE]":
                continue
            try:
                data = json.loads(payload)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            choices = data.get("choices") if isinstance(data, dict) else None
            if not isinstance(choices, list):
                continue
            for choice in choices:
                if isinstance(choice, dict):
                    self._absorb_choice(choice)

    def _absorb_choice(self, choice: dict[str, Any]) -> None:
        index = choice.get("index", 0)
        index = index if isinstance(index, int) else 0
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            return
        content = delta.get("content")
        if isinstance(content, str) and content:
            self._state.texts[index] = self._state.texts.get(index, "") + content
            self._state.unscanned += len(content)
        calls = delta.get("tool_calls")
        if isinstance(calls, list):
            for position, call in enumerate(calls):
                if not isinstance(call, dict):
                    continue
                call_index = call.get("index", position)
                key = (index, call_index if isinstance(call_index, int) else position)
                assembled = self._state.calls.setdefault(key, _ToolCall())
                function = call.get("function")
                if isinstance(function, dict):
                    assembled.name += str(function.get("name") or "")
                    assembled.arguments += str(function.get("arguments") or "")
                self._state.holding_for_tools = True

    # -- scanning -----------------------------------------------------------------------------
    def _due(self) -> bool:
        state = self._state
        return not state.holding_for_tools and state.unscanned >= self._checkpoint

    async def _checkpoint_scan(self) -> ScanResponse | None:
        # The window always reaches back past everything not yet released, however large the
        # last event was, plus the configured context before it.
        size = max(self._window, self._state.unscanned + self._window // 2)
        segments = [
            SegmentInput(text=text[-size:], provenance=Provenance.MODEL_OUTPUT)
            for text in self._state.texts.values()
            if text
        ]
        self._state.unscanned = 0
        if not segments:
            return None
        verdict = await self._scan(segments)
        if verdict.shadow and verdict.decision is not VerdictDecision.ALLOW:
            self._state.passthrough = True
        return verdict

    def tool_calls(self) -> list[tuple[str, Any]]:
        """The assembled tool calls, arguments parsed from JSON where they are JSON."""
        calls: list[tuple[str, Any]] = []
        for call in self._state.calls.values():
            arguments: Any = call.arguments
            with contextlib.suppress(json.JSONDecodeError):
                arguments = json.loads(call.arguments)
            calls.append((call.name, arguments))
        return calls

    async def _final_scan(self) -> ScanResponse | None:
        segments = [
            SegmentInput(text=text, provenance=Provenance.MODEL_OUTPUT)
            for text in self._state.texts.values()
            if text
        ]
        for name, arguments in self.tool_calls():
            segments.append(
                SegmentInput(
                    text=json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False),
                    provenance=Provenance.TOOL_CALL,
                )
            )
        if not segments:
            return None
        return await self._scan(segments)


def _enforced_block(verdict: ScanResponse) -> bool:
    return verdict.decision is VerdictDecision.BLOCK and not verdict.shadow


def _denial_event(message: str) -> bytes:
    body = {
        "error": {
            "message": message,
            "type": "guardian_blocked",
            "param": None,
            "code": "tool_call_denied",
        }
    }
    return b"data: " + json.dumps(body).encode("utf-8") + b"\n\n"


def _error_event(verdict: ScanResponse) -> bytes:
    kinds = sorted({kind for segment in verdict.segments for kind in segment.signals.matched_rules})
    body = {
        "error": {
            "message": "The model's response was withheld by the guardian: "
            + (", ".join(kinds) or verdict.decision.value),
            "type": "guardian_blocked",
            "param": None,
            "code": "output_blocked",
        }
    }
    return b"data: " + json.dumps(body).encode("utf-8") + b"\n\n"
