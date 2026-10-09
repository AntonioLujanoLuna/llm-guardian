"""Per-conversation risk that accumulates across turns.

A gradual attack spreads itself over several requests so that no single one crosses a threshold:
a little role-play here, a hint at hidden instructions there. Each request is judged on its own,
so each one is allowed. This tracker keeps a decayed running sum of each request's worst segment
score, per ``(tenant, conversation_id)``::

    risk = previous_risk * decay + contribution

where ``contribution`` is the request's worst score when it is at least ``min_contribution`` and
zero otherwise. When the running risk reaches ``escalate_at`` on a request that would otherwise be
allowed, the service lifts that request's most suspicious segment to *escalate*. It never blocks:
accumulated weak evidence is a reason for a human to look, not for a hard stop.

A retry (same ``request_id``) does not count twice, and the table is bounded by capacity and an
idle TTL, so an attacker minting conversation ids can only evict, never grow, it.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

from .settings import SessionSettings


@dataclass
class _State:
    risk: float
    touched: float
    last_request_id: str | None


class SessionRiskTracker:
    """A bounded, thread-safe table of decayed per-conversation risk."""

    def __init__(
        self, settings: SessionSettings, *, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self._settings = settings
        self._clock = clock
        self._lock = threading.Lock()
        self._states: OrderedDict[tuple[str, str], _State] = OrderedDict()

    @property
    def settings(self) -> SessionSettings:
        return self._settings

    def observe(
        self,
        tenant: str | None,
        conversation_id: str,
        score: float,
        request_id: str | None = None,
    ) -> float:
        """Fold one request's worst score into the conversation's risk and return the new risk."""
        key = (tenant or "", conversation_id)
        contribution = score if score >= self._settings.min_contribution else 0.0
        now = self._clock()
        with self._lock:
            state = self._states.get(key)
            if state is not None and now - state.touched > self._settings.idle_ttl_seconds:
                state = None
            if state is None:
                state = _State(risk=0.0, touched=now, last_request_id=None)
            elif request_id is not None and request_id == state.last_request_id:
                return state.risk  # a retry of the same request is not new evidence
            state.risk = state.risk * self._settings.decay + contribution
            state.touched = now
            state.last_request_id = request_id
            self._states[key] = state
            self._states.move_to_end(key)
            while len(self._states) > self._settings.max_sessions:
                self._states.popitem(last=False)
            return state.risk

    def risk(self, tenant: str | None, conversation_id: str) -> float:
        with self._lock:
            state = self._states.get((tenant or "", conversation_id))
            return 0.0 if state is None else state.risk

    def __len__(self) -> int:
        with self._lock:
            return len(self._states)
