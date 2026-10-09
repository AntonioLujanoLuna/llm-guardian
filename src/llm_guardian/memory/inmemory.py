"""The in-memory risk memory: deterministic token-set cosine, no ML dependencies.

It implements the design's scoring exactly: ``attack_sim`` is the best cosine to any reviewed
attack, ``benign_sim`` the best to any benign hard negative, and ``margin = attack_sim -
benign_sim``. A benign lookalike that out-scores every attack neighbour yields a *negative* margin,
which is exactly the situation the design wants surfaced rather than hidden.
"""

from __future__ import annotations

import math
import re
import threading
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime

from ..models import IncidentStatus, MemoryEvidence, MemoryStats
from ..provenance import Provenance
from ..settings import MemorySettings
from .base import (
    Neighbor,
    VerifiedExample,
    assert_trusted_label,
    evidence_from,
)

BACKEND = "inmemory"

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenise(text: str) -> frozenset[str]:
    """Lowercased alphanumeric tokens of length >= 2, as a set (the indicator-vector support)."""
    return frozenset(t for t in _TOKEN.findall(text.lower()) if len(t) >= 2)


def cosine(left: frozenset[str], right: frozenset[str]) -> float:
    """Cosine similarity of two token sets: ``|A n B| / sqrt(|A| |B|)``."""
    if not left or not right:
        return 0.0
    intersection = len(left & right)
    if intersection == 0:
        return 0.0
    return intersection / math.sqrt(len(left) * len(right))


class InMemoryRiskMemory:
    """Reviewed precedents kept in lists; similarity is deterministic token-set cosine."""

    backend = BACKEND
    reachable = True
    degraded_reason: str | None = None

    def __init__(
        self,
        settings: MemorySettings,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._default_k = settings.retrieval_k
        self._clock = clock or _utcnow
        self._lock = threading.Lock()
        self._attacks: dict[str, tuple[VerifiedExample, frozenset[str]]] = {}
        self._benign: dict[str, tuple[VerifiedExample, frozenset[str]]] = {}

    def record_verified(self, example: VerifiedExample) -> None:
        assert_trusted_label(example.label)
        entry = (example, tokenise(example.text))
        with self._lock:
            # A re-review supersedes the previous label for the same case: drop it from *both*
            # partitions first, so a stale opposite label (a benign precedent later confirmed
            # malicious, or vice versa) can never be retrieved alongside the current one.
            self._attacks.pop(example.case_id, None)
            self._benign.pop(example.case_id, None)
            target = (
                self._attacks if example.label is IncidentStatus.CONFIRMED_ATTACK else self._benign
            )
            target[example.case_id] = entry

    def query(
        self,
        text: str,
        provenance: Provenance,
        k: int,
        tenant: str | None = None,
    ) -> MemoryEvidence:
        started = time.perf_counter()
        tokens = tokenise(text)
        now = self._clock()
        with self._lock:
            attacks = list(self._attacks.values())
            benign = list(self._benign.values())
        neighbors: list[Neighbor] = []
        expired = 0
        for example, example_tokens in (*attacks, *benign):
            if _has_expired(example, now):
                expired += 1
                continue
            if not _visible(example, tenant):
                continue
            neighbors.append(
                Neighbor(
                    case_id=example.case_id,
                    label=example.label,
                    similarity=cosine(tokens, example_tokens),
                    owasp=example.owasp,
                    family=example.family,
                    digest=example.digest,
                )
            )
        query_ms = (time.perf_counter() - started) * 1000.0
        return evidence_from(
            neighbors, k=k or self._default_k, query_ms=query_ms, expired_examples=expired
        )

    def stats(self) -> MemoryStats:
        with self._lock:
            attacks = len(self._attacks)
            benign = len(self._benign)
        return MemoryStats(
            backend=BACKEND,
            reachable=True,
            confirmed_attacks=attacks,
            benign_hard_negatives=benign,
            pending_incidents=0,
            detail=None,
        )

    def counts(self) -> dict[str, int]:
        with self._lock:
            return {
                "confirmed_attacks": len(self._attacks),
                "benign_hard_negatives": len(self._benign),
                "pending_incidents": 0,
            }

    # -- test/inspection helpers --------------------------------------------------------------
    def examples(self, label: IncidentStatus) -> Sequence[VerifiedExample]:
        with self._lock:
            target = self._attacks if label is IncidentStatus.CONFIRMED_ATTACK else self._benign
            return [example for example, _ in target.values()]


def _visible(example: VerifiedExample, tenant: str | None) -> bool:
    """Global examples (no tenant) are visible to all; a tenant-scoped one only to its tenant."""
    return example.tenant is None or example.tenant == tenant


def _has_expired(example: VerifiedExample, now: datetime) -> bool:
    """A precedent whose retention deadline has passed must not be retrieved."""
    deadline = example.retention_deadline
    return deadline is not None and deadline <= now


def _utcnow() -> datetime:
    return datetime.now(UTC)
