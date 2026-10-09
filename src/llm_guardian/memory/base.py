"""The risk-memory contract.

Risk memory holds *reviewed* precedents only: confirmed attacks and benign hard negatives, in
separately permissioned partitions. It answers with evidence -- a similarity and a margin -- and
never with a verdict. The pipeline consults it only inside the uncertain band, and a cosine
similarity never whitelists content.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from ..models import (
    IncidentStatus,
    MemoryEvidence,
    MemoryNeighbor,
    MemoryStats,
    OwaspCategory,
)
from ..provenance import Provenance


class MemoryBackendUnavailable(RuntimeError):
    """Raised when a risk-memory backend cannot be constructed (missing extra, bad config)."""


class VerifiedExample(BaseModel):
    """A reviewed precedent. Only ``confirmed_attack`` and ``benign`` labels are accepted."""

    model_config = ConfigDict(frozen=True)

    case_id: str
    label: IncidentStatus
    text: str
    digest: str
    owasp: tuple[OwaspCategory, ...] = ()
    family: str | None = None
    tenant: str | None = None
    #: Carried from the reviewed incident. ``None`` means the precedent never expires; a past
    #: deadline means a query must drop it rather than keep applying a stale review.
    retention_deadline: datetime | None = None


class RiskMemory(Protocol):
    """Query reviewed precedents, promote reviewed evidence, report partition sizes."""

    backend: str
    reachable: bool
    degraded_reason: str | None

    def query(
        self,
        text: str,
        provenance: Provenance,
        k: int,
        tenant: str | None = None,
    ) -> MemoryEvidence: ...

    def record_verified(self, example: VerifiedExample) -> None: ...

    def stats(self) -> MemoryStats: ...

    def counts(self) -> dict[str, int]: ...


TRUSTED_LABELS: frozenset[IncidentStatus] = frozenset(
    {IncidentStatus.CONFIRMED_ATTACK, IncidentStatus.BENIGN}
)


def assert_trusted_label(label: IncidentStatus) -> IncidentStatus:
    """Guard against promoting an unreviewed or dismissed case into trusted memory."""
    if label not in TRUSTED_LABELS:
        raise ValueError(
            f"{label.value!r} is not a trusted label; only confirmed_attack or benign may be "
            "promoted into the risk memory."
        )
    return label


class Neighbor(BaseModel):
    """Internal scored neighbour before it becomes a wire :class:`MemoryNeighbor`."""

    model_config = ConfigDict(frozen=True)

    case_id: str
    label: IncidentStatus
    similarity: float
    owasp: tuple[OwaspCategory, ...] = ()
    family: str | None = None
    digest: str


def evidence_from(
    neighbors: list[Neighbor],
    *,
    k: int,
    query_ms: float,
    reason_not_queried: str | None = None,
    queried: bool = True,
    expired_examples: int = 0,
) -> MemoryEvidence:
    """Fold scored neighbours into a :class:`MemoryEvidence` using the design's formula."""
    attacks = [n.similarity for n in neighbors if n.label is IncidentStatus.CONFIRMED_ATTACK]
    benign = [n.similarity for n in neighbors if n.label is IncidentStatus.BENIGN]
    attack_sim = max(attacks, default=0.0)
    benign_sim = max(benign, default=0.0)
    ranked = sorted(neighbors, key=lambda n: (-n.similarity, n.case_id))[:k]
    wire = tuple(
        MemoryNeighbor(
            case_id=n.case_id,
            label=n.label,
            similarity=round(n.similarity, 6),
            owasp=n.owasp,
            family=n.family,
            digest=n.digest,
        )
        for n in ranked
    )
    return MemoryEvidence(
        queried=queried,
        reason_not_queried=reason_not_queried,
        attack_sim=round(attack_sim, 6) if queried else None,
        benign_sim=round(benign_sim, 6) if queried else None,
        margin=round(attack_sim - benign_sim, 6) if queried else None,
        neighbors=wire,
        query_ms=query_ms,
        expired_examples=expired_examples,
    )


def not_queried(reason: str) -> MemoryEvidence:
    """The normal-path result: retrieval did not happen, and here is why."""
    return MemoryEvidence(queried=False, reason_not_queried=reason, neighbors=())
