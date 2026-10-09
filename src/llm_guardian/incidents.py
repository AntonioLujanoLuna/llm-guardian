"""The incident lifecycle: an untrusted alert becomes a trusted label only through review.

``create`` records a *pending* incident, which is an observation and never a label. ``review`` is
the only method that can promote a case into the risk memory, and it records the reviewer, the
timestamp and the note as an audit entry. A ``dismissed`` case closes without promoting anything.
``settings.memory.auto_promote`` is frozen off, and the store refuses to be constructed at all if
it is not false -- promotion without a human being is structurally impossible, not merely
discouraged.

Two bounds keep the store from growing without limit:

* A repeat sighting of the same ``(tenant, provenance, content digest)`` is folded into the
  existing case (``occurrences`` and ``last_seen_at``) rather than opening a new one, so a client
  retry loop cannot flood the review queue. The fold never changes a case's status or label.
* A case past its ``retention_deadline`` is purged on the next write.

Persistence is a :class:`IncidentRepository`: in-process by default, SQLite when configured
(``incidents_sqlite.py``). With a persistent repository the reviewed cases are the source of truth
for the risk memory, so an in-process memory is rebuilt from them at startup instead of starting
empty after every restart.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol

from .memory import VerifiedExample
from .memory.base import TRUSTED_LABELS, RiskMemory
from .models import (
    IncidentCreateRequest,
    IncidentRecord,
    IncidentStatus,
    ReviewRequest,
)
from .provenance import Provenance
from .settings import GuardianSettings


class NoAutoPromotionError(RuntimeError):
    """Raised if anything tries to enable automatic promotion of incidents."""


def dedup_key(tenant: str | None, provenance: Provenance, content_digest: str) -> str:
    """The identity a repeat sighting is folded under. Tenants never share a case."""
    return f"{tenant or ''}\x1f{provenance.value}\x1f{content_digest}"


@dataclass(frozen=True)
class IncidentFilter:
    """Which incidents a listing returns. ``None`` on a field means "any"."""

    status: IncidentStatus | None = None
    tenant: str | None = None
    provenance: Provenance | None = None

    def matches(self, record: IncidentRecord) -> bool:
        return (
            (self.status is None or record.status is self.status)
            and (self.tenant is None or record.tenant == self.tenant)
            and (self.provenance is None or record.provenance is self.provenance)
        )


class IncidentRepository(Protocol):
    """Storage for incident records and the review audit trail. No lifecycle logic lives here."""

    def get(self, case_id: str) -> IncidentRecord | None: ...

    def find(self, key: str) -> IncidentRecord | None: ...

    def put(self, record: IncidentRecord) -> None: ...

    def records(
        self,
        where: IncidentFilter | None = None,
        *,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[IncidentRecord]:
        """Matching records, oldest first (the review queue's order)."""
        ...

    def count(self, where: IncidentFilter | None = None) -> int: ...

    def pending_count(self) -> int: ...

    def purge_expired(self, now: datetime) -> int: ...

    def append_audit(self, entry: dict[str, str]) -> None: ...

    def audit_entries(self) -> list[dict[str, str]]: ...


class InMemoryIncidentRepository:
    """The default repository: dictionaries, lost on restart."""

    def __init__(self) -> None:
        self._by_id: dict[str, IncidentRecord] = {}
        self._by_key: dict[str, str] = {}
        self._audit: list[dict[str, str]] = []

    def get(self, case_id: str) -> IncidentRecord | None:
        return self._by_id.get(case_id)

    def find(self, key: str) -> IncidentRecord | None:
        case_id = self._by_key.get(key)
        return None if case_id is None else self._by_id.get(case_id)

    def put(self, record: IncidentRecord) -> None:
        self._by_id[record.case_id] = record
        self._by_key[_key_of(record)] = record.case_id

    def records(
        self,
        where: IncidentFilter | None = None,
        *,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[IncidentRecord]:
        where = where or IncidentFilter()
        records = sorted(
            (record for record in self._by_id.values() if where.matches(record)),
            key=lambda record: (record.created_at, record.case_id),
        )
        end = None if limit is None else offset + limit
        return records[offset:end]

    def count(self, where: IncidentFilter | None = None) -> int:
        where = where or IncidentFilter()
        return sum(1 for record in self._by_id.values() if where.matches(record))

    def pending_count(self) -> int:
        return sum(1 for r in self._by_id.values() if r.status is IncidentStatus.PENDING)

    def purge_expired(self, now: datetime) -> int:
        expired = [r for r in self._by_id.values() if r.retention_deadline <= now]
        for record in expired:
            del self._by_id[record.case_id]
            key = _key_of(record)
            if self._by_key.get(key) == record.case_id:
                del self._by_key[key]
        return len(expired)

    def append_audit(self, entry: dict[str, str]) -> None:
        self._audit.append(dict(entry))

    def audit_entries(self) -> list[dict[str, str]]:
        return [dict(entry) for entry in self._audit]


class IncidentStore:
    """The incident lifecycle over a pluggable repository."""

    def __init__(
        self,
        settings: GuardianSettings,
        memory: RiskMemory,
        clock: Callable[[], datetime] | None = None,
        repository: IncidentRepository | None = None,
    ) -> None:
        if settings.memory.auto_promote:
            raise NoAutoPromotionError(
                "auto_promote must be false: a verified label requires an independent review."
            )
        self._settings = settings
        self._memory = memory
        self._clock = clock or _utcnow
        self._lock = threading.Lock()
        self._repo: IncidentRepository = repository or InMemoryIncidentRepository()
        if repository is not None and memory.backend == "inmemory":
            self._rehydrate_memory()

    # -- create / read ------------------------------------------------------------------------
    def create(self, request: IncidentCreateRequest) -> IncidentRecord:
        """Record a pending incident, or fold a repeat sighting into its existing case.

        Nothing here is a label and nothing is promoted.
        """
        now = self._clock()
        key = dedup_key(request.tenant, request.provenance, request.content_digest)
        with self._lock:
            self._repo.purge_expired(now)
            existing = self._repo.find(key)
            if existing is not None:
                folded = existing.model_copy(
                    update={"occurrences": existing.occurrences + 1, "last_seen_at": now}
                )
                self._repo.put(folded)
                return folded
            record = IncidentRecord(
                case_id=uuid.uuid4().hex,
                created_at=now,
                conversation_id=request.conversation_id,
                tenant=request.tenant,
                provenance=request.provenance,
                content_digest=request.content_digest,
                redacted_excerpt=request.redacted_excerpt,
                findings_summary=request.findings_summary,
                owasp=request.owasp,
                detector_versions=request.detector_versions,
                score=request.score,
                status=IncidentStatus.PENDING,
                last_seen_at=now,
                retention_deadline=now + timedelta(days=request.retention_days),
            )
            self._repo.put(record)
        return record

    def get(self, case_id: str) -> IncidentRecord | None:
        with self._lock:
            return self._repo.get(case_id)

    def list(
        self,
        status: IncidentStatus | None = None,
        *,
        tenant: str | None = None,
        provenance: Provenance | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[IncidentRecord]:
        where = IncidentFilter(status=status, tenant=tenant, provenance=provenance)
        with self._lock:
            return self._repo.records(where, limit=limit, offset=offset)

    def count(
        self,
        status: IncidentStatus | None = None,
        *,
        tenant: str | None = None,
        provenance: Provenance | None = None,
    ) -> int:
        where = IncidentFilter(status=status, tenant=tenant, provenance=provenance)
        with self._lock:
            return self._repo.count(where)

    def pending_count(self) -> int:
        with self._lock:
            return self._repo.pending_count()

    def reviewed_benign(
        self, tenant: str | None, provenance: Provenance, content_digest: str
    ) -> IncidentRecord | None:
        """The still-retained case a human reviewed as benign for exactly these bytes, if any."""
        with self._lock:
            record = self._repo.find(dedup_key(tenant, provenance, content_digest))
        if record is None or record.status is not IncidentStatus.BENIGN:
            return None
        if record.retention_deadline <= self._clock():
            return None
        return record

    def purge_expired(self) -> int:
        """Drop every case past its retention deadline; returns how many were dropped."""
        with self._lock:
            return self._repo.purge_expired(self._clock())

    # -- review: the only path to a trusted label ---------------------------------------------
    def review(self, case_id: str, request: ReviewRequest) -> IncidentRecord:
        with self._lock:
            record = self._repo.get(case_id)
            if record is None:
                raise KeyError(case_id)
            now = self._clock()
            updated = record.model_copy(
                update={
                    "status": request.label,
                    "reviewed_by": request.reviewer,
                    "reviewed_at": now,
                    "review_note": request.note,
                    "family": request.family or record.family,
                    "owasp": request.owasp or record.owasp,
                }
            )
            self._repo.put(updated)
            self._repo.append_audit(
                {
                    "case_id": case_id,
                    "reviewer": request.reviewer,
                    "label": request.label.value,
                    "at": now.isoformat(),
                    "note": request.note or "",
                }
            )
        if request.label in TRUSTED_LABELS:
            self._memory.record_verified(_example_of(updated))
        return updated

    def audit_entries(self) -> Sequence[dict[str, str]]:
        with self._lock:
            return self._repo.audit_entries()

    # -- startup ------------------------------------------------------------------------------
    def _rehydrate_memory(self) -> None:
        """Re-promote every still-retained reviewed case into a memory that starts empty."""
        with self._lock:
            self._repo.purge_expired(self._clock())
            reviewed = [r for r in self._repo.records() if r.status in TRUSTED_LABELS]
        for record in reviewed:
            self._memory.record_verified(_example_of(record))


def _example_of(record: IncidentRecord) -> VerifiedExample:
    return VerifiedExample(
        case_id=record.case_id,
        label=record.status,
        text=record.redacted_excerpt,
        digest=record.content_digest,
        owasp=record.owasp,
        family=record.family,
        tenant=record.tenant,
        retention_deadline=record.retention_deadline,
    )


def _key_of(record: IncidentRecord) -> str:
    return dedup_key(record.tenant, record.provenance, record.content_digest)


def _utcnow() -> datetime:
    return datetime.now(UTC)
