"""The incident lifecycle: an untrusted alert becomes a trusted label only through review.

``create`` records a *pending* incident, which is an observation and never a label. ``review`` is
the only method that can promote a case into the risk memory, and it records the reviewer, the
timestamp and the note as an audit entry. A ``dismissed`` case closes without promoting anything.
``settings.memory.auto_promote`` is frozen off, and the store refuses to be constructed at all if
it is not false -- promotion without a human being is structurally impossible, not merely
discouraged.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta

from .memory import VerifiedExample
from .memory.base import TRUSTED_LABELS, RiskMemory
from .models import (
    IncidentCreateRequest,
    IncidentRecord,
    IncidentStatus,
    ReviewRequest,
)
from .settings import GuardianSettings


class NoAutoPromotionError(RuntimeError):
    """Raised if anything tries to enable automatic promotion of incidents."""


class IncidentStore:
    """An in-process store of incidents and their review audit trail."""

    def __init__(
        self,
        settings: GuardianSettings,
        memory: RiskMemory,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if settings.memory.auto_promote:
            raise NoAutoPromotionError(
                "auto_promote must be false: a verified label requires an independent review."
            )
        self._settings = settings
        self._memory = memory
        self._clock = clock or _utcnow
        self._lock = threading.Lock()
        self._records: dict[str, IncidentRecord] = {}
        self._audit: list[dict[str, str]] = []

    # -- create / read ------------------------------------------------------------------------
    def create(self, request: IncidentCreateRequest) -> IncidentRecord:
        """Record a pending incident. Nothing here is a label and nothing is promoted."""
        now = self._clock()
        record = IncidentRecord(
            case_id=uuid.uuid4().hex,
            created_at=now,
            conversation_id=request.conversation_id,
            tenant=request.tenant,
            provenance=request.provenance,
            content_digest=request.content_digest,
            redacted_excerpt=request.redacted_excerpt,
            owasp=request.owasp,
            detector_versions=request.detector_versions,
            score=request.score,
            status=IncidentStatus.PENDING,
            retention_deadline=now + timedelta(days=request.retention_days),
        )
        with self._lock:
            self._records[record.case_id] = record
        return record

    def get(self, case_id: str) -> IncidentRecord | None:
        with self._lock:
            return self._records.get(case_id)

    def list(self, status: IncidentStatus | None = None) -> list[IncidentRecord]:
        with self._lock:
            records = list(self._records.values())
        if status is not None:
            records = [record for record in records if record.status is status]
        return sorted(records, key=lambda record: (record.created_at, record.case_id))

    def pending_count(self) -> int:
        with self._lock:
            return sum(
                1 for record in self._records.values() if record.status is IncidentStatus.PENDING
            )

    # -- review: the only path to a trusted label ---------------------------------------------
    def review(self, case_id: str, request: ReviewRequest) -> IncidentRecord:
        with self._lock:
            record = self._records.get(case_id)
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
            self._records[case_id] = updated
            self._audit.append(
                {
                    "case_id": case_id,
                    "reviewer": request.reviewer,
                    "label": request.label.value,
                    "at": now.isoformat(),
                    "note": request.note or "",
                }
            )
        if request.label in TRUSTED_LABELS:
            self._memory.record_verified(
                VerifiedExample(
                    case_id=updated.case_id,
                    label=updated.status,
                    text=updated.redacted_excerpt,
                    digest=updated.content_digest,
                    owasp=updated.owasp,
                    family=updated.family,
                    tenant=updated.tenant,
                )
            )
        return updated

    def audit_entries(self) -> Sequence[dict[str, str]]:
        with self._lock:
            return list(self._audit)


def _utcnow() -> datetime:
    return datetime.now(UTC)
