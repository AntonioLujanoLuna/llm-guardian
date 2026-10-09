"""Incident lifecycle: pending is not a label, review promotes, and nothing auto-promotes."""

from __future__ import annotations

import types
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from llm_guardian.incidents import IncidentStore, NoAutoPromotionError
from llm_guardian.memory import InMemoryRiskMemory
from llm_guardian.models import (
    IncidentCreateRequest,
    IncidentStatus,
    ReviewRequest,
)
from llm_guardian.provenance import Provenance
from llm_guardian.settings import GuardianSettings

FIXED = datetime(2020, 6, 1, tzinfo=UTC)


def make_store() -> tuple[IncidentStore, InMemoryRiskMemory]:
    settings = GuardianSettings()
    memory = InMemoryRiskMemory(settings.memory)
    return IncidentStore(settings, memory, clock=lambda: FIXED), memory


def create(store: IncidentStore, retention_days: int = 30):
    return store.create(
        IncidentCreateRequest(
            conversation_id="c1",
            tenant="platform",
            provenance=Provenance.TOOL_RESULT,
            content_digest="digest-1",
            redacted_excerpt="instruction_override (critical): ...",
            score=0.9,
            retention_days=retention_days,
        )
    )


def test_create_is_pending_and_never_promotes() -> None:
    store, memory = make_store()
    record = create(store)
    assert record.status is IncidentStatus.PENDING
    assert record.reviewed_by is None
    assert memory.counts() == {
        "confirmed_attacks": 0,
        "benign_hard_negatives": 0,
        "pending_incidents": 0,
    }
    assert store.pending_count() == 1


def test_retention_deadline() -> None:
    store, _ = make_store()
    record = create(store, retention_days=10)
    assert (record.retention_deadline - record.created_at).days == 10


def test_get_and_list_with_filter() -> None:
    store, _ = make_store()
    record = create(store)
    assert store.get(record.case_id) is not None
    assert store.get("missing") is None
    assert len(store.list()) == 1
    assert store.list(IncidentStatus.PENDING) == [record]
    assert store.list(IncidentStatus.BENIGN) == []


def test_review_confirmed_attack_promotes_and_audits() -> None:
    store, memory = make_store()
    record = create(store)
    reviewed = store.review(
        record.case_id,
        ReviewRequest(
            reviewer="ana", label=IncidentStatus.CONFIRMED_ATTACK, note="exfil", family="f1"
        ),
    )
    assert reviewed.status is IncidentStatus.CONFIRMED_ATTACK
    assert reviewed.reviewed_by == "ana"
    assert reviewed.reviewed_at == FIXED
    assert reviewed.review_note == "exfil"
    assert reviewed.family == "f1"
    assert memory.counts()["confirmed_attacks"] == 1
    audit = store.audit_entries()
    assert audit[-1]["reviewer"] == "ana"
    assert audit[-1]["label"] == "confirmed_attack"


def test_review_benign_promotes_benign_partition() -> None:
    store, memory = make_store()
    record = create(store)
    store.review(record.case_id, ReviewRequest(reviewer="bob", label=IncidentStatus.BENIGN))
    assert memory.counts()["benign_hard_negatives"] == 1
    assert memory.counts()["confirmed_attacks"] == 0


def test_review_dismissed_closes_without_promoting() -> None:
    store, memory = make_store()
    record = create(store)
    reviewed = store.review(
        record.case_id, ReviewRequest(reviewer="carl", label=IncidentStatus.DISMISSED)
    )
    assert reviewed.status is IncidentStatus.DISMISSED
    assert memory.counts()["confirmed_attacks"] == 0
    assert memory.counts()["benign_hard_negatives"] == 0


def test_review_unknown_case() -> None:
    store, _ = make_store()
    with pytest.raises(KeyError):
        store.review("nope", ReviewRequest(reviewer="x", label=IncidentStatus.BENIGN))


def test_reviewer_identity_required() -> None:
    with pytest.raises(ValidationError):
        ReviewRequest(reviewer="", label=IncidentStatus.BENIGN)


def test_no_auto_promotion_guard() -> None:
    fake_settings = types.SimpleNamespace(
        memory=types.SimpleNamespace(auto_promote=True),
    )
    with pytest.raises(NoAutoPromotionError):
        IncidentStore(fake_settings, InMemoryRiskMemory(GuardianSettings().memory))


def test_auto_promote_setting_is_frozen_off() -> None:
    assert GuardianSettings().memory.auto_promote is False
