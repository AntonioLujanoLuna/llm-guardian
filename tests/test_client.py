"""The typed client, against the real application in-process (no server, no network)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from llm_guardian.app import create_app
from llm_guardian.client import AsyncGuardianClient, GuardianClient, GuardianError
from llm_guardian.models import (
    EvidenceRef,
    GuardMode,
    IncidentCreateRequest,
    IncidentStatus,
    PolicyAction,
    PolicyCheckRequest,
    ReviewRequest,
    Severity,
    VerdictDecision,
)
from llm_guardian.provenance import Provenance
from llm_guardian.settings import GuardianSettings, PolicySettings, ServerSettings
from tests.support import POLICY_TOML, TOOL_RESULT_INJECTION


@pytest.fixture
def settings(tmp_path: Path) -> GuardianSettings:
    policy = tmp_path / "policy.toml"
    policy.write_text(POLICY_TOML, encoding="utf-8")
    return GuardianSettings(
        mode=GuardMode.ENFORCE,
        policy=PolicySettings(file=policy),
        server=ServerSettings(service_token=SecretStr("t0k")),
    )


@pytest.fixture
def guardian(settings: GuardianSettings) -> GuardianClient:
    transport = TestClient(create_app(settings))._transport
    return GuardianClient("http://guardian/", token="t0k", transport=transport)


def test_scan_text_returns_the_wire_model(guardian: GuardianClient) -> None:
    verdict = guardian.scan_text(TOOL_RESULT_INJECTION, provenance="tool_result")
    assert verdict.blocked
    assert verdict.segments[0].provenance is Provenance.TOOL_RESULT
    assert guardian.scan_text("hello").decision is VerdictDecision.ALLOW


def test_authorize_and_check_policy(guardian: GuardianClient) -> None:
    allowed = guardian.authorize(principal="ci-agent", tool="run_tests")
    assert allowed.decision is VerdictDecision.ALLOW
    denied = guardian.authorize(principal="ci-agent", tool="write_file")
    assert denied.denied_by == "tool_not_permitted"
    deputy = guardian.authorize(
        principal="ci-agent",
        tool="run_tests",
        evidence=(
            EvidenceRef(
                segment_id="s",
                finding_kind="tool_call_instruction",
                severity=Severity.HIGH,
                provenance=Provenance.TOOL_RESULT,
            ),
        ),
    )
    assert deputy.evidence_used == ("tool_call_instruction",)
    check = guardian.check_policy(
        PolicyCheckRequest(principal="ci-agent", action=PolicyAction.EGRESS, egress_hosts=("x.io",))
    )
    assert check.check.allowed is False


def test_incident_lifecycle_and_paging(guardian: GuardianClient) -> None:
    for index in range(5):
        guardian.create_incident(
            IncidentCreateRequest(
                provenance=Provenance.TOOL_RESULT,
                content_digest=f"d{index}",
                redacted_excerpt="x",
                tenant="acme",
            )
        )
    listed = list(guardian.incidents(tenant="acme", page_size=2))
    assert [record.content_digest for record in listed] == [f"d{i}" for i in range(5)]
    reviewed = guardian.review(
        listed[0].case_id, ReviewRequest(reviewer="ana", label=IncidentStatus.BENIGN)
    )
    assert reviewed.status is IncidentStatus.BENIGN
    assert [r.content_digest for r in guardian.incidents(status=IncidentStatus.BENIGN)] == ["d0"]
    assert len(list(guardian.incidents(provenance="user"))) == 0


def test_health_and_metrics(guardian: GuardianClient) -> None:
    guardian.scan_text("hello")
    assert guardian.health().mode is GuardMode.ENFORCE
    assert guardian.metrics().scans == 1


def test_errors_carry_status_and_detail(settings: GuardianSettings) -> None:
    transport = TestClient(create_app(settings))._transport
    with (
        GuardianClient("http://guardian", token="wrong", transport=transport) as bad,
        pytest.raises(GuardianError) as raised,
    ):
        bad.scan_text("hello")
    assert raised.value.status_code == 401
    assert "X-Guardian-Token" in str(raised.value.detail)
    with (
        GuardianClient("http://guardian", token="t0k", transport=transport) as good,
        pytest.raises(GuardianError) as missing,
    ):
        good.review("nope", ReviewRequest(reviewer="a", label=IncidentStatus.BENIGN))
    assert missing.value.status_code == 404


def test_error_with_a_non_json_body() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(502, text="bad gateway")

    with (
        GuardianClient("http://g", transport=httpx.MockTransport(handler)) as client,
        pytest.raises(GuardianError) as raised,
    ):
        client.health()
    assert raised.value.detail == "bad gateway"


def test_async_client(settings: GuardianSettings) -> None:
    app = create_app(settings)

    async def run() -> None:
        transport = httpx.ASGITransport(app=app)
        async with AsyncGuardianClient("http://guardian", token="t0k", transport=transport) as g:
            verdict = await g.scan_text(TOOL_RESULT_INJECTION, provenance=Provenance.TOOL_RESULT)
            assert verdict.blocked
            assert (await g.authorize(principal="ci-agent", tool="run_tests")).check.allowed
            assert (await g.health()).status
            incident = verdict.segments[0]
            assert incident.decision is VerdictDecision.BLOCK
            with pytest.raises(GuardianError):
                await g.review("nope", ReviewRequest(reviewer="a", label=IncidentStatus.BENIGN))

    asyncio.run(run())
