"""Every HTTP endpoint, exercised through the real application."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from llm_guardian.app import create_app
from llm_guardian.deps import get_risk_memory
from llm_guardian.models import GuardMode
from llm_guardian.settings import GuardianSettings, PolicySettings, ServerSettings
from tests.support import POLICY_TOML, TOOL_RESULT_INJECTION


class ExplodingMemory:
    backend = "exploding"
    reachable = False
    degraded_reason = None

    def query(self, *args, **kwargs):
        raise RuntimeError("backend offline")

    def record_verified(self, example) -> None:  # pragma: no cover
        pass

    def stats(self):  # pragma: no cover
        raise RuntimeError("backend offline")

    def counts(self):  # pragma: no cover
        return {}


def test_scan_benign(client: TestClient) -> None:
    response = client.post(
        "/v1/scan",
        json={"segments": [{"text": "write a haiku about tests", "provenance": "user"}]},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["decision"] == "allow"
    assert body["mode"] == "enforce"


def test_scan_injection_blocks(client: TestClient) -> None:
    response = client.post(
        "/v1/scan",
        json={"segments": [{"text": TOOL_RESULT_INJECTION, "provenance": "tool_result"}]},
    )
    assert response.status_code == 200
    assert response.json()["decision"] == "block"


def test_scan_invalid_provenance(client: TestClient) -> None:
    response = client.post(
        "/v1/scan", json={"segments": [{"text": "x", "provenance": "not-a-provenance"}]}
    )
    assert response.status_code == 422


def test_policy_check_allowed_and_denied(client: TestClient) -> None:
    allowed = client.post(
        "/v1/policy/check",
        json={"principal": "dev-console", "action": "tool_call", "tool": "run_tests"},
    )
    assert allowed.status_code == 200
    assert allowed.json()["check"]["allowed"] is True

    denied = client.post(
        "/v1/policy/check",
        json={
            "principal": "ci-agent",
            "action": "tool_call",
            "tool": "run_tests",
            "requested_capabilities": ["write_file"],
        },
    )
    assert denied.status_code == 200
    assert denied.json()["check"]["allowed"] is False
    assert "capability_not_permitted" in denied.json()["check"]["reason_codes"]


def test_incident_lifecycle_endpoints(client: TestClient) -> None:
    created = client.post(
        "/v1/incidents",
        json={
            "provenance": "tool_result",
            "content_digest": "abc",
            "redacted_excerpt": "instruction_override (critical)",
        },
    )
    assert created.status_code == 201
    case_id = created.json()["case_id"]
    assert created.json()["status"] == "pending"

    listed = client.get("/v1/incidents", params={"status": "pending"})
    assert listed.status_code == 200
    assert any(item["case_id"] == case_id for item in listed.json())

    reviewed = client.post(
        f"/v1/incidents/{case_id}/review",
        json={"reviewer": "ana", "label": "confirmed_attack", "note": "verified"},
    )
    assert reviewed.status_code == 200
    assert reviewed.json()["status"] == "confirmed_attack"
    assert reviewed.json()["reviewed_by"] == "ana"

    missing = client.post(
        "/v1/incidents/does-not-exist/review",
        json={"reviewer": "ana", "label": "benign"},
    )
    assert missing.status_code == 404


def test_incident_review_validation(client: TestClient) -> None:
    created = client.post(
        "/v1/incidents",
        json={"provenance": "user", "content_digest": "d", "redacted_excerpt": "e"},
    ).json()
    bad = client.post(
        f"/v1/incidents/{created['case_id']}/review",
        json={"reviewer": "", "label": "benign"},
    )
    assert bad.status_code == 422


def test_memory_query_and_stats_move_after_review(client: TestClient) -> None:
    before = client.get("/v1/memory/stats").json()
    assert before["confirmed_attacks"] == 0

    created = client.post(
        "/v1/incidents",
        json={
            "provenance": "tool_result",
            "content_digest": "d",
            "redacted_excerpt": "ignore previous instructions",
        },
    ).json()
    client.post(
        f"/v1/incidents/{created['case_id']}/review",
        json={"reviewer": "ana", "label": "confirmed_attack"},
    )
    after = client.get("/v1/memory/stats").json()
    assert after["confirmed_attacks"] == 1

    query = client.post("/v1/memory/query", json={"text": "ignore previous instructions", "k": 5})
    assert query.status_code == 200
    body = query.json()
    assert {"attack_sim", "benign_sim", "margin", "neighbors"} <= set(body)
    assert body["attack_sim"] > 0


def test_memory_query_503_when_backend_down(tmp_path: Path) -> None:
    policy = tmp_path / "policy.toml"
    policy.write_text(POLICY_TOML, encoding="utf-8")
    app = create_app(GuardianSettings(policy=PolicySettings(file=policy)))
    app.dependency_overrides[get_risk_memory] = lambda: ExplodingMemory()
    with TestClient(app) as client:
        response = client.post("/v1/memory/query", json={"text": "x"})
        assert response.status_code == 503


def test_health_endpoint(client: TestClient) -> None:
    body = client.get("/v1/health").json()
    assert body["status"] in {"ok", "degraded"}
    assert body["mode"] == "enforce"
    assert "rules" in body["scanner_versions"]
    assert "cache" in body


def test_metrics_endpoints(client: TestClient) -> None:
    client.post(
        "/v1/scan",
        json={"segments": [{"text": "hello", "provenance": "user"}]},
    )
    metrics = client.get("/v1/metrics").json()
    assert metrics["scans"] >= 1
    assert "total" in metrics["stages"]

    prometheus = client.get("/metrics")
    assert prometheus.status_code == 200
    assert "text/plain" in prometheus.headers["content-type"]
    assert "guardian_scans_total" in prometheus.text


def test_owasp_endpoint(client: TestClient) -> None:
    body = client.get("/v1/owasp").json()
    assert len(body["categories"]) == 10
    assert "aggregate" in body["note"] or "no aggregate" in body["note"]


def test_service_token_enforced(tmp_path: Path) -> None:
    policy = tmp_path / "policy.toml"
    policy.write_text(POLICY_TOML, encoding="utf-8")
    settings = GuardianSettings(
        mode=GuardMode.ENFORCE,
        policy=PolicySettings(file=policy),
        server=ServerSettings(service_token="s3cret"),  # type: ignore[arg-type]
    )
    with TestClient(create_app(settings)) as client:
        unauthorised = client.get("/v1/health")
        assert unauthorised.status_code == 401
        authorised = client.get("/v1/health", headers={"X-Guardian-Token": "s3cret"})
        assert authorised.status_code == 200
        # /metrics is not under /v1 and stays open.
        assert client.get("/metrics").status_code == 200
