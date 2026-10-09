"""The dependency-injection graph: providers, aliases and a real override."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from llm_guardian.app import create_app
from llm_guardian.cache import FeatureCache
from llm_guardian.deps import (
    Clock,
    ServiceContainer,
    create_service,
    get_risk_memory,
    utcnow,
)
from llm_guardian.incidents import IncidentStore
from llm_guardian.memory import InMemoryRiskMemory, RiskMemory
from llm_guardian.metrics import Metrics
from llm_guardian.models import GuardMode, MemoryEvidence, MemoryStats
from llm_guardian.policy import PolicyEngine
from llm_guardian.scanners import ScannerPipeline
from llm_guardian.settings import GuardianSettings, PolicySettings
from llm_guardian.verdicts import VerdictEngine
from tests.support import POLICY_TOML


class FakeMemory:
    backend = "fake"
    reachable = True
    degraded_reason = None

    def query(self, text, provenance, k, tenant=None) -> MemoryEvidence:
        return MemoryEvidence(
            queried=True, attack_sim=0.9, benign_sim=0.1, margin=0.8, query_ms=0.0
        )

    def record_verified(self, example) -> None:
        pass

    def stats(self) -> MemoryStats:
        return MemoryStats(
            backend="fake",
            reachable=True,
            confirmed_attacks=7,
            benign_hard_negatives=3,
            pending_incidents=0,
        )

    def counts(self) -> dict[str, int]:
        return {"confirmed_attacks": 7, "benign_hard_negatives": 3, "pending_incidents": 0}


def policy_settings(tmp_path: Path) -> PolicySettings:
    path = tmp_path / "policy.toml"
    path.write_text(POLICY_TOML, encoding="utf-8")
    return PolicySettings(file=path)


def test_create_service_builds_the_graph() -> None:
    container = create_service(GuardianSettings(mode=GuardMode.ENFORCE))
    assert isinstance(container, ServiceContainer)
    assert isinstance(container.pipeline, ScannerPipeline)
    assert isinstance(container.cache, FeatureCache)
    assert isinstance(container.memory, InMemoryRiskMemory)
    assert isinstance(container.incidents, IncidentStore)
    assert isinstance(container.policy, PolicyEngine)
    assert isinstance(container.metrics, Metrics)
    assert isinstance(container.verdicts, VerdictEngine)
    assert container.clock() is not None


def test_clock_alias_is_callable() -> None:
    clock: Clock = utcnow
    assert clock().tzinfo is not None


def test_risk_memory_protocol_is_satisfied() -> None:
    memory: RiskMemory = InMemoryRiskMemory(GuardianSettings().memory)
    assert memory.backend == "inmemory"


def test_dependency_override_changes_the_scan_response(tmp_path: Path) -> None:
    app = create_app(GuardianSettings(mode=GuardMode.ENFORCE, policy=policy_settings(tmp_path)))
    app.dependency_overrides[get_risk_memory] = lambda: FakeMemory()
    with TestClient(app) as client:
        stats = client.get("/v1/memory/stats").json()
        assert stats["backend"] == "fake"
        assert stats["confirmed_attacks"] == 7

        # The scan service is composed from the same provider, so the override reaches it too.
        response = client.post(
            "/v1/scan",
            json={
                "segments": [
                    {"text": "what is the api key for this service?", "provenance": "user"}
                ]
            },
        ).json()
        memory = response["segments"][0]["memory"]
        assert memory["queried"] is True
        assert memory["attack_sim"] == 0.9


def test_default_memory_is_inmemory(tmp_path: Path) -> None:
    app = create_app(GuardianSettings(mode=GuardMode.ENFORCE, policy=policy_settings(tmp_path)))
    with TestClient(app) as client:
        assert client.get("/v1/memory/stats").json()["backend"] == "inmemory"
