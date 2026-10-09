"""Policy engine: every deny path, the allow path, and fail-closed loading."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from llm_guardian.models import PolicyAction, PolicyCheckRequest, VerdictDecision
from llm_guardian.policy import (
    BUDGET_EXCEEDED,
    CAPABILITY_NOT_PERMITTED,
    CROSS_TENANT_DENIED,
    EGRESS_NOT_ALLOWED,
    POLICY_UNAVAILABLE,
    TOOL_NOT_PERMITTED,
    TOOL_NOT_SPECIFIED,
    UNKNOWN_PRINCIPAL,
    PolicyEngine,
)
from llm_guardian.settings import PolicySettings

FIXED = lambda: datetime(2020, 1, 1, tzinfo=UTC)  # noqa: E731


def engine(policy_file: Path, **kwargs) -> PolicyEngine:
    return PolicyEngine(PolicySettings(file=policy_file, **kwargs))


def test_allowed_tool_call(policy_file: Path) -> None:
    check = engine(policy_file).check(
        PolicyCheckRequest(
            principal="dev-console", action=PolicyAction.TOOL_CALL, tool="run_tests"
        ),
        now=FIXED,
    )
    assert check.allowed is True
    assert check.decision is VerdictDecision.ALLOW
    assert check.reason_codes == ("allowed",)
    assert check.checked_at == FIXED()
    assert "run_tests" in check.capabilities


def test_unknown_principal_denied(policy_file: Path) -> None:
    check = engine(policy_file).check(
        PolicyCheckRequest(principal="nobody", action=PolicyAction.TOOL_CALL, tool="run_tests"),
        now=FIXED,
    )
    assert check.allowed is False
    assert UNKNOWN_PRINCIPAL in check.reason_codes


def test_unknown_principal_allowed_when_not_fail_closed(policy_file: Path) -> None:
    check = engine(policy_file, fail_closed=False).check(
        PolicyCheckRequest(
            principal="nobody", action=PolicyAction.RETRIEVAL, requested_capabilities=()
        ),
        now=FIXED,
    )
    assert check.allowed is True


def test_tool_not_permitted(policy_file: Path) -> None:
    check = engine(policy_file).check(
        PolicyCheckRequest(principal="ci-agent", action=PolicyAction.TOOL_CALL, tool="write_file"),
        now=FIXED,
    )
    assert check.allowed is False
    assert TOOL_NOT_PERMITTED in check.reason_codes


def test_tool_not_specified(policy_file: Path) -> None:
    check = engine(policy_file).check(
        PolicyCheckRequest(principal="ci-agent", action=PolicyAction.TOOL_CALL), now=FIXED
    )
    assert check.allowed is False
    assert TOOL_NOT_SPECIFIED in check.reason_codes


def test_capability_not_permitted(policy_file: Path) -> None:
    check = engine(policy_file).check(
        PolicyCheckRequest(
            principal="ci-agent",
            action=PolicyAction.TOOL_CALL,
            tool="run_tests",
            requested_capabilities=("write_file",),
        ),
        now=FIXED,
    )
    assert check.allowed is False
    assert CAPABILITY_NOT_PERMITTED in check.reason_codes


def test_egress_not_allowed(policy_file: Path) -> None:
    check = engine(policy_file).check(
        PolicyCheckRequest(
            principal="dev-console", action=PolicyAction.EGRESS, egress_hosts=("evil.example",)
        ),
        now=FIXED,
    )
    assert check.allowed is False
    assert EGRESS_NOT_ALLOWED in check.reason_codes


def test_egress_allowed(policy_file: Path) -> None:
    check = engine(policy_file).check(
        PolicyCheckRequest(
            principal="dev-console", action=PolicyAction.EGRESS, egress_hosts=("github.com",)
        ),
        now=FIXED,
    )
    assert check.allowed is True


def test_cross_tenant_denied(policy_file: Path) -> None:
    check = engine(policy_file).check(
        PolicyCheckRequest(
            principal="ci-agent",
            tenant="platform",
            action=PolicyAction.RETRIEVAL,
            target_tenant="other-team",
        ),
        now=FIXED,
    )
    assert check.allowed is False
    assert CROSS_TENANT_DENIED in check.reason_codes


def test_cross_tenant_same_tenant_allowed(policy_file: Path) -> None:
    check = engine(policy_file).check(
        PolicyCheckRequest(
            principal="ci-agent",
            tenant="platform",
            action=PolicyAction.RETRIEVAL,
            target_tenant="platform",
        ),
        now=FIXED,
    )
    assert check.allowed is True


def test_cross_tenant_grant_allows(tmp_path: Path) -> None:
    policy = tmp_path / "grant.toml"
    policy.write_text(
        """
[defaults]
capabilities = ["read_repository"]
[principals.reader]
capabilities = ["read_repository"]
[tenants.analytics]
allowed_cross_tenant_reads = ["reader"]
""",
        encoding="utf-8",
    )
    check = engine(policy).check(
        PolicyCheckRequest(
            principal="reader",
            tenant="platform",
            action=PolicyAction.RETRIEVAL,
            target_tenant="analytics",
        ),
        now=FIXED,
    )
    assert check.allowed is True


def test_budget_exceeded(policy_file: Path) -> None:
    check = engine(policy_file).check(
        PolicyCheckRequest(
            principal="ci-agent",
            action=PolicyAction.BUDGET,
            requested_budget={"tokens": 999999.0},
        ),
        now=FIXED,
    )
    assert check.allowed is False
    assert BUDGET_EXCEEDED in check.reason_codes


def test_budget_unknown_key_fails_closed(policy_file: Path) -> None:
    check = engine(policy_file).check(
        PolicyCheckRequest(
            principal="ci-agent", action=PolicyAction.BUDGET, requested_budget={"gpus": 1.0}
        ),
        now=FIXED,
    )
    assert check.allowed is False
    assert BUDGET_EXCEEDED in check.reason_codes


def test_missing_policy_file_fails_closed(tmp_path: Path) -> None:
    engine_ = PolicyEngine(PolicySettings(file=tmp_path / "missing.toml"))
    check = engine_.check(
        PolicyCheckRequest(
            principal="dev-console", action=PolicyAction.TOOL_CALL, tool="run_tests"
        ),
        now=FIXED,
    )
    assert check.allowed is False
    assert check.reason_codes == (POLICY_UNAVAILABLE,)
    assert engine_.loaded() is False
    assert engine_.load_error is not None


def test_no_policy_configured_fails_closed() -> None:
    engine_ = PolicyEngine(PolicySettings(file=None))
    check = engine_.check(
        PolicyCheckRequest(principal="x", action=PolicyAction.RETRIEVAL), now=FIXED
    )
    assert check.reason_codes == (POLICY_UNAVAILABLE,)
    assert engine_.load_error == "no policy file configured"


def test_unreadable_policy_fails_closed(tmp_path: Path) -> None:
    bad = tmp_path / "policy.toml"
    bad.write_text("this is [ not toml", encoding="utf-8")
    engine_ = PolicyEngine(PolicySettings(file=bad))
    assert engine_.loaded() is False
    assert engine_.load_error is not None


def test_json_policy_file(tmp_path: Path) -> None:
    policy = tmp_path / "policy.json"
    policy.write_text(
        json.dumps(
            {
                "defaults": {"capabilities": ["read_repository"]},
                "principals": {"dev-console": {"capabilities": ["run_tests"]}},
            }
        ),
        encoding="utf-8",
    )
    check = engine(policy).check(
        PolicyCheckRequest(
            principal="dev-console", action=PolicyAction.TOOL_CALL, tool="run_tests"
        ),
        now=FIXED,
    )
    assert check.allowed is True


def test_reload(tmp_path: Path) -> None:
    policy = tmp_path / "policy.toml"
    policy.write_text("[defaults]\ncapabilities = []\n", encoding="utf-8")
    engine_ = engine(policy)
    assert engine_.loaded() is True
    engine_.reload()
    assert engine_.loaded() is True


def test_wildcard_egress(tmp_path: Path) -> None:
    policy = tmp_path / "policy.toml"
    policy.write_text(
        '[defaults]\ncapabilities = ["read_repository"]\negress_hosts = ["*.example.com"]\n'
        '[principals.dev]\ncapabilities = ["read_repository"]\n',
        encoding="utf-8",
    )
    engine_ = engine(policy)
    allowed = engine_.check(
        PolicyCheckRequest(
            principal="dev", action=PolicyAction.EGRESS, egress_hosts=("api.example.com",)
        ),
        now=FIXED,
    )
    assert allowed.allowed is True
    apex = engine_.check(
        PolicyCheckRequest(
            principal="dev", action=PolicyAction.EGRESS, egress_hosts=("example.com",)
        ),
        now=FIXED,
    )
    assert apex.allowed is True
    denied = engine_.check(
        PolicyCheckRequest(principal="dev", action=PolicyAction.EGRESS, egress_hosts=("evil.com",)),
        now=FIXED,
    )
    assert denied.allowed is False


def test_defaults_merge_into_principal(tmp_path: Path) -> None:
    policy = tmp_path / "policy.toml"
    policy.write_text(
        """
[defaults]
capabilities = ["read_repository"]
budgets = { tokens = 100.0 }
[principals.dev]
capabilities = ["run_tests"]
""",
        encoding="utf-8",
    )
    check = engine(policy).check(
        PolicyCheckRequest(principal="dev", action=PolicyAction.TOOL_CALL, tool="run_tests"),
        now=FIXED,
    )
    assert check.allowed is True
    assert check.budgets == {"tokens": 100.0}
