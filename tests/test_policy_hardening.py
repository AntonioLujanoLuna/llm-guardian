"""Parent-authored hardening tests for the tool boundary.

Both cases here were found by re-deriving the child workstream's own measurement with an
independent loop: the corpus exercise reported a gap the child had documented in prose but not
covered by a test, and reading `_check_cross_tenant` confirmed it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_guardian.models import PolicyCheckRequest
from llm_guardian.policy import PolicyEngine
from llm_guardian.settings import PolicySettings

POLICY = Path("policies/example.toml")


@pytest.fixture
def engine() -> PolicyEngine:
    return PolicyEngine(PolicySettings(file=POLICY))


def check(engine: PolicyEngine, **kwargs: object) -> tuple[bool, tuple[str, ...]]:
    decision = engine.check(PolicyCheckRequest(**kwargs))  # type: ignore[arg-type]
    return decision.allowed, decision.reason_codes


class TestCrossTenantGrantsAreDirectional:
    """A tenant that invites a principal to read has not invited it to write."""

    def test_a_read_grant_does_not_authorise_a_write(self, engine: PolicyEngine) -> None:
        # `analytics` grants dev-console `allowed_cross_tenant_reads` and nothing else.
        allowed, reasons = check(
            engine,
            principal="dev-console",
            tenant="platform",
            action="tool_call",
            tool="db_write",
            tool_arguments={"table": "orders", "row": '{"id": 1}'},
            target_tenant="analytics",
        )
        assert not allowed
        assert "cross_tenant_denied" in reasons

    def test_the_same_principal_may_still_read_that_tenant(self, engine: PolicyEngine) -> None:
        allowed, reasons = check(
            engine,
            principal="dev-console",
            tenant="platform",
            action="retrieval",
            target_tenant="analytics",
        )
        assert allowed, reasons

    def test_a_declared_write_grant_is_honoured(self, tmp_path: Path) -> None:
        policy = tmp_path / "policy.toml"
        policy.write_text(
            """
[tenants.platform]
principals = ["dev-console"]
allowed_cross_tenant_reads = []
allowed_cross_tenant_writes = []

[tenants.partner]
principals = ["dev-console"]
allowed_cross_tenant_reads = ["dev-console"]
allowed_cross_tenant_writes = ["dev-console"]

[principals.dev-console]
capabilities = ["db_write"]

[tools.db_write]
allowed = ["table", "row"]
required = ["table", "row"]
"""
        )
        engine = PolicyEngine(PolicySettings(file=policy))
        allowed, reasons = check(
            engine,
            principal="dev-console",
            tenant="platform",
            action="tool_call",
            tool="db_write",
            tool_arguments={"table": "orders", "row": '{"id": 1}'},
            target_tenant="partner",
        )
        assert allowed, reasons

    def test_a_write_into_your_own_tenant_is_not_a_cross_tenant_call(
        self, engine: PolicyEngine
    ) -> None:
        allowed, reasons = check(
            engine,
            principal="dev-console",
            tenant="platform",
            action="tool_call",
            tool="db_write",
            tool_arguments={"table": "orders", "row": '{"id": 1}'},
            target_tenant="platform",
        )
        assert allowed, reasons


class TestShellCommandShape:
    """`run_shell` constrains the command's shape; a sensitive token inside it is denied."""

    @pytest.mark.parametrize(
        "command",
        [
            "cat /home/ubuntu/.ssh/id_rsa",
            "cat ~/.aws/credentials",
            "python3 -c 'print(open(\"/etc/passwd\").read())'",
            "ls .env",
            "cat /home/ubuntu/deploy.pem",
        ],
    )
    def test_sensitive_tokens_are_denied(self, engine: PolicyEngine, command: str) -> None:
        allowed, reasons = check(
            engine,
            principal="dev-console",
            tenant="platform",
            action="tool_call",
            tool="run_shell",
            tool_arguments={"cmd": command},
        )
        assert not allowed
        assert "argument_forbidden_pattern" in reasons

    @pytest.mark.parametrize(
        "command",
        ["ls -la /workspace", "pytest -q tests/unit", "git status", "make test"],
    )
    def test_ordinary_commands_are_allowed(self, engine: PolicyEngine, command: str) -> None:
        allowed, reasons = check(
            engine,
            principal="dev-console",
            tenant="platform",
            action="tool_call",
            tool="run_shell",
            tool_arguments={"cmd": command},
        )
        assert allowed, reasons

    def test_the_deny_pattern_does_not_narrow_free_form_arguments(
        self, engine: PolicyEngine
    ) -> None:
        """A declared free-form value may still hold punctuation: the rule is per argument."""
        allowed, reasons = check(
            engine,
            principal="dev-console",
            tenant="platform",
            action="tool_call",
            tool="query_db",
            tool_arguments={"sql": "SELECT note FROM orders WHERE note = 'a; b'"},
        )
        assert allowed, reasons

    def test_a_deny_pattern_is_a_shape_rule_not_a_sandbox(self, engine: PolicyEngine) -> None:
        """An allow-listed binary that reads no sensitive token is still permitted.

        Stated as a test so the limit is explicit: the layer sees the argument, not what the
        program does at run time.
        """
        allowed, reasons = check(
            engine,
            principal="dev-console",
            tenant="platform",
            action="tool_call",
            tool="run_shell",
            tool_arguments={"cmd": "python3 script_that_reads_anything_at_runtime.py"},
        )
        assert allowed, reasons
