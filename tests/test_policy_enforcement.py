"""Tool-boundary enforcement: argument schemas, the evidence rule, and ``/v1/agent/authorize``.

These are additive to ``tests/test_policy.py`` (which pins the existing behaviour) and
``tests/test_api.py``. They cover the two new rules and the combined endpoint: every
``argument_*`` failure type, the confused-deputy evidence rule for each injection kind and each
untrusted provenance, the operator-provenance non-denial, and the fail-closed paths.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from llm_guardian.app import create_app
from llm_guardian.models import (
    EvidenceRef,
    PolicyAction,
    PolicyCheckRequest,
    Provenance,
    Severity,
    VerdictDecision,
)
from llm_guardian.policy import (
    ARGUMENT_HOST_NOT_ALLOWED,
    ARGUMENT_MISSING,
    ARGUMENT_PATH_ESCAPE,
    ARGUMENT_PATTERN,
    ARGUMENT_SCHEMA_MISSING,
    ARGUMENT_SHELL_METACHAR,
    ARGUMENT_TOO_LONG,
    ARGUMENT_UNDECLARED,
    CROSS_TENANT_DENIED,
    EVIDENCE_UNTRUSTED_INJECTION,
    POLICY_UNAVAILABLE,
    PolicyEngine,
)
from llm_guardian.settings import GuardianSettings, PolicySettings

FIXED = lambda: datetime(2020, 1, 1, tzinfo=UTC)  # noqa: E731

#: A policy with argument schemas, for the tool boundary. ``call_mcp`` is deliberately a declared
#: capability with *no* ``[tools.call_mcp]`` schema, to exercise the fail-closed
#: missing-schema path.
POLICY_TOML = r"""
[defaults]
capabilities = ["read_repository"]
egress_hosts = []
budgets = { tokens = 200000.0 }

[principals.dev]
capabilities = [
    "read_file", "write_file", "run_shell", "query_db", "send_mail",
    "deploy_staging", "fetch_dependency", "update_ticket", "db_write", "call_mcp",
]
egress_hosts = ["pypi.org", "github.com"]

[tenants.platform]
allowed_cross_tenant_reads = []

[tools.read_file]
allowed = ["path"]
required = ["path"]
[tools.read_file.constraints.path]
path_within = "/workspace"
max_len = 512

[tools.write_file]
allowed = ["path", "content"]
required = ["path", "content"]
free_form = ["content"]
[tools.write_file.constraints.path]
path_within = "/workspace"

[tools.run_shell]
allowed = ["cmd"]
required = ["cmd"]
[tools.run_shell.constraints.cmd]
pattern = '^(pytest|python3|git|ls|cat)\b[^;&|`$()\n]*$'
max_len = 512

[tools.query_db]
allowed = ["sql"]
required = ["sql"]
free_form = ["sql"]
[tools.query_db.constraints.sql]
max_len = 4096

[tools.send_mail]
allowed = ["to", "subject", "body"]
required = ["to"]
free_form = ["subject", "body"]
[tools.send_mail.constraints.to]
host_allow = ["example.internal"]

[tools.deploy_staging]
allowed = ["service", "version"]
required = ["service", "version"]
[tools.deploy_staging.constraints.service]
pattern = '^[a-z0-9][a-z0-9-]{1,63}$'

[tools.fetch_dependency]
allowed = ["host"]
required = ["host"]
[tools.fetch_dependency.constraints.host]
host_allow = ["pypi.org", "github.com"]

[tools.update_ticket]
allowed = ["ticket"]
required = ["ticket"]
[tools.update_ticket.constraints.ticket]
pattern = '^[A-Z][A-Z0-9]*-[0-9]+$'

[tools.db_write]
allowed = ["table", "row"]
required = ["table", "row"]
free_form = ["row"]
"""


@pytest.fixture
def tools_policy_file(tmp_path: Path) -> Path:
    path = tmp_path / "tools.toml"
    path.write_text(POLICY_TOML, encoding="utf-8")
    return path


def engine(policy_file: Path) -> PolicyEngine:
    return PolicyEngine(PolicySettings(file=policy_file))


def check(policy_file: Path, **kwargs: object):
    request = PolicyCheckRequest(**kwargs)  # type: ignore[arg-type]
    return engine(policy_file).check(request, now=FIXED)


def tool(policy_file: Path, tool_name: str, arguments: dict[str, object], **kwargs: object):
    return check(
        policy_file,
        principal="dev",
        action=PolicyAction.TOOL_CALL,
        tool=tool_name,
        tool_arguments=arguments,
        **kwargs,
    )


def evidence(kind: str, provenance: Provenance) -> tuple[EvidenceRef, ...]:
    return (
        EvidenceRef(
            segment_id="seg-1",
            finding_kind=kind,
            severity=Severity.HIGH,
            provenance=provenance,
        ),
    )


# =================================================================================================
# Argument schemas
# =================================================================================================
def test_valid_tool_call_is_allowed(tools_policy_file: Path) -> None:
    result = tool(tools_policy_file, "read_file", {"path": "src/app.py"})
    assert result.allowed is True
    assert result.decision is VerdictDecision.ALLOW
    assert result.denied_by is None
    assert result.argument_failures == ()
    assert result.evidence_used == ()


def test_path_outside_root_is_denied(tools_policy_file: Path) -> None:
    result = tool(tools_policy_file, "read_file", {"path": "/etc/passwd"})
    assert result.allowed is False
    assert ARGUMENT_PATH_ESCAPE in result.reason_codes
    assert result.denied_by == ARGUMENT_PATH_ESCAPE
    assert result.argument_failures == ("argument_path_escape:path",)


def test_tilde_path_is_denied(tools_policy_file: Path) -> None:
    result = tool(tools_policy_file, "read_file", {"path": "~/.ssh/id_rsa"})
    assert result.allowed is False
    assert ARGUMENT_PATH_ESCAPE in result.reason_codes


@pytest.mark.parametrize(
    "path",
    [
        "src/../tests/test_flow.py",
        "/workspace/../workspace/README.md",
        "config/../config/settings.toml",
        "docs/../../workspace/docs/index.md",
    ],
)
def test_dotdot_that_stays_inside_is_allowed(tools_policy_file: Path, path: str) -> None:
    result = tool(tools_policy_file, "read_file", {"path": path})
    assert result.allowed is True, result.argument_failures


@pytest.mark.parametrize(
    "path", ["../../etc/shadow", "/workspace/../.env", "secrets/../../etc/hosts"]
)
def test_dotdot_that_escapes_is_denied(tools_policy_file: Path, path: str) -> None:
    result = tool(tools_policy_file, "read_file", {"path": path})
    assert result.allowed is False
    assert ARGUMENT_PATH_ESCAPE in result.reason_codes


def test_shell_metacharacter_in_a_non_free_form_value_is_denied(tools_policy_file: Path) -> None:
    result = tool(tools_policy_file, "run_shell", {"cmd": "pytest; rm -rf /"})
    assert result.allowed is False
    assert ARGUMENT_SHELL_METACHAR in result.reason_codes
    assert result.denied_by == ARGUMENT_SHELL_METACHAR


def test_shell_metacharacter_in_a_free_form_value_is_allowed(tools_policy_file: Path) -> None:
    result = tool(tools_policy_file, "query_db", {"sql": "SELECT name WHERE label = 'a;b'"})
    assert result.allowed is True, result.argument_failures


def test_pattern_violation_is_denied(tools_policy_file: Path) -> None:
    result = tool(tools_policy_file, "run_shell", {"cmd": "rm -rf /data"})
    assert result.allowed is False
    assert ARGUMENT_PATTERN in result.reason_codes


def test_host_outside_allow_list_is_denied(tools_policy_file: Path) -> None:
    result = tool(tools_policy_file, "fetch_dependency", {"host": "evil.example"})
    assert result.allowed is False
    assert ARGUMENT_HOST_NOT_ALLOWED in result.reason_codes


def test_recipient_domain_is_checked(tools_policy_file: Path) -> None:
    allowed = tool(
        tools_policy_file,
        "send_mail",
        {"to": "alerts@example.internal", "subject": "s", "body": "b"},
    )
    assert allowed.allowed is True
    denied = tool(
        tools_policy_file, "send_mail", {"to": "attacker@evil.example", "subject": "s", "body": "b"}
    )
    assert denied.allowed is False
    assert ARGUMENT_HOST_NOT_ALLOWED in denied.reason_codes


def test_value_over_max_len_is_denied(tools_policy_file: Path) -> None:
    result = tool(tools_policy_file, "read_file", {"path": "/workspace/" + "a" * 600})
    assert result.allowed is False
    assert ARGUMENT_TOO_LONG in result.reason_codes
    assert result.argument_failures == ("argument_too_long:path",)


def test_missing_required_argument_is_denied(tools_policy_file: Path) -> None:
    result = tool(tools_policy_file, "read_file", {})
    assert result.allowed is False
    assert ARGUMENT_MISSING in result.reason_codes
    assert result.argument_failures == ("argument_missing:path",)


def test_undeclared_argument_is_denied(tools_policy_file: Path) -> None:
    result = tool(tools_policy_file, "read_file", {"path": "/workspace/a", "mode": "0777"})
    assert result.allowed is False
    assert ARGUMENT_UNDECLARED in result.reason_codes
    assert result.argument_failures == ("argument_undeclared:mode",)


def test_tool_with_arguments_but_no_schema_fails_closed(tools_policy_file: Path) -> None:
    result = tool(tools_policy_file, "call_mcp", {"server": "x"})
    assert result.allowed is False
    assert result.reason_codes == (ARGUMENT_SCHEMA_MISSING,)
    assert result.denied_by == ARGUMENT_SCHEMA_MISSING


def test_tool_with_no_schema_and_no_arguments_is_fine(tools_policy_file: Path) -> None:
    result = tool(tools_policy_file, "call_mcp", {})
    assert result.allowed is True


def test_non_tool_action_skips_argument_validation(tools_policy_file: Path) -> None:
    result = check(
        tools_policy_file,
        principal="dev",
        action=PolicyAction.RETRIEVAL,
        tool_arguments={"path": "/etc/passwd"},
    )
    assert result.allowed is True
    assert result.argument_failures == ()


# =================================================================================================
# The evidence rule
# =================================================================================================
@pytest.mark.parametrize(
    "kind",
    [
        "agent_directive_control_override",
        "agent_directive_adopt_as_instruction",
        "tool_call_instruction",
    ],
)
def test_each_injection_kind_denies_from_untrusted_data(tools_policy_file: Path, kind: str) -> None:
    result = tool(
        tools_policy_file,
        "read_file",
        {"path": "/workspace/report.md"},
        evidence=evidence(kind, Provenance.TOOL_RESULT),
    )
    assert result.allowed is False
    assert EVIDENCE_UNTRUSTED_INJECTION in result.reason_codes
    assert result.denied_by == EVIDENCE_UNTRUSTED_INJECTION
    assert result.evidence_used == (kind,)


@pytest.mark.parametrize(
    "provenance",
    [Provenance.TOOL_RESULT, Provenance.RETRIEVED_CHUNK, Provenance.REPOSITORY_FILE],
)
def test_each_untrusted_provenance_denies(tools_policy_file: Path, provenance: Provenance) -> None:
    result = tool(
        tools_policy_file,
        "read_file",
        {"path": "/workspace/report.md"},
        evidence=evidence("tool_call_instruction", provenance),
    )
    assert result.allowed is False
    assert EVIDENCE_UNTRUSTED_INJECTION in result.reason_codes


def test_operator_provenance_evidence_does_not_deny(tools_policy_file: Path) -> None:
    result = tool(
        tools_policy_file,
        "read_file",
        {"path": "/workspace/report.md"},
        evidence=evidence("agent_directive_control_override", Provenance.USER),
    )
    assert result.allowed is True
    assert result.evidence_used == ()


@pytest.mark.parametrize(
    "kind", ["agent_directive_control_suspended", "instruction_override", "secret_present"]
)
def test_non_injection_kinds_do_not_deny_even_from_untrusted_data(
    tools_policy_file: Path, kind: str
) -> None:
    result = tool(
        tools_policy_file,
        "read_file",
        {"path": "/workspace/report.md"},
        evidence=evidence(kind, Provenance.RETRIEVED_CHUNK),
    )
    assert result.allowed is True, result.reason_codes


def test_empty_evidence_does_not_turn_a_denial_into_an_allow(tools_policy_file: Path) -> None:
    result = tool(tools_policy_file, "read_file", {"path": "/etc/passwd"}, evidence=())
    assert result.allowed is False
    assert ARGUMENT_PATH_ESCAPE in result.reason_codes


def test_evidence_never_authorises_a_denied_call(tools_policy_file: Path) -> None:
    # Bad path *and* a benign operator evidence list: still denied, and evidence used for nothing.
    result = tool(
        tools_policy_file,
        "read_file",
        {"path": "/etc/passwd"},
        evidence=evidence("tool_call_instruction", Provenance.USER),
    )
    assert result.allowed is False
    assert result.evidence_used == ()


# =================================================================================================
# Cross-tenant tool calls (the write side of the same invariant)
# =================================================================================================
def test_cross_tenant_tool_call_is_denied(tools_policy_file: Path) -> None:
    result = tool(
        tools_policy_file,
        "db_write",
        {"table": "items", "row": "{}"},
        tenant="platform",
        target_tenant="other-team",
    )
    assert result.allowed is False
    assert CROSS_TENANT_DENIED in result.reason_codes


def test_same_tenant_tool_call_is_allowed(tools_policy_file: Path) -> None:
    result = tool(
        tools_policy_file,
        "db_write",
        {"table": "items", "row": "{}"},
        tenant="platform",
        target_tenant="platform",
    )
    assert result.allowed is True


# =================================================================================================
# Fail-closed loading still reports a denied_by
# =================================================================================================
def test_unavailable_policy_reports_denied_by(tmp_path: Path) -> None:
    result = engine(tmp_path / "missing.toml").check(
        PolicyCheckRequest(principal="dev", action=PolicyAction.TOOL_CALL, tool="read_file"),
        now=FIXED,
    )
    assert result.reason_codes == (POLICY_UNAVAILABLE,)
    assert result.denied_by == POLICY_UNAVAILABLE


# =================================================================================================
# /v1/agent/authorize end to end
# =================================================================================================
@pytest.fixture
def client(tools_policy_file: Path) -> TestClient:
    settings = GuardianSettings(policy=PolicySettings(file=tools_policy_file))
    return TestClient(create_app(settings))


def _body(**kwargs: object) -> dict[str, object]:
    return {"principal": "dev", "action": "tool_call", **kwargs}


def test_agent_authorize_allows_a_valid_call(client: TestClient) -> None:
    response = client.post(
        "/v1/agent/authorize",
        json=_body(tool="read_file", tool_arguments={"path": "src/app.py"}),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["decision"] == "allow"
    assert body["denied_by"] is None
    assert body["argument_failures"] == []
    assert body["evidence_used"] == []
    assert body["guardian_version"]
    assert body["check"]["allowed"] is True


def test_agent_authorize_reports_argument_failures(client: TestClient) -> None:
    response = client.post(
        "/v1/agent/authorize",
        json=_body(tool="read_file", tool_arguments={"path": "/etc/passwd"}),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["decision"] == "block"
    assert body["denied_by"] == "argument_path_escape"
    assert body["argument_failures"] == ["argument_path_escape:path"]


def test_agent_authorize_reports_evidence_used(client: TestClient) -> None:
    response = client.post(
        "/v1/agent/authorize",
        json=_body(
            tool="read_file",
            tool_arguments={"path": "/workspace/report.md"},
            evidence=[
                {
                    "segment_id": "seg-1",
                    "finding_kind": "tool_call_instruction",
                    "severity": "high",
                    "provenance": "tool_result",
                }
            ],
        ),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["decision"] == "block"
    assert body["denied_by"] == "evidence_untrusted_injection"
    assert body["evidence_used"] == ["tool_call_instruction"]


def test_agent_authorize_rejects_a_bad_severity(client: TestClient) -> None:
    response = client.post(
        "/v1/agent/authorize",
        json=_body(
            tool="read_file",
            tool_arguments={"path": "/workspace/a"},
            evidence=[
                {
                    "segment_id": "s",
                    "finding_kind": "tool_call_instruction",
                    "severity": "not-a-severity",
                    "provenance": "tool_result",
                }
            ],
        ),
    )
    assert response.status_code == 422


def test_policy_check_endpoint_is_unchanged(client: TestClient) -> None:
    response = client.post(
        "/v1/policy/check",
        json=_body(tool="read_file", tool_arguments={"path": "src/app.py"}),
    )
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"check", "guardian_version"}
    assert body["check"]["allowed"] is True
