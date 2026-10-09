"""Hard authorization for action boundaries.

Every decision here is a comparison against a declared policy: a principal's capabilities, its
egress allow-list, its budgets and any explicit cross-tenant grant. No model output is ever
consulted, and an absent or unreadable policy file fails closed. Reason codes are stable machine
strings so a gateway can branch on them without parsing prose.
"""

from __future__ import annotations

import json
import posixpath
import re
import tomllib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .models import (
    PolicyAction,
    PolicyCheck,
    PolicyCheckRequest,
    VerdictDecision,
)
from .provenance import Provenance
from .settings import PolicySettings

#: Stable reason codes. A gateway may log or branch on these; the order here is the evaluation
#: order, and a denied request keeps every code it earned (not just the first).
POLICY_UNAVAILABLE = "policy_unavailable"
UNKNOWN_PRINCIPAL = "unknown_principal"
TOOL_NOT_SPECIFIED = "tool_not_specified"
TOOL_NOT_PERMITTED = "tool_not_permitted"
ARGUMENT_MISSING = "argument_missing"
ARGUMENT_UNDECLARED = "argument_undeclared"
ARGUMENT_PATH_ESCAPE = "argument_path_escape"
ARGUMENT_SHELL_METACHAR = "argument_shell_metachar"
ARGUMENT_HOST_NOT_ALLOWED = "argument_host_not_allowed"
ARGUMENT_TOO_LONG = "argument_too_long"
ARGUMENT_PATTERN = "argument_pattern"
ARGUMENT_SCHEMA_MISSING = "argument_schema_missing"
EVIDENCE_UNTRUSTED_INJECTION = "evidence_untrusted_injection"
CAPABILITY_NOT_PERMITTED = "capability_not_permitted"
EGRESS_NOT_ALLOWED = "egress_not_allowed"
CROSS_TENANT_DENIED = "cross_tenant_denied"
BUDGET_EXCEEDED = "budget_exceeded"
ALLOWED = "allowed"

#: Provenance that marks content fetched from somewhere other than the console. A proposal whose
#: arguments came from one of these is untrusted data, whatever the text looks like.
UNTRUSTED_PROPOSAL_PROVENANCE: frozenset[Provenance] = frozenset(
    {Provenance.TOOL_RESULT, Provenance.RETRIEVED_CHUNK, Provenance.REPOSITORY_FILE}
)

#: The injection-class finding kinds. When a proposal's arguments came from untrusted data *and*
#: that data carried one of these, the call is a confused deputy and is denied. Deliberately three
#: kinds: the detector-only rule (deny on *any* injection finding) is measured against this in
#: ``docs/evaluations/policy-enforcement.md`` and is the reason for the narrow scope.
INJECTION_EVIDENCE_KINDS: frozenset[str] = frozenset(
    {
        "agent_directive_control_override",
        "agent_directive_adopt_as_instruction",
        "tool_call_instruction",
    }
)

#: Shell metacharacters that only make sense when a value is being interpreted by a shell. A value
#: declared free-form skips this check; everything else must not carry them.
_SHELL_METACHAR = re.compile(r"[;|&`$()\n]")


@dataclass(frozen=True)
class ArgumentConstraint:
    """Per-argument rules from ``[tools.<name>.constraints.<argument>]``."""

    path_within: str | None = None
    host_allow: tuple[str, ...] = ()
    pattern: str | None = None
    max_len: int | None = None
    free_form: bool = False


@dataclass(frozen=True)
class ToolSchema:
    """The declared argument surface of one tool: what may appear and what it must satisfy."""

    allowed: tuple[str, ...]
    required: tuple[str, ...]
    constraints: dict[str, ArgumentConstraint] = field(default_factory=dict)


@dataclass(frozen=True)
class Principal:
    """A declared principal after the ``[defaults]`` block has been merged in."""

    name: str
    capabilities: tuple[str, ...]
    egress_hosts: tuple[str, ...]
    budgets: dict[str, float]


@dataclass
class _PolicyData:
    defaults: Principal
    principals: dict[str, Principal] = field(default_factory=dict)
    tenants: dict[str, dict[str, Any]] = field(default_factory=dict)
    tools: dict[str, ToolSchema] = field(default_factory=dict)


class PolicyEngine:
    """Reads a TOML/JSON policy file once and answers :class:`PolicyCheckRequest`."""

    def __init__(self, settings: PolicySettings) -> None:
        self._settings = settings
        self._path = settings.file
        self._data: _PolicyData | None = None
        self._load_error: str | None = None

    # -- loading ------------------------------------------------------------------------------
    def loaded(self) -> bool:
        """Whether a policy file was read successfully. ``/v1/health`` reports this."""
        return self._load_error is None and self._ensure_loaded() is not None

    @property
    def load_error(self) -> str | None:
        self._ensure_loaded()
        return self._load_error

    def reload(self) -> None:
        self._data = None
        self._load_error = None
        self._ensure_loaded()

    def _ensure_loaded(self) -> _PolicyData | None:
        if self._data is not None or self._load_error is not None:
            return self._data
        if self._path is None:
            self._load_error = "no policy file configured"
            return None
        try:
            raw = _read_policy(self._path)
        except (OSError, ValueError, tomllib.TOMLDecodeError) as exc:
            self._load_error = f"{type(exc).__name__}: {exc}"
            return None
        self._data = _parse_policy(raw)
        return self._data

    # -- the check ----------------------------------------------------------------------------
    def check(
        self, request: PolicyCheckRequest, *, now: Callable[[], datetime] | None = None
    ) -> PolicyCheck:
        clock = now or _utcnow
        data = self._ensure_loaded()
        if data is None:
            return PolicyCheck(
                decision=VerdictDecision.BLOCK,
                allowed=False,
                reason_codes=(POLICY_UNAVAILABLE,),
                capabilities=(),
                egress_hosts=(),
                budgets={},
                denied_by=POLICY_UNAVAILABLE,
                checked_at=clock(),
            )

        reasons: list[str] = []
        principal = data.principals.get(request.principal)
        if principal is None:
            if self._settings.fail_closed:
                reasons.append(UNKNOWN_PRINCIPAL)
            principal = data.defaults

        self._check_tool(request, principal, reasons)
        argument_failures = self._check_arguments(request, data, reasons)
        evidence_used = self._check_evidence(request, reasons)
        self._check_capabilities(request, principal, reasons)
        self._check_egress(request, principal, reasons)
        self._check_cross_tenant(request, data, reasons)
        self._check_budget(request, principal, reasons)

        allowed = not reasons
        return PolicyCheck(
            decision=VerdictDecision.ALLOW if allowed else VerdictDecision.BLOCK,
            allowed=allowed,
            reason_codes=tuple(reasons) if reasons else (ALLOWED,),
            capabilities=principal.capabilities,
            egress_hosts=principal.egress_hosts,
            budgets=dict(principal.budgets),
            denied_by=None if allowed else reasons[0],
            argument_failures=tuple(argument_failures),
            evidence_used=tuple(evidence_used),
            checked_at=clock(),
        )

    # -- individual rules ---------------------------------------------------------------------
    def _check_tool(
        self, request: PolicyCheckRequest, principal: Principal, reasons: list[str]
    ) -> None:
        if request.action is not PolicyAction.TOOL_CALL:
            return
        if request.tool is None:
            if self._settings.fail_closed:
                reasons.append(TOOL_NOT_SPECIFIED)
            return
        if request.tool not in principal.capabilities and self._settings.fail_closed:
            reasons.append(TOOL_NOT_PERMITTED)

    def _check_arguments(
        self,
        request: PolicyCheckRequest,
        data: _PolicyData,
        reasons: list[str],
    ) -> list[str]:
        """Validate ``tool_arguments`` against the declared schema. Deterministic, fail-closed.

        Returns the detailed failures (``"<code>:<argument>"``) for the caller to publish; the bare
        stable ``argument_*`` code is added to ``reasons`` once per kind so a gateway can branch on
        it without parsing the argument name out.
        """
        if request.action is not PolicyAction.TOOL_CALL or request.tool is None:
            return []
        arguments = request.tool_arguments
        schema = data.tools.get(request.tool)
        if schema is None:
            # A tool with arguments but no declared schema cannot be validated: fail closed.
            if arguments:
                return _record(reasons, ARGUMENT_SCHEMA_MISSING)
            return []

        failures: list[str] = []
        allowed = set(schema.allowed)
        for name, value in arguments.items():
            if name not in allowed:
                failures.append(f"{ARGUMENT_UNDECLARED}:{name}")
                continue
            constraint = schema.constraints.get(name)
            if constraint is not None:
                failures.extend(_constraint_failures(name, value, constraint))
        for name in schema.required:
            if name not in arguments:
                failures.append(f"{ARGUMENT_MISSING}:{name}")

        for failure in failures:
            _record(reasons, failure.split(":", 1)[0])
        return failures

    def _check_evidence(self, request: PolicyCheckRequest, reasons: list[str]) -> list[str]:
        """Deny a proposal whose arguments came from untrusted data that carried an injection.

        The confused-deputy case: a tool result, a retrieved chunk or a repository file that names a
        tool and its arguments is an instruction to the agent from a channel that may only carry
        data. Evidence never authorises -- an empty list changes nothing, and a finding from the
        operator's own segment is not treated as an injection. Only the three kinds in
        :data:`INJECTION_EVIDENCE_KINDS` and only untrusted provenance trigger the rule.
        """
        used: list[str] = []
        for ref in request.evidence:
            if (
                ref.provenance in UNTRUSTED_PROPOSAL_PROVENANCE
                and ref.finding_kind in INJECTION_EVIDENCE_KINDS
                and ref.finding_kind not in used
            ):
                used.append(ref.finding_kind)
        if used:
            _record(reasons, EVIDENCE_UNTRUSTED_INJECTION)
        return used

    def _check_capabilities(
        self, request: PolicyCheckRequest, principal: Principal, reasons: list[str]
    ) -> None:
        missing = [
            cap for cap in request.requested_capabilities if cap not in principal.capabilities
        ]
        if missing:
            reasons.append(CAPABILITY_NOT_PERMITTED)

    def _check_egress(
        self, request: PolicyCheckRequest, principal: Principal, reasons: list[str]
    ) -> None:
        for host in request.egress_hosts:
            if not _host_allowed(host, principal.egress_hosts):
                reasons.append(EGRESS_NOT_ALLOWED)
                return

    def _check_cross_tenant(
        self,
        request: PolicyCheckRequest,
        data: _PolicyData,
        reasons: list[str],
    ) -> None:
        # A read is a retrieval; a write is a tool call. Both may name a target tenant, and both
        # must be denied unless the target tenant lists the principal with an explicit grant.
        if request.action not in (PolicyAction.RETRIEVAL, PolicyAction.TOOL_CALL):
            return
        target = request.target_tenant
        if target is None or target == request.tenant:
            return
        tenant = data.tenants.get(target, {})
        grants = tenant.get("allowed_cross_tenant_reads", [])
        if request.principal not in grants:
            reasons.append(CROSS_TENANT_DENIED)

    def _check_budget(
        self, request: PolicyCheckRequest, principal: Principal, reasons: list[str]
    ) -> None:
        for name, requested in request.requested_budget.items():
            limit = principal.budgets.get(name)
            if limit is None:
                if self._settings.fail_closed:
                    reasons.append(BUDGET_EXCEEDED)
                continue
            if requested > limit:
                reasons.append(BUDGET_EXCEEDED)
                return


def _host_allowed(host: str, allow_list: Sequence[str]) -> bool:
    host = host.lower()
    for entry in allow_list:
        candidate = entry.lower()
        if candidate.startswith("*."):
            if host.endswith(candidate[1:]) or host == candidate[2:]:
                return True
        elif host == candidate:
            return True
    return False


def _value_allowed(value: str, allow_list: Sequence[str]) -> bool:
    """Whether a host/recipient value is on the allow-list.

    Broader than :func:`_host_allowed`: a value may be a bare host, a subdomain, or a recipient
    address (``alerts@example.internal`` matches the declared domain ``example.internal``).
    """
    candidate = value.lower()
    for entry in allow_list:
        allowed = entry.lower()
        if allowed.startswith("*."):
            if candidate.endswith(allowed[1:]) or candidate == allowed[2:]:
                return True
        elif (
            candidate == allowed
            or candidate.endswith("@" + allowed)
            or candidate.endswith("." + allowed)
        ):
            return True
    return False


def _record(reasons: list[str], code: str) -> list[str]:
    """Add a stable reason code once, preserving evaluation order. Returns ``[code]``."""
    if code not in reasons:
        reasons.append(code)
    return [code]


def _path_escapes(value: str, root: str) -> bool:
    """Whether ``value`` resolves outside ``root`` once normalised.

    Relative paths are joined to the root first, so ``sub/../file`` stays inside while ``../file``
    does not; absolute paths must sit within the root; a leading ``~`` is always outside, because a
    home directory is never a declared workspace root.
    """
    if value.startswith("~"):
        return True
    root_norm = posixpath.normpath(root)
    if value.startswith("/"):
        candidate = posixpath.normpath(value)
    else:
        candidate = posixpath.normpath(posixpath.join(root_norm, value))
    if candidate == root_norm:
        return False
    return not candidate.startswith(root_norm + "/")


def _constraint_failures(name: str, value: Any, constraint: ArgumentConstraint) -> list[str]:
    """Every per-argument violation, in a fixed order, each as ``"<code>:<argument>"``."""
    text = value if isinstance(value, str) else str(value)
    failures: list[str] = []
    if constraint.max_len is not None and len(text) > constraint.max_len:
        failures.append(f"{ARGUMENT_TOO_LONG}:{name}")
    if not constraint.free_form and _SHELL_METACHAR.search(text):
        failures.append(f"{ARGUMENT_SHELL_METACHAR}:{name}")
    if constraint.path_within is not None and _path_escapes(text, constraint.path_within):
        failures.append(f"{ARGUMENT_PATH_ESCAPE}:{name}")
    if constraint.host_allow and not _value_allowed(text, constraint.host_allow):
        failures.append(f"{ARGUMENT_HOST_NOT_ALLOWED}:{name}")
    if constraint.pattern is not None and re.fullmatch(constraint.pattern, text) is None:
        failures.append(f"{ARGUMENT_PATTERN}:{name}")
    return failures


def _read_policy(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    parsed = json.loads(text) if path.suffix.lower() == ".json" else tomllib.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("policy file must decode to a table/object")
    return parsed


def _parse_policy(raw: dict[str, Any]) -> _PolicyData:
    defaults_raw = raw.get("defaults", {})
    defaults = _principal("defaults", defaults_raw)
    principals: dict[str, Principal] = {}
    for name, body in dict(raw.get("principals", {})).items():
        principals[name] = _merge(defaults, _principal(name, body))
    tenants = {str(name): dict(body) for name, body in dict(raw.get("tenants", {})).items()}
    tools = {str(name): _parse_tool(body) for name, body in dict(raw.get("tools", {})).items()}
    return _PolicyData(defaults=defaults, principals=principals, tenants=tenants, tools=tools)


def _parse_tool(body: Any) -> ToolSchema:
    """Parse a ``[tools.<name>]`` table, including its per-argument ``constraints`` subtree."""
    data = dict(body) if isinstance(body, dict) else {}
    constraints: dict[str, ArgumentConstraint] = {}
    raw_constraints = data.get("constraints", {})
    if isinstance(raw_constraints, dict):
        for argument, cbody in raw_constraints.items():
            constraints[str(argument)] = _argument_constraint(cbody)
    free_form = tuple(str(a) for a in _as_sequence(data.get("free_form", [])))
    for argument in free_form:
        base = constraints.get(argument, ArgumentConstraint())
        constraints[argument] = ArgumentConstraint(
            path_within=base.path_within,
            host_allow=base.host_allow,
            pattern=base.pattern,
            max_len=base.max_len,
            free_form=True,
        )
    required = tuple(str(a) for a in _as_sequence(data.get("required", [])))
    if "allowed" in data:
        allowed = tuple(str(a) for a in _as_sequence(data.get("allowed", [])))
    else:
        allowed = tuple(sorted(set(required) | set(constraints)))
    return ToolSchema(allowed=allowed, required=required, constraints=constraints)


def _argument_constraint(body: Any) -> ArgumentConstraint:
    data = dict(body) if isinstance(body, dict) else {}
    max_len = data.get("max_len")
    pattern = data.get("pattern")
    path_within = data.get("path_within")
    return ArgumentConstraint(
        path_within=str(path_within) if path_within is not None else None,
        host_allow=tuple(str(h) for h in _as_sequence(data.get("host_allow", []))),
        pattern=str(pattern) if pattern is not None else None,
        max_len=int(max_len) if max_len is not None else None,
    )


def _as_sequence(value: Any) -> list[Any]:
    """Normalise a TOML value that should be a list, tolerating a bare scalar."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _principal(name: str, body: Any) -> Principal:
    data = dict(body) if isinstance(body, dict) else {}
    return Principal(
        name=name,
        capabilities=tuple(str(c) for c in data.get("capabilities", [])),
        egress_hosts=tuple(str(h) for h in data.get("egress_hosts", [])),
        budgets={str(k): float(v) for k, v in dict(data.get("budgets", {})).items()},
    )


def _merge(defaults: Principal, principal: Principal) -> Principal:
    """A principal inherits the default capabilities, egress hosts and budgets unless declared."""
    capabilities = principal.capabilities or defaults.capabilities
    egress = principal.egress_hosts if principal.egress_hosts else defaults.egress_hosts
    budgets = {**defaults.budgets, **principal.budgets}
    return Principal(
        name=principal.name,
        capabilities=capabilities,
        egress_hosts=egress,
        budgets=budgets,
    )


def _utcnow() -> datetime:
    from datetime import UTC

    return datetime.now(UTC)
