"""Hard authorization for action boundaries.

Every decision here is a comparison against a declared policy: a principal's capabilities, its
egress allow-list, its budgets and any explicit cross-tenant grant. No model output is ever
consulted, and an absent or unreadable policy file fails closed. Reason codes are stable machine
strings so a gateway can branch on them without parsing prose.
"""

from __future__ import annotations

import json
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
from .settings import PolicySettings

#: Stable reason codes. A gateway may log or branch on these; the order here is the evaluation
#: order, and a denied request keeps every code it earned (not just the first).
POLICY_UNAVAILABLE = "policy_unavailable"
UNKNOWN_PRINCIPAL = "unknown_principal"
TOOL_NOT_SPECIFIED = "tool_not_specified"
TOOL_NOT_PERMITTED = "tool_not_permitted"
CAPABILITY_NOT_PERMITTED = "capability_not_permitted"
EGRESS_NOT_ALLOWED = "egress_not_allowed"
CROSS_TENANT_DENIED = "cross_tenant_denied"
BUDGET_EXCEEDED = "budget_exceeded"
ALLOWED = "allowed"


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
                checked_at=clock(),
            )

        reasons: list[str] = []
        principal = data.principals.get(request.principal)
        if principal is None:
            if self._settings.fail_closed:
                reasons.append(UNKNOWN_PRINCIPAL)
            principal = data.defaults

        self._check_tool(request, principal, reasons)
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
        if request.action is not PolicyAction.RETRIEVAL:
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
    return _PolicyData(defaults=defaults, principals=principals, tenants=tenants)


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
