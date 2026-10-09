"""Secret and credential detection.

A finding reports *that* a secret is present and *where* it sits (a span), plus a redacted hint
that lets a reviewer correlate two sightings without re-leaking the value. The secret itself never
enters a :class:`~llm_guardian.models.Finding` -- not in the message, not in the attributes.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass

from ..models import Finding, OwaspCategory, Severity
from ..provenance import Provenance
from .base import build_finding

VERSION = "secrets-1"

_PEM = re.compile(r"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----")
_JWT = re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{6,}\.[A-Za-z0-9_\-]{6,}\b")
_CONN = re.compile(
    r"\b(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|rediss|amqp|amqps|mssql|"
    r"clickhouse|kafka)://[^\s'\"<>]+",
    re.IGNORECASE,
)
_AWS_KEY = re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")
_OPENAI_KEY = re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_\-]{20,}\b")
_GITHUB_TOKEN = re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")
_SLACK_TOKEN = re.compile(r"\bxox[baprs]-[0-9A-Za-z-]{10,}\b")
_GOOGLE_KEY = re.compile(r"\bAIza[0-9A-Za-z_\-]{30,}\b")
_ASSIGNMENT = re.compile(
    r"(?m)^[ \t]*(?:export[ \t]+)?([A-Z][A-Z0-9_]{2,})\s*[:=]\s*['\"]?([^\s'\"#]{8,})['\"]?\s*$"
)

_SECRETISH_NAMES = re.compile(r"(SECRET|TOKEN|KEY|PASS|PWD|CRED|AUTH|PRIVATE|API)", re.IGNORECASE)


@dataclass(frozen=True)
class _Match:
    kind: str
    severity: Severity
    owasp: tuple[OwaspCategory, ...]
    start: int
    end: int
    hint: str
    message: str


def _redact(value: str) -> str:
    """A correlatable hint that never reveals the value: prefix, length and a short hash."""
    digest = hashlib.sha256(value.encode("utf-8", "surrogatepass")).hexdigest()[:10]
    prefix = value[:3] if len(value) >= 6 else ""
    return f"{prefix}****len{len(value)}sha{digest}"


def _looks_secret(value: str) -> bool:
    """Heuristic for a bare assignment value: long enough and not a plain lowercase word."""
    if len(value) < 12:
        return False
    has_upper = any(ch.isupper() for ch in value)
    has_digit = any(ch.isdigit() for ch in value)
    has_symbol = any(not ch.isalnum() for ch in value)
    return sum((has_upper, has_digit, has_symbol)) >= 2


class SecretScanner:
    """PEM blocks, JWTs, provider API keys, connection strings and ``.env``-shaped assignments."""

    name = "secrets"
    version = VERSION

    def scan(self, text: str, provenance: Provenance) -> Sequence[Finding]:
        matches: list[_Match] = []
        _collect_primitives(text, matches)
        for match in _iter_assignments(text):
            matches.append(match)

        findings: list[Finding] = []
        seen: set[tuple[str, int, int]] = set()
        for match in matches:
            key = (match.kind, match.start, match.end)
            if key in seen:
                continue
            seen.add(key)
            findings.append(
                build_finding(
                    scanner=self.name,
                    scanner_version=self.version,
                    kind=match.kind,
                    severity=match.severity,
                    message=match.message,
                    owasp=match.owasp,
                    span=(match.start, match.end),
                    attributes={"redacted": match.hint},
                )
            )
        findings.sort(key=lambda f: f.span or (0, 0))
        return findings


def _collect_primitives(text: str, out: list[_Match]) -> None:
    for m in _PEM.finditer(text):
        out.append(
            _Match(
                "private_key",
                Severity.CRITICAL,
                (OwaspCategory.LLM02,),
                m.start(),
                m.end(),
                _redact(m.group(0)),
                "A PEM private key block is present in the content.",
            )
        )
    for m in _CONN.finditer(text):
        out.append(
            _Match(
                "connection_string",
                Severity.HIGH,
                (OwaspCategory.LLM02,),
                m.start(),
                m.end(),
                _redact(m.group(0)),
                "A connection string that may embed credentials is present.",
            )
        )
    for m in _JWT.finditer(text):
        out.append(
            _Match(
                "jwt",
                Severity.HIGH,
                (OwaspCategory.LLM02,),
                m.start(),
                m.end(),
                _redact(m.group(0)),
                "A JSON Web Token is present in the content.",
            )
        )
    for pattern, kind, message in (
        (_AWS_KEY, "api_key", "An AWS access key identifier is present."),
        (_OPENAI_KEY, "api_key", "An OpenAI-style API key is present."),
        (_GITHUB_TOKEN, "api_key", "A GitHub token is present."),
        (_SLACK_TOKEN, "api_key", "A Slack token is present."),
        (_GOOGLE_KEY, "api_key", "A Google API key is present."),
    ):
        for m in pattern.finditer(text):
            out.append(
                _Match(
                    kind,
                    Severity.HIGH,
                    (OwaspCategory.LLM02,),
                    m.start(),
                    m.end(),
                    _redact(m.group(0)),
                    message,
                )
            )


def _iter_assignments(text: str) -> list[_Match]:
    """``.env``-shaped assignments, flagged only when the name or value looks secret."""
    out: list[_Match] = []
    for m in _ASSIGNMENT.finditer(text):
        name = m.group(1)
        value = m.group(2)
        if not (_SECRETISH_NAMES.search(name) or _looks_secret(value)):
            continue
        start, end = m.start(2), m.end(2)
        out.append(
            _Match(
                "env_assignment",
                Severity.MEDIUM,
                (OwaspCategory.LLM02,),
                start,
                end,
                _redact(value),
                "An environment-style assignment carries what looks like a secret value.",
            )
        )
    return out
