"""URL and egress-shaped signals: where content is trying to send a reader, or data.

These are *signals about the text*, not network policy. Deciding whether a host may be reached is
``policy.py``'s job and never a scanner's.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from ..models import Finding, OwaspCategory, Severity
from ..provenance import Provenance
from .base import build_finding

VERSION = "urls-1"

_RAW_IP = re.compile(r"\b(?:https?|ftp)://\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?")
_SCHEME = re.compile(r"\b(javascript|data|vbscript|file|jar|gopher)\s*:", re.IGNORECASE)
_LONG_B64_QUERY = re.compile(r"[?&][A-Za-z0-9_\-]{1,40}=([A-Za-z0-9+/%_=\-]{80,})")
_URL_HOST = re.compile(r"\bhttps?://([A-Za-z0-9.\-]+\.[A-Za-z]{2,})", re.IGNORECASE)

#: Destinations whose whole purpose is receiving or laundering data: pastebins, throwaway
#: webhooks and file drops. Matching a host is a signal, not a block -- see the module docstring.
_PASTE_HOSTS = frozenset(
    {
        "pastebin.com",
        "paste.ee",
        "hastebin.com",
        "ghostbin.co",
        "ghostbin.com",
        "dpaste.org",
        "dpaste.com",
        "termbin.com",
        "transfer.sh",
        "file.io",
        "0x0.st",
        "webhook.site",
        "requestbin.net",
        "requestbin.com",
        "pipedream.net",
        "ngrok.io",
        "ngrok-free.app",
        "hooks.slack.com",
        "discord.com",
        "t.me",
        "anonfiles.com",
    }
)


@dataclass(frozen=True)
class _Hit:
    kind: str
    severity: Severity
    owasp: tuple[OwaspCategory, ...]
    span: tuple[int, int]
    message: str
    attributes: dict[str, str]


class UrlScanner:
    """Raw-IP hosts, non-http schemes, exfil-shaped base64 query params and paste/webhook hosts."""

    name = "urls"
    version = VERSION

    def scan(self, text: str, provenance: Provenance) -> Sequence[Finding]:
        hits: list[_Hit] = []
        for m in _RAW_IP.finditer(text):
            hits.append(
                _Hit(
                    "raw_ip_host",
                    Severity.MEDIUM,
                    (OwaspCategory.LLM02, OwaspCategory.LLM10),
                    (m.start(), m.end()),
                    "A URL points at a bare IP address rather than a hostname.",
                    {"host": m.group(0).split("//", 1)[-1].split("/", 1)[0]},
                )
            )
        for m in _SCHEME.finditer(text):
            scheme = m.group(1).lower()
            severity = (
                Severity.HIGH if scheme in {"javascript", "data", "vbscript"} else Severity.MEDIUM
            )
            hits.append(
                _Hit(
                    "dangerous_scheme",
                    severity,
                    (OwaspCategory.LLM10,),
                    (m.start(), m.end()),
                    f"A '{scheme}:' URI scheme can execute or embed content.",
                    {"scheme": scheme},
                )
            )
        for m in _LONG_B64_QUERY.finditer(text):
            hits.append(
                _Hit(
                    "long_encoded_query",
                    Severity.MEDIUM,
                    (OwaspCategory.LLM02,),
                    (m.start(), m.end()),
                    "A query parameter carries a long opaque (possibly base64) payload.",
                    {"param_len": str(len(m.group(1)))},
                )
            )
        for m in _URL_HOST.finditer(text):
            host = m.group(1).lower()
            if host in _PASTE_HOSTS:
                hits.append(
                    _Hit(
                        "paste_or_webhook_host",
                        Severity.MEDIUM,
                        (OwaspCategory.LLM02,),
                        (m.start(), m.end()),
                        "A URL points at a pastebin, throwaway webhook or file-drop host.",
                        {"host": host},
                    )
                )
        findings = [
            build_finding(
                scanner=self.name,
                scanner_version=self.version,
                kind=hit.kind,
                severity=hit.severity,
                message=hit.message,
                owasp=hit.owasp,
                span=hit.span,
                attributes=hit.attributes,
            )
            for hit in hits
        ]
        findings.sort(key=lambda f: f.span or (0, 0))
        return findings
