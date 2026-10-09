"""URL and egress-shaped signals: where content is trying to send a reader, or data.

These are *signals about the text*, not network policy. Deciding whether a host may be reached is
``policy.py``'s job and never a scanner's.

Two families live here. The first is a URL whose *shape* is suspicious on its own (a bare IP, a
``javascript:`` scheme, an opaque base64 query parameter, a pastebin/webhook destination). The
second is the markdown or HTML **image beacon**: ``![alt](https://...?d=<payload>)`` renders
nothing a reader can inspect, and is the standard way an injected instruction exfiltrates context
through a reply the user asked for. A beacon is judged by its destination, which is why
:func:`is_untrusted_destination` is shared with ``exfil.py``: one definition of "a place that
exists to receive data", used by both.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from ..models import Finding, OwaspCategory, Severity
from ..provenance import Provenance
from .base import build_finding

VERSION = "urls-2"

_RAW_IP = re.compile(r"\b(?:https?|ftp)://\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?")
_SCHEME = re.compile(r"\b(javascript|data|vbscript|file|jar|gopher)\s*:", re.IGNORECASE)
_LONG_B64_QUERY = re.compile(r"[?&][A-Za-z0-9_\-]{1,40}=([A-Za-z0-9+/%_=\-]{80,})")
_URL_HOST = re.compile(r"\bhttps?://([A-Za-z0-9.\-]+\.[A-Za-z]{2,})", re.IGNORECASE)

_MD_IMAGE = re.compile(r"!\[[^\]]*\]\(\s*(https?://[^)\s]+)")
_HTML_IMAGE = re.compile(r"<img\b[^>]*?\bsrc\s*=\s*['\"](https?://[^'\"]+)", re.IGNORECASE)
_MD_LINK = re.compile(r"(?<!!)\[[^\]]*\]\(\s*(https?://[^)\s]+)")

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

#: Host *labels* that name a collection point rather than a service. A real attacker on a branded
#: domain defeats this, which is why it is one signal among several and never a verdict.
_UNTRUSTED_HOST_MARKERS: tuple[str, ...] = (
    "webhook",
    "paste",
    "collect",
    "exfil",
    "drop",
    "ingest",
    "hook",
    "leak",
    "attacker",
    "evil",
    "beeceptor",
    "oast",
    "interact",
    "requestbin",
    "transfer",
    "upload",
)

#: A query parameter whose value is a payload placeholder or a secret-shaped word.
_PAYLOAD_QUERY = re.compile(
    r"[?&][^=\s&]{1,40}=[^&\s]{0,40}"
    r"(?:\{[^}\n]{1,60}\}|SECRET|TOKEN|API[_-]?KEY|PASSWORD|CREDENTIALS?|contenido|contexto)"
    r"[^&\s]{0,40}",
    re.IGNORECASE,
)


def _host_of(url: str) -> str:
    match = re.match(r"https?://([^/:?#\s]+)", url, re.IGNORECASE)
    return match.group(1).lower() if match else ""


def is_untrusted_destination(url: str) -> bool:
    """Whether a URL's destination looks like a collection point rather than an ordinary link.

    True for a pastebin-style host, a host whose own name says it collects data, a bare IP, or a
    query string carrying an opaque blob or a payload placeholder.
    """
    host = _host_of(url)
    if not host:
        return False
    if re.fullmatch(r"\d{1,3}(?:\.\d{1,3}){3}", host):
        return True
    if host in _PASTE_HOSTS:
        return True
    labels = host.split(".")
    for label in labels:
        if any(label == marker or label.startswith(marker) for marker in _UNTRUSTED_HOST_MARKERS):
            return True
    return bool(_LONG_B64_QUERY.search(url) or _PAYLOAD_QUERY.search(url))


@dataclass(frozen=True)
class _Hit:
    kind: str
    severity: Severity
    owasp: tuple[OwaspCategory, ...]
    span: tuple[int, int]
    message: str
    attributes: dict[str, str]


class UrlScanner:
    """Raw-IP hosts, non-http schemes, exfil-shaped queries, paste hosts, and rendered images
    whose destination is a collection point."""

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
        for pattern, kind, message in (
            (
                _MD_IMAGE,
                "markdown_image_beacon",
                "A markdown image renders content from a URL that looks like a collection point; "
                "image URLs are not visible to the reader, so they carry data out.",
            ),
            (
                _HTML_IMAGE,
                "html_image_beacon",
                "An HTML <img> renders content from a URL that looks like a collection point; "
                "image URLs are not visible to the reader, so they carry data out.",
            ),
        ):
            for m in pattern.finditer(text):
                url = m.group(1)
                if not is_untrusted_destination(url):
                    continue
                hits.append(
                    _Hit(
                        kind,
                        Severity.HIGH,
                        (OwaspCategory.LLM02, OwaspCategory.LLM10),
                        (m.start(), m.end()),
                        message,
                        {"host": _host_of(url)},
                    )
                )
        for m in _MD_LINK.finditer(text):
            url = m.group(1)
            if not is_untrusted_destination(url):
                continue
            hits.append(
                _Hit(
                    "link_to_collection_point",
                    Severity.MEDIUM,
                    (OwaspCategory.LLM02,),
                    (m.start(), m.end()),
                    "A link points at a destination that looks like a collection point.",
                    {"host": _host_of(url)},
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
