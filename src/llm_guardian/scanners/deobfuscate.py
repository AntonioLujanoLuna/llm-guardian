"""Deobfuscation: detect an encoding, decode it, and re-run the deterministic rules on the result.

Encodings are only *reported* when the decoded payload actually trips an inner rule -- decoding a
hash to noise is not a finding, and a legitimate percent-encoded URL must not become one. Every
finding that survives inherits the transform chain (``base64 -> rot13``) so a reviewer can see the
steps. Recursion is bounded by ``deobfuscation_max_depth``; a nested payload is not chased forever.
"""

from __future__ import annotations

import base64
import binascii
import codecs
import re
from collections.abc import Sequence
from dataclasses import dataclass
from urllib.parse import unquote

from ..models import Finding, OwaspCategory, Severity
from ..provenance import Provenance
from .base import Scanner, build_finding, severity_rank

VERSION = "deobfuscate-2"

#: Base64 / hex runs that stand alone between non-alphabet characters.
_B64 = re.compile(r"(?<![A-Za-z0-9+/=])([A-Za-z0-9+/]{16,}={0,2})(?![A-Za-z0-9+/=])")
_HEX = re.compile(r"(?<![0-9A-Fa-f])((?:[0-9A-Fa-f]{2}){10,})(?![0-9A-Fa-f])")
_PERCENT = re.compile(r"%[0-9A-Fa-f]{2}")

#: Characters that render as nothing (or as a direction change) but sit inside the words a rule
#: has to match: zero-width spaces and joiners, bidi overrides and isolates, word joiner, BOM.
_INVISIBLE = re.compile("[\u200b-\u200f\u202a-\u202e\u2060-\u2064\u2066-\u2069\ufeff\u00ad]")


@dataclass(frozen=True)
class _Decoded:
    """A decoded payload and the chain of transforms that produced it.

    ``recurse`` is False for the whole-text transforms (rot13, reversed): applying either twice
    returns the original, so recursing would report every text as its own obfuscation.
    """

    transform: str
    text: str
    span: tuple[int, int] | None
    recurse: bool = True


def _printable_ratio(value: str) -> float:
    if not value:
        return 0.0
    printable = sum(1 for ch in value if ch.isprintable() or ch in "\n\t\r")
    return printable / len(value)


def _try_b64(raw: str, span: tuple[int, int] | None) -> _Decoded | None:
    try:
        data = base64.b64decode(raw + "=" * (-len(raw) % 4), validate=False)
        text = data.decode("utf-8")
    except (binascii.Error, ValueError, UnicodeDecodeError):
        return None
    if len(text) < 6 or _printable_ratio(text) < 0.85:
        return None
    return _Decoded("base64", text, span)


def _try_hex(raw: str, span: tuple[int, int] | None) -> _Decoded | None:
    try:
        data = bytes.fromhex(raw)
        text = data.decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return None
    if len(text) < 6 or _printable_ratio(text) < 0.85:
        return None
    return _Decoded("hex", text, span)


class DeobfuscatingScanner:
    """Detect base64/hex/percent/rot13/reversed payloads and re-run ``inner`` on the result."""

    name = "deobfuscate"
    version = VERSION

    def __init__(self, inner: Scanner, max_depth: int) -> None:
        self._inner = inner
        self._max_depth = max(1, max_depth)

    def scan(self, text: str, provenance: Provenance) -> Sequence[Finding]:
        return _dedupe(self._walk(text, provenance, depth=0, chain=()))

    def _walk(
        self,
        text: str,
        provenance: Provenance,
        *,
        depth: int,
        chain: tuple[str, ...],
    ) -> list[Finding]:
        if depth >= self._max_depth:
            return []
        findings: list[Finding] = []
        for decoded in self._candidates(text):
            inner_findings = list(self._inner.scan(decoded.text, provenance))
            child_chain = (*chain, decoded.transform)
            if inner_findings:
                findings.append(_obfuscation_finding(decoded, child_chain, inner_findings))
                findings.extend(_propagate(decoded, child_chain, inner_findings))
            if decoded.recurse:
                findings.extend(
                    self._walk(decoded.text, provenance, depth=depth + 1, chain=child_chain)
                )
        return findings

    def _candidates(self, text: str) -> list[_Decoded]:
        out: list[_Decoded] = []
        for m in _B64.finditer(text):
            decoded = _try_b64(m.group(1), (m.start(1), m.end(1)))
            if decoded is not None:
                out.append(decoded)
        for m in _HEX.finditer(text):
            decoded = _try_hex(m.group(1), (m.start(1), m.end(1)))
            if decoded is not None:
                out.append(decoded)
        if _PERCENT.search(text):
            decoded_text = unquote(text)
            if decoded_text != text and _printable_ratio(decoded_text) >= 0.85:
                out.append(_Decoded("percent", decoded_text, None))
        if _INVISIBLE.search(text):
            # The words are all there; the characters between them are not. Strip and re-run, so a
            # smuggled instruction is seen the way the model would read it.
            stripped = _INVISIBLE.sub("", text)
            if stripped != text:
                out.append(_Decoded("zero_width_strip", stripped, None, recurse=False))
        rot = codecs.encode(text, "rot13")
        if rot != text:
            out.append(_Decoded("rot13", rot, None, recurse=False))
        reversed_text = text[::-1]
        if reversed_text != text:
            out.append(_Decoded("reversed", reversed_text, None, recurse=False))
        return out


def _obfuscation_finding(
    decoded: _Decoded, chain: tuple[str, ...], inner: list[Finding]
) -> Finding:
    top = max(inner, key=lambda f: severity_rank(f.severity))
    severity = (
        Severity.HIGH
        if severity_rank(top.severity) < severity_rank(Severity.HIGH)
        else top.severity
    )
    return build_finding(
        scanner=DeobfuscatingScanner.name,
        scanner_version=DeobfuscatingScanner.version,
        kind="obfuscated_payload",
        severity=severity,
        message="Content decoded from " + " -> ".join(chain) + " contains an attack signal.",
        owasp=(OwaspCategory.LLM01, OwaspCategory.LLM10),
        span=decoded.span,
        attributes={
            "transform_chain": " -> ".join(chain),
            "decoded_kinds": ",".join(sorted({f.kind for f in inner})),
        },
    )


def _propagate(decoded: _Decoded, chain: tuple[str, ...], inner: list[Finding]) -> list[Finding]:
    return [
        build_finding(
            scanner=DeobfuscatingScanner.name,
            scanner_version=DeobfuscatingScanner.version,
            kind=found.kind,
            severity=found.severity,
            message="[" + " -> ".join(chain) + "] " + found.message,
            owasp=found.owasp,
            span=decoded.span,
            attributes={**found.attributes, "transform_chain": " -> ".join(chain)},
        )
        for found in inner
    ]


def _dedupe(findings: list[Finding]) -> list[Finding]:
    best: dict[tuple[str, str], Finding] = {}
    for found in findings:
        key = (found.kind, found.attributes.get("transform_chain", ""))
        current = best.get(key)
        if current is None or severity_rank(found.severity) > severity_rank(current.severity):
            best[key] = found
    return sorted(best.values(), key=lambda f: (-severity_rank(f.severity), f.kind))
