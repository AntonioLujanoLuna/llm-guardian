"""Hidden-text detection: the characters and constructs a reader cannot see.

Zero-width joiners, bidirectional overrides and the Unicode tag block let an attacker smuggle
instructions past a human reviewer while a model still reads them. Mixed Latin/Cyrillic inside a
single token is the fingerprint of a homoglyph attack, and an HTML comment that carries an
instruction is a directive disguised as markup.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from ..models import Finding, OwaspCategory, Severity
from ..provenance import Provenance
from .base import build_finding

VERSION = "invisible-1"

_ZERO_WIDTH = re.compile("[\u200b\u200c\u200d\u2060\ufeff]")
_BIDI = re.compile("[\u202a-\u202e\u2066-\u2069]")
_TAG_BLOCK = re.compile("[\U000e0000-\U000e007f]")
#: A single token that mixes Latin and Cyrillic letters is a homoglyph substitution.
_MIXED_SCRIPT = re.compile(r"\b\w*(?:[a-zA-Z]\w*[\u0400-\u04ff]|[\u0400-\u04ff]\w*[a-zA-Z])\w*\b")
_HTML_COMMENT = re.compile(r"<!--(.*?)-->", re.DOTALL)
_COMMENT_INSTRUCTION = re.compile(
    r"\b(ignore|disregard|override|system|instruction|prompt|assistant|tool|execute|"
    r"reveal|secret|password|token)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class _Hit:
    kind: str
    severity: Severity
    owasp: tuple[OwaspCategory, ...]
    span: tuple[int, int]
    message: str
    attributes: dict[str, str]


class InvisibleTextScanner:
    """Zero-width characters, bidi overrides, tag-block smuggling, homoglyphs, hidden comments."""

    name = "invisible"
    version = VERSION

    def scan(self, text: str, provenance: Provenance) -> Sequence[Finding]:
        hits: list[_Hit] = []
        _emit_count(
            hits,
            _ZERO_WIDTH,
            text,
            "zero_width_character",
            Severity.MEDIUM,
            (OwaspCategory.LLM08,),
            "Zero-width characters are hidden inside the text.",
        )
        _emit_count(
            hits,
            _BIDI,
            text,
            "bidi_override",
            Severity.HIGH,
            (OwaspCategory.LLM01, OwaspCategory.LLM08),
            "A bidirectional override can reorder how the text is displayed.",
        )
        _emit_count(
            hits,
            _TAG_BLOCK,
            text,
            "unicode_tag_smuggling",
            Severity.CRITICAL,
            (OwaspCategory.LLM01, OwaspCategory.LLM08),
            "Unicode tag-block characters are hidden; this is a smuggling technique.",
        )
        _emit_count(
            hits,
            _MIXED_SCRIPT,
            text,
            "mixed_script_token",
            Severity.HIGH,
            (OwaspCategory.LLM01,),
            "A token mixes Latin and Cyrillic letters (homoglyph substitution).",
        )
        for m in _HTML_COMMENT.finditer(text):
            if _COMMENT_INSTRUCTION.search(m.group(1)):
                hits.append(
                    _Hit(
                        "hidden_html_instruction",
                        Severity.HIGH,
                        (OwaspCategory.LLM01, OwaspCategory.LLM08),
                        (m.start(), m.end()),
                        "An HTML comment hides a phrasing that looks like an instruction.",
                        {},
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


def _emit_count(
    hits: list[_Hit],
    pattern: re.Pattern[str],
    text: str,
    kind: str,
    severity: Severity,
    owasp: tuple[OwaspCategory, ...],
    message: str,
) -> None:
    matches = list(pattern.finditer(text))
    if not matches:
        return
    first = matches[0]
    hits.append(
        _Hit(
            kind,
            severity,
            owasp,
            (first.start(), first.end()),
            message,
            {"count": str(len(matches))},
        )
    )
