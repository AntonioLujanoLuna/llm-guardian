"""Secret masking for content the guardian keeps.

An incident needs the *content* for its review to be useful twice: the reviewer judges what was
actually sent, and a reviewed case becomes a risk-memory precedent that a later identical segment
can match. What it must never keep is secret material. So the text is run through the secret
scanner -- always, whatever ``enable_secrets`` says for scanning -- and every reported span is
replaced by a marker naming the kind and carrying the scanner's correlatable hint, never the value.
"""

from __future__ import annotations

from .provenance import Provenance
from .scanners.secrets import SecretScanner

_SCANNER = SecretScanner()


def redact_secrets(text: str, provenance: Provenance, *, limit: int) -> str:
    """``text`` with every secret span masked, clipped to ``limit`` characters."""
    spans: list[tuple[int, int, str]] = []
    for finding in _SCANNER.scan(text, provenance):
        if finding.span is None:
            continue
        hint = finding.attributes.get("redacted", "")
        spans.append((finding.span[0], finding.span[1], f"[REDACTED:{finding.kind}:{hint}]"))

    parts: list[str] = []
    cursor = 0
    for start, end, marker in sorted(spans):
        if end <= cursor:
            continue  # wholly inside a span already masked
        parts.append(text[cursor : max(cursor, start)])
        parts.append(marker)
        cursor = end
    parts.append(text[cursor:])
    redacted = "".join(parts)
    return redacted if len(redacted) <= limit else redacted[: limit - 1] + "…"
