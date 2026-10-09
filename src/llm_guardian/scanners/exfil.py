"""The exfiltration *pair*: a sensitive source, a transfer verb, and somewhere to send it.

``rules.py`` matches exfiltration as a phrasing ("read the .env and send it") and misses the same
request phrased as ordinary work. The signal that survives rephrasing is the **combination**: a
segment that names something sensitive, names a movement, and names a destination that exists to
receive data. Any one of the three alone is normal engineering prose -- "see the docs on
environment variables at https://platform.example/docs" has a source and a URL and is benign --
so this scanner fires only on all three, with the destination judged by
:func:`~llm_guardian.scanners.urls.is_untrusted_destination` (pastebin/webhook/collection hosts,
bare IPs, opaque query payloads) or by generic phrasing that defers the destination
("the external endpoint below").

The second shape is the same intent with the data hidden in a rendered resource: a reply that
embeds an image *whose URL contains* the context. Nothing in that URL is visible to the reader,
which is why it is reported even when no URL is present yet.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from ..models import Finding, OwaspCategory, Severity
from ..provenance import Provenance
from .base import build_finding
from .urls import is_untrusted_destination

VERSION = "exfil-1"

#: Things whose movement out of the environment is the event, in English and Spanish.
_SOURCE = re.compile(
    r"(?:\.env\b|environment\s+variables?|env\s+vars?|variables?\s+de\s+entorno|\bid_rsa\b|"
    r"\b\.ssh\b|authorized_keys|private\s+key|clave\s+privada|\bPEM\b|/etc/passwd|/etc/shadow|"
    r"credentials?|credenciales?|api[\s_-]?keys?|claves?\s+de\s+api|access\s+tokens?|"
    r"secrets?\b|secretos?\b|tenant\s+database\s+dump|volcado\s+de\s+(?:la\s+)?base|"
    r"\.config\b|\.aws\b|\.kube\b|directorio\s+de\s+configuraci)",
    re.IGNORECASE,
)

#: Movement verbs. Deliberately excludes "export as an environment variable", which is ordinary
#: setup advice; the verb has to be about sending something *away*.
_TRANSFER = re.compile(
    r"\b(?:send|sends|sent|sending|upload|uploads|uploaded|uploading|post|posts|posted|posting|"
    r"exfiltrate|exfiltrating|forward|forwards|forwarded|transmit|transmits|leak|leaks|leaked|"
    r"publish|publishes|published|email|emails|emailed|dump|dumps|dumped|paste|pastes|pasted|"
    r"env[i\u00ed]a(?:lo|la|los|las)?|env[i\u00ed]as|enviar|enviado|s[u\u00fa]be(?:la|lo|las|los)?|"
    r"subir|subido|manda(?:lo|la)?|mandar|mandado|reenv[i\u00ed]a(?:lo|la)?|"
    r"reenviar|filtra(?:lo|la)?|filtrar|filtrado|exfiltra|exfiltrar|volcar|publica|publicar|"
    r"pegar|pega)\b",
    re.IGNORECASE,
)

#: A destination that defers to something else in the message -- still a destination.
_GENERIC_DESTINATION = re.compile(
    r"\b(?:the\s+)?(?:external|remote|attacker|configured|given|target)\s+"
    r"(?:endpoint|server|host|domain|webhook|address|collection\s+point)|"
    r"\b(?:the\s+)?(?:endpoint|webhook|domain|address|url)\s+(?:below|configured|indicated|given)\b|"
    r"\bpaste\s+site\b|\bcollection\s+point\b|"
    r"\bdominio\s+externo\b|\bendpoint\s+externo\b|\bsitio\s+de\s+pegado\b|"
    r"\bservidor\s+(?:externo|remoto|indicado)\b|\bsitio\s+externo\b|"
    r"\bwebhook\s+(?:externo|remoto|indicado|propio)\b|"
    r"\b(?:indicado|configurado|proporcionado)\s+m[aá]s\s+abajo\b",
    re.IGNORECASE,
)

_URL = re.compile(r"https?://[^\s)\]\"'<>]+", re.IGNORECASE)

#: Rendering verbs and media nouns: content that becomes a fetched URL.
_RENDER = re.compile(
    r"\b(?:embed|embedded|embedding|render|rendered|include|includes|included|prepend|append|"
    r"add|insert|attach|reply\s+with|incrusta|incrustar|incrustado|a[ñn]ade|a[ñn]adir|incluye|"
    r"renderiza|inserta|adjunta|responde\s+con)\b",
    re.IGNORECASE,
)
_MEDIA = re.compile(
    r"\b(?:image|images|imagen|im[aá]genes|img\b|markdown|badge|banner)\b|!\[",
    re.IGNORECASE,
)
#: "whose URL contains / includes the ..." -- the payload is the source itself.
_MEDIA_CARRIES_SOURCE = re.compile(
    r"\b(?:whose\s+(?:url|link)\s+(?:contains?|includes?)|url\s+(?:that\s+)?(?:contains?|includes?)|"
    r"cuy[ao]\s+(?:url|enlace)\s+(?:contiene|incluye)|cuya\s+url\s+incluye|"
    r"que\s+(?:contenga|incluya))\b",
    re.IGNORECASE,
)


class ExfilChannelScanner:
    """Sensitive source + movement + collection destination, or source hidden in rendered media."""

    name = "exfil"
    version = VERSION

    def scan(self, text: str, provenance: Provenance) -> Sequence[Finding]:
        findings: list[Finding] = []
        source = _SOURCE.search(text)
        if source is None:
            return findings

        destination_span: tuple[int, int] | None = None
        destination_kind: str | None = None
        for url_match in _URL.finditer(text):
            if is_untrusted_destination(url_match.group(0)):
                destination_span = (url_match.start(), url_match.end())
                destination_kind = "untrusted_destination"
                break
        if destination_span is None:
            generic = _GENERIC_DESTINATION.search(text)
            if generic is not None:
                destination_span = (generic.start(), generic.end())
                destination_kind = "declared_destination"

        transfer = _TRANSFER.search(text)
        if destination_span is not None and transfer is not None:
            findings.append(
                build_finding(
                    scanner=self.name,
                    scanner_version=self.version,
                    kind="exfil_source_to_destination",
                    severity=Severity.HIGH,
                    message=(
                        "The segment pairs something sensitive with a movement verb and a "
                        "destination that exists to receive data."
                    ),
                    owasp=(OwaspCategory.LLM02, OwaspCategory.LLM03),
                    span=(source.start(), destination_span[1]),
                    attributes={
                        "source": source.group(0)[:40],
                        "destination_kind": destination_kind or "unknown",
                    },
                )
            )

        marker = _MEDIA_CARRIES_SOURCE.search(text)
        render = _RENDER.search(text)
        if marker is not None and _MEDIA.search(text) is not None and render is not None:
            findings.append(
                build_finding(
                    scanner=self.name,
                    scanner_version=self.version,
                    kind="exfil_to_rendered_media",
                    severity=Severity.HIGH,
                    message=(
                        "The segment asks for something sensitive to be carried in a rendered "
                        "image or link, where the reader cannot see it."
                    ),
                    owasp=(OwaspCategory.LLM02, OwaspCategory.LLM10),
                    span=(min(render.start(), marker.start()), marker.end()),
                    attributes={"source": source.group(0)[:40], "carrier": "rendered_media"},
                )
            )
        return findings
