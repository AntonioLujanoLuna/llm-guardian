"""Fast, deterministic regex rules for instruction-level attacks, in English and Spanish.

The hard part is not matching an attack phrase; it is *not* matching security documentation that
quotes one. Every match therefore carries a context test: a phrase inside quotation marks and
surrounded by discussion vocabulary ("attack", "example", "vulnerability", "research" ...) is
reported at a low severity -- it is an artifact under discussion, not an instruction being
issued. A directive that appears in untrusted data (a tool result, a retrieved chunk) is raised a
step instead, because instructions there are anomalous by construction.

Nothing here produces a verdict. RuleScanner emits findings; ``aggregate`` and ``verdicts`` decide
what they mean.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from ..models import Finding, OwaspCategory, Severity
from ..provenance import Provenance, TrustLevel, trust_of
from .base import (
    build_finding,
    cap_severity,
    raise_severity,
    severity_rank,
)

VERSION = "rules-1"


@dataclass(frozen=True)
class Rule:
    """One named pattern. Several rules may share a ``kind``; the strongest match wins."""

    name: str
    kind: str
    severity: Severity
    owasp: tuple[OwaspCategory, ...]
    pattern: re.Pattern[str]
    message: str


def _p(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


def default_rules() -> list[Rule]:
    """The shipped rule set. Kept as a function so a caller may extend or replace it."""
    return [
        # --- Instruction override (LLM01, LLM10) --------------------------------------------
        Rule(
            "instruction_override.en",
            "instruction_override",
            Severity.HIGH,
            (OwaspCategory.LLM01,),
            _p(
                r"\b(ignore|disregard|forget|override)\s+"
                r"(all\s+|any\s+|the\s+|your\s+)*"
                r"(previous|prior|above|earlier|preceding|system)\s+"
                r"(instruction|prompt|message|context|direction|rule|guideline)s?\b"
            ),
            "An instruction tries to override earlier or system instructions.",
        ),
        Rule(
            "instruction_override.en2",
            "instruction_override",
            Severity.HIGH,
            (OwaspCategory.LLM01,),
            _p(r"\bignore\s+your\s+(instruction|rule|guideline|system\s+prompt)s?\b"),
            "An instruction tries to override the model's own rules.",
        ),
        Rule(
            "instruction_override.es",
            "instruction_override",
            Severity.HIGH,
            (OwaspCategory.LLM01,),
            _p(
                r"\b(ignora|olvida|desestima|omite)\s+"
                r"(todas?\s+las?\s+|las?\s+|cualquier\s+|tus?\s+)*"
                r"(instrucciones?|indicaciones?|reglas?|prompts?|pol\u00edtica|pol\u00edticas)"
                r"(\s+(anteriores?|previas?|de\s+arriba|del?\s+sistema))?"
            ),
            "Una instrucci\u00f3n intenta anular instrucciones anteriores o del sistema.",
        ),
        Rule(
            "instruction_override.forget",
            "instruction_override",
            Severity.HIGH,
            (OwaspCategory.LLM01,),
            _p(r"\bforget\s+(everything|all|anything)(\s+(you\s+)?(were|was|told|know))?\b"),
            "An instruction tries to make the model forget what it was told.",
        ),
        # --- System-prompt exfiltration (LLM01, LLM08) --------------------------------------
        Rule(
            "system_prompt_exfil.en",
            "system_prompt_exfiltration",
            Severity.HIGH,
            (OwaspCategory.LLM01, OwaspCategory.LLM08),
            _p(
                r"\b(print|show|reveal|display|repeat|output|leak|give|dump|translate)\s+"
                r"(me\s+)?(your\s+|the\s+)?"
                r"(system\s+prompt|system\s+message|initial\s+prompt|initial\s+instructions|"
                r"instructions|guidelines)\b"
            ),
            "A request tries to make the model reveal its system prompt or instructions.",
        ),
        Rule(
            "system_prompt_exfil.en2",
            "system_prompt_exfiltration",
            Severity.HIGH,
            (OwaspCategory.LLM01, OwaspCategory.LLM08),
            _p(
                r"\brepeat\s+(everything|all(\s+the)?\s+(text|content)|the\s+text)\s+(above|before|prior)\b"
            ),
            "A request tries to make the model echo everything above it.",
        ),
        Rule(
            "system_prompt_exfil.es",
            "system_prompt_exfiltration",
            Severity.HIGH,
            (OwaspCategory.LLM01, OwaspCategory.LLM08),
            _p(
                r"\b(imprime|mu[e\u00e9]stra(?:me)?|revela|repite|dame|ens\u00e9\u00f1ame|vuelca|traduce)\s+"
                r"(tu|el|la|las|los)?\s*"
                r"(prompt|instrucciones?|mensaje\s+del?\s+sistema|directrices?)"
            ),
            "Una petici\u00f3n intenta revelar el prompt del sistema o las instrucciones.",
        ),
        # --- Delimiter / role spoofing (LLM01, LLM10) ---------------------------------------
        Rule(
            "role_spoof.control",
            "delimiter_role_spoofing",
            Severity.HIGH,
            (OwaspCategory.LLM01, OwaspCategory.LLM10),
            _p(
                r"</?(system|assistant|tool|function)\s*>"
                r"|<\|\s*(im_start|im_end|endoftext|start_header_id|end_header_id)\s*\|>"
                r"|\[/?INST\]|<<SYS>>|<</SYS>>"
                r"|<\|(system|user|assistant)\|>"
            ),
            "A raw chat delimiter or control token is present in the content.",
        ),
        Rule(
            "role_spoof.header",
            "delimiter_role_spoofing",
            Severity.HIGH,
            (OwaspCategory.LLM01,),
            _p(r"(?m)^\s*#{2,}\s*(system|assistant|user|developer)\s*:"),
            "A markdown role header tries to impersonate a system or assistant turn.",
        ),
        Rule(
            "role_spoof.plain",
            "delimiter_role_spoofing",
            Severity.LOW,
            (OwaspCategory.LLM01,),
            _p(r"(?m)^\s*(system|assistant|developer)\s*:"),
            "A bare role label appears at the start of a line.",
        ),
        # --- Tool-call injection (LLM03, LLM01) ---------------------------------------------
        Rule(
            "tool_call.tag",
            "tool_call_injection",
            Severity.HIGH,
            (OwaspCategory.LLM03, OwaspCategory.LLM01),
            _p(
                r"</?tool_call>|</?tool_use>|</?function_call>|\"tool_call\"\s*:|\"function_call\"\s*:"
            ),
            "A tool/function call marker appears inside data that should be inert.",
        ),
        Rule(
            "tool_call.shell",
            "tool_call_injection",
            Severity.HIGH,
            (OwaspCategory.LLM03, OwaspCategory.LLM10),
            _p(
                r"\b(call|invoke|execute|run|use)\b[^\n]{0,60}\b"
                r"(bash|shell|terminal|exec|command|tool|function)\b"
                r"[^\n]{0,120}"
                r"(rm\s+-rf|curl\s|wget\s|powershell|bash\s+-c|sh\s+-c|/etc/passwd|\.env|chmod\s+777)"
            ),
            "Content tries to trigger a shell or tool execution.",
        ),
        Rule(
            "tool_call.destructive",
            "tool_call_injection",
            Severity.HIGH,
            (OwaspCategory.LLM03, OwaspCategory.LLM10),
            _p(r"\brm\s+-rf\s+/|\b(curl|wget)\s+[^\s]+[^\n]{0,40}\|\s*(ba|z?|d)?sh\b"),
            "Content carries a destructive shell command or a curl-pipe-to-shell.",
        ),
        # --- Policy / safety bypass (LLM01, LLM03) ------------------------------------------
        Rule(
            "policy_bypass.en",
            "policy_bypass",
            Severity.HIGH,
            (OwaspCategory.LLM01, OwaspCategory.LLM03),
            _p(
                r"\b(ignore|bypass|disable|circumvent|turn\s+off)\b[^\n]{0,40}?"
                r"\b(security|safety|content\s+moderation|moderation|guardrail|filter|"
                r"restriction|control|policy|policies)\b"
            ),
            "An instruction tries to bypass a security or safety control.",
        ),
        Rule(
            "policy_bypass.en2",
            "policy_bypass",
            Severity.HIGH,
            (OwaspCategory.LLM01,),
            _p(r"\bignore\s+your\s+(safety|security)\s+(guideline|rule|policy|policies)s?\b"),
            "An instruction tries to make the model ignore its safety guidelines.",
        ),
        Rule(
            "policy_bypass.conceal",
            "policy_bypass",
            Severity.HIGH,
            (OwaspCategory.LLM01,),
            _p(
                r"\bdo\s+not\s+(mention|tell|reveal|disclose|inform)\b[^\n]{0,60}"
                r"\b(user|human|operator|anyone|anybody)\b"
            ),
            "An instruction tries to make the model hide an action from the user.",
        ),
        Rule(
            "policy_bypass.es",
            "policy_bypass",
            Severity.HIGH,
            (OwaspCategory.LLM01, OwaspCategory.LLM03),
            _p(
                r"\b(ignora|omite|desactiva|elude)\s+"
                r"(cualquier|toda|las?|tus?|los?)?\s*"
                r"(pol\u00edtica|pol\u00edticas|medidas?\s+de\s+seguridad|"
                r"restricciones?|filtros?|protecciones?|controles?)"
            ),
            "Una instrucci\u00f3n intenta eludir una pol\u00edtica o medida de seguridad.",
        ),
        # --- Exfiltration instruction (LLM02, LLM06) ----------------------------------------
        Rule(
            "exfil_instruction.en",
            "exfiltration_instruction",
            Severity.HIGH,
            (OwaspCategory.LLM02,),
            _p(
                r"\b(print|show|read|cat|dump|send|upload|post|leak|exfiltrate|reveal|email)\b"
                r"[^\n]{0,60}\b"
                r"(\.env\b|id_rsa|\.ssh|credential|secret|api[\s_-]?key|password|passwd|"
                r"/etc/passwd|environment\s+variable|env\s+var)"
            ),
            "An instruction tries to read or send a secret, credential or environment file.",
        ),
        Rule(
            "exfil_instruction.es",
            "exfiltration_instruction",
            Severity.HIGH,
            (OwaspCategory.LLM02,),
            _p(
                r"\b(imprime|muestra|lee|env\u00eda|sube|exfiltra|revela|manda)\b"
                r"[^\n]{0,60}\b"
                r"(\.env\b|credencial|secreto|clave|claves|contrase\u00f1a|"
                r"variables?\s+de\s+entorno)"
            ),
            "Una instrucci\u00f3n intenta leer o enviar un secreto o credencial.",
        ),
        Rule(
            "secret_request.en",
            "secret_request",
            Severity.MEDIUM,
            (OwaspCategory.LLM02,),
            _p(
                r"\b(what|give|tell|show|reveal)\b[^\n]{0,40}\b(api[\s_-]?key|token|password|secret)\b"
            ),
            "Content asks for a credential or secret value.",
        ),
    ]


#: Characters that open or close a quotation, including the curly and guillemet forms common in
#: Spanish and French prose.
_QUOTE_CHARS = frozenset("'\"`\u00ab\u00bb\u201c\u201d\u2018\u2019")

#: Vocabulary that indicates the surrounding text is *discussing* an attack rather than issuing
#: one. Deliberately broad: a false "discussed" downgrade costs a missed rule hit, a missed one
#: costs a blocked security researcher.
_DISCUSSION_MARKERS: tuple[str, ...] = (
    "attack",
    "injection",
    "malicious",
    "example",
    "for instance",
    "e.g.",
    "such as",
    "explain",
    "document",
    "research",
    "describ",
    "quot",
    "vulnerab",
    "defender",
    "detection",
    "detector",
    "pattern",
    "classic",
    "phrase",
    "payload",
    "technique",
    "adversar",
    "in the wild",
    "we show",
    "we can see",
    "in this",
)


def _context(text: str, start: int, end: int) -> tuple[bool, bool]:
    """Return ``(quoted, discussed)`` for a match at ``[start, end)``."""
    left = text[start - 1] if start > 0 else ""
    right = text[end] if end < len(text) else ""
    quoted = left in _QUOTE_CHARS or right in _QUOTE_CHARS
    window = text[max(0, start - 200) : end + 200].lower()
    discussed = any(marker in window for marker in _DISCUSSION_MARKERS)
    return quoted, discussed


def _contextual_severity(rule: Rule, quoted: bool, discussed: bool, trust: TrustLevel) -> Severity:
    """Downgrade quoted/discussed matches; raise directives that appear in untrusted data."""
    operator = trust is TrustLevel.OPERATOR
    if quoted and discussed:
        return cap_severity(rule.severity, Severity.LOW)
    if quoted:
        return cap_severity(rule.severity, Severity.MEDIUM)
    if discussed and operator:
        return cap_severity(rule.severity, Severity.LOW)
    if discussed:
        return cap_severity(rule.severity, Severity.MEDIUM)
    if not operator and severity_rank(rule.severity) >= severity_rank(Severity.HIGH):
        return raise_severity(rule.severity)
    return rule.severity


class RuleScanner:
    """English + Spanish instruction rules with a quotation/discussion context guard."""

    name = "rules"
    version = VERSION

    def __init__(self, rules: Sequence[Rule] | None = None) -> None:
        self._rules: tuple[Rule, ...] = (
            tuple(rules) if rules is not None else tuple(default_rules())
        )

    @property
    def rules(self) -> tuple[Rule, ...]:
        return self._rules

    def scan(self, text: str, provenance: Provenance) -> Sequence[Finding]:
        trust = trust_of(provenance)
        best: dict[str, Finding] = {}
        for rule in self._rules:
            match = rule.pattern.search(text)
            if match is None:
                continue
            quoted, discussed = _context(text, match.start(), match.end())
            severity = _contextual_severity(rule, quoted, discussed, trust)
            candidate = build_finding(
                scanner=self.name,
                scanner_version=self.version,
                kind=rule.kind,
                severity=severity,
                message=rule.message,
                owasp=rule.owasp,
                span=(match.start(), match.end()),
                attributes={
                    "rule": rule.name,
                    "context": _context_label(quoted, discussed),
                },
            )
            current = best.get(rule.kind)
            if current is None or severity_rank(severity) > severity_rank(current.severity):
                best[rule.kind] = candidate
        return _ordered(best.values())


def _context_label(quoted: bool, discussed: bool) -> str:
    if quoted and discussed:
        return "quoted_discussion"
    if quoted:
        return "quoted"
    if discussed:
        return "discussion"
    return "directive"


def _ordered(findings: Iterable[Finding]) -> list[Finding]:
    return sorted(
        findings,
        key=lambda f: (-severity_rank(f.severity), f.kind, f.span or (0, 0)),
    )
