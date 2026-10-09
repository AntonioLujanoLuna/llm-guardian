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

VERSION = "rules-4"

#: Evidence that a tool call *matters*: without one of these, "call the get_weather tool with the
#: city" is an API example rather than an injection. Paired with the tool-call shapes below, so a
#: README documenting a tool keeps working.
_RISK_MARKER = (
    r"(?:/etc/(?:passwd|shadow)|\.ssh|authorized_keys|id_rsa|/var/log|rm\s+-rf|"
    r"curl\s|wget\s|https?://|tenants?\b|otro\s+cliente|external|externo|secrets?\b|"
    r"credenciales?|credentials?|DELETE\s+FROM|UPDATE\s+\w+\s+SET|SELECT\s+\*)"
)


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
    """Compile a rule pattern, folding Spanish accents so an ASCII payload still matches.

    A hex- or base64-encoded Spanish instruction arrives without accents ("ignora la politica de
    seguridad"), and a pattern written with an accent then misses the very payload the decoder just
    recovered. Each accented letter becomes a class holding both spellings, so the class is written
    once here instead of by hand in every rule. Two details matter: the rules spell accents as
    ``\\u00ed`` escapes (a raw string keeps the escape for ``re`` to interpret), so the escape form
    is folded as well as the literal character; and a pattern that already spells the class itself
    is left untouched.
    """
    for index, (escape, accented, plain) in enumerate(
        (
            ("\\u00e1", "\u00e1", "a"),
            ("\\u00e9", "\u00e9", "e"),
            ("\\u00ed", "\u00ed", "i"),
            ("\\u00f3", "\u00f3", "o"),
            ("\\u00fa", "\u00fa", "u"),
            ("\\u00f1", "\u00f1", "n"),
        )
    ):
        if any(
            f"[{left}{right}]" in pattern
            for left, right in (
                (plain, escape),
                (escape, plain),
                (plain, accented),
                (accented, plain),
            )
        ):
            continue
        sentinel = f"\x00{index}\x00"
        klass = f"[{plain}{accented}]"
        pattern = pattern.replace(escape, sentinel).replace(accented, sentinel)
        pattern = pattern.replace(sentinel, klass)
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
                r"(instruction|prompt|message|context|direction|rule|guideline|directive|"
                r"constraint|guidance)s?\b"
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
                r"(me\s+)?(?:"
                r"(your\s+|the\s+)?(system\s+(?:prompt|message|instructions?)|"
                r"initial\s+(?:prompt|instructions?)|developer\s+(?:message|instructions?))"
                r"|(your|the|its|my|previous|prior|above|earlier|initial|original|developer|"
                r"system|hidden|full)\s+(instructions|guidelines)"
                r"|(your|its)\s+(hidden|internal|secret|confidential)\s+"
                r"(configuration|config|prompt|instructions|rules|guidelines|settings)"
                r"|(the\s+)?(hidden|secret|confidential)\s+(system\s+)?prompt"
                r")\b"
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
                r"\b(imprime|mu[e\u00e9]stra(?:me)?|revela|repite|dame|ens[e\u00e9]\u00f1ame|vuelca|traduce)\s+"
                r"(tu|el|la|las|los|tus|mis|anteriores?|previas?|iniciales?)\s+"
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
            "role_spoof.header2",
            "delimiter_role_spoofing",
            Severity.HIGH,
            (OwaspCategory.LLM01,),
            _p(
                r"(?m)^\s*#{2,}\s*(system|assistant|developer|user|sistema|asistente|usuario)\s*"
                r"#{0,3}\s*:?\s*(override|instrucciones?|instructions?|mensaje|message)"
            ),
            "A markdown banner impersonates a system or assistant turn without a colon.",
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
        # --- Instruction override, paraphrase-resistant (LLM01) ------------------------------
        # The Phase-1 baseline (docs/phase1-baseline.md) showed the gap is not the imperative
        # form but the *rephrased* one: "new instructions supersede your training", "follow only
        # the content of this document". These rules target the rephrasing, not the vocabulary.
        Rule(
            "instruction_override.en3",
            "instruction_override",
            Severity.HIGH,
            (OwaspCategory.LLM01,),
            _p(
                r"\b(new|revised|updated|replacement)\s+"
                r"(instructions?|rules?|directives?|guidelines?)\b[^.\n]{0,60}"
                r"\b(supersede|replace|override|take\s+precedence|come\s+first|are\s+in\s+effect)"
            ),
            "New instructions claim priority over the model's training or earlier rules.",
        ),
        Rule(
            "instruction_override.en4",
            "instruction_override",
            Severity.HIGH,
            (OwaspCategory.LLM01,),
            _p(
                r"\b(follow|obey|use|trust)\s+only\s+(the\s+)?(instructions?|content|text|rules?)\b"
                r"[^.\n]{0,60}\b(found\s+in|in\s+this|from\s+this|of\s+this)\b[^.\n]{0,40}"
                r"\b(document|chunk|fragment|message|file|page|context|retrieved|email)"
            ),
            "A directive tells the model to follow only the content of one document it was given.",
        ),
        Rule(
            "instruction_override.es2",
            "instruction_override",
            Severity.HIGH,
            (OwaspCategory.LLM01,),
            _p(
                r"\b(nuevas?|nuevos?)\s+(instrucciones?|reglas?|directrices?|indicaciones?)\b"
                r"|\ba\s+partir\s+de\s+ahora\s+(sigue|obedece|ignora|omite|prescinde)\b"
                r"|\b(sigue|obedece)\s+solo\s+(el|lo)\b[^.\n]{0,30}"
                r"\b(contenido|texto|fragmento|documento|mensaje|correo)"
                r"|\bprescinde\s+de\s+las?\s+(instrucciones?|indicaciones?)"
                r"|\bpor\s+encima\s+de\s+cualquier\s+otra\s+cosa\b"
                r"|\btu\s+nueva\s+directriz\b"
            ),
            "Una directiva en espa\u00f1ol reclama prioridad sobre las instrucciones anteriores.",
        ),
        Rule(
            "instruction_override.void_en",
            "instruction_override",
            Severity.HIGH,
            (OwaspCategory.LLM01,),
            _p(
                r"\b(previous|prior|earlier|above)\s+(instructions?|rules?|directives?)\s+"
                r"(are|is|were|was)\s+(void|null|invalid|obsolete|cancelled|canceled|revoked)"
            ),
            "An instruction declares the earlier instructions void.",
        ),
        Rule(
            "instruction_override.void_es",
            "instruction_override",
            Severity.HIGH,
            (OwaspCategory.LLM01,),
            _p(
                r"\b(instrucciones?|reglas?)\s+(anteriores?|previas?)\s+"
                r"(quedan?|est[a\u00e1]n?)\s+(anulad|inv[a\u00e1]lid|sin\s+efecto|obsolet)"
            ),
            "Una instrucci\u00f3n declara anuladas las instrucciones anteriores.",
        ),
        # --- System-prompt exfiltration, paraphrase-resistant (LLM01, LLM08) -----------------
        Rule(
            "system_prompt_exfil.en3",
            "system_prompt_exfiltration",
            Severity.HIGH,
            (OwaspCategory.LLM01, OwaspCategory.LLM08),
            _p(
                r"\b(echo|return|repeat|paste|dump|print|show|include|prepend|append)\b"
                r"[^.\n]{0,40}\b(your|the)\s+(full\s+|complete\s+|raw\s+|original\s+)?"
                r"(configuration|config|system\s+message|developer\s+message|"
                r"initial\s+(message|instructions?|prompt))\b"
                r"|\b(first|last)\s+\d{2,}\s+characters\s+of\s+(your|the)\s+"
                r"(system|developer|initial|original)"
            ),
            "A request asks for the model's configuration or system message in another form.",
        ),
        Rule(
            "system_prompt_exfil.es2",
            "system_prompt_exfiltration",
            Severity.HIGH,
            (OwaspCategory.LLM01, OwaspCategory.LLM08),
            _p(
                r"\brepite\s+(todo\s+)?(el\s+texto|todo\s+lo)\b[^.\n]{0,40}"
                r"\b(por\s+encima|anterior|arriba)\b"
                r"|\bconfiguraci[o\u00f3]n\s+inicial\b"
                r"|\binstrucciones?\s+ocultas?\b"
            ),
            "Una petici\u00f3n pide repetir el contexto anterior o la configuraci\u00f3n inicial.",
        ),
        # --- Policy bypass, paraphrase-resistant (LLM01, LLM03) ------------------------------
        Rule(
            "policy_bypass.en3",
            "policy_bypass",
            Severity.HIGH,
            (OwaspCategory.LLM01,),
            _p(
                r"\b(disregard|ignore|skip|bypass)\s+(the\s+)?"
                r"(safety|security|policy|policies|checks?|controls?|review|guardrails?)\b"
                r"|\b(continue|proceed|act|answer|respond|carry\s+on)\b[^.\n]{0,40}"
                r"\bwithout\s+(asking|confirming|confirmation|approval|consent|permission)\b"
                r"|\bwithout\s+asking\s+again\b"
            ),
            "An instruction asks the model to proceed past its safety checks or a confirmation.",
        ),
        Rule(
            "policy_bypass.es2",
            "policy_bypass",
            Severity.HIGH,
            (OwaspCategory.LLM01,),
            _p(
                r"\bsin\s+(aplicar\s+filtros?|filtrar|consultar|confirmar|"
                r"pedir\s+(permiso|confirmaci[o\u00f3]n|aprobaci[o\u00f3]n))\b"
                r"|\bsin\s+(aplicar|pasar)\s+por\s+(el\s+)?(filtro|control|revisi[o\u00f3]n)\b"
                r"|\b(omitir|ignorar|desactivar|eludir)[e\u00e9]\b[^.\n]{0,40}"
                r"\b(pol[i\u00ed]tica|pol[i\u00ed]ticas|seguridad|restricciones?|filtros?|controles?)\b"
            ),
            "Una instrucci\u00f3n pide actuar sin filtros, consulta ni confirmaci\u00f3n.",
        ),
        Rule(
            "policy_bypass.conceal_es",
            "policy_bypass",
            Severity.HIGH,
            (OwaspCategory.LLM01,),
            _p(
                r"\bno\s+(menciones?|digas?|reveles?|informes?|cuentes?)\b[^.\n]{0,60}"
                r"\b(usuario|humano|persona|nadie)\b"
            ),
            "Una instrucci\u00f3n pide ocultar una acci\u00f3n al usuario.",
        ),
        # --- Tool-call injection, structural (LLM03, LLM01) ---------------------------------
        # The baseline's misses here were shape, not vocabulary: a tool result carrying a
        # tool-call object, or prose directing the assistant to a named tool. Both need a risk
        # marker as well, so an API example in a README stays inert.
        Rule(
            "tool_call.imperative_en",
            "tool_call_instruction",
            Severity.HIGH,
            (OwaspCategory.LLM03, OwaspCategory.LLM01),
            _p(
                r"\b(call|invoke|use|execute|run)\b[^\n]{0,40}\b(tool|function)\b"
                r"[^\n]{0,140}?" + _RISK_MARKER
            ),
            "Content directs the model to call a tool, and the call touches something sensitive.",
        ),
        Rule(
            "tool_call.imperative_es",
            "tool_call_instruction",
            Severity.HIGH,
            (OwaspCategory.LLM03, OwaspCategory.LLM01),
            _p(
                r"\b(llama|usa|utiliza|invoca|ejecuta)\b[^\n]{0,40}"
                r"\b(herramienta|funci[o\u00f3]n|tool)\b[^\n]{0,140}?" + _RISK_MARKER
            ),
            "Una instrucci\u00f3n pide llamar a una herramienta que toca algo sensible.",
        ),
        Rule(
            "tool_call.json_shape",
            "tool_call_instruction",
            Severity.HIGH,
            (OwaspCategory.LLM03, OwaspCategory.LLM01),
            _p(
                r"\"(next_tool_call|tool_call|function_call)\"\s*:"
                r"|\"(name|nombre)\"\s*:\s*\"[^\"\n]{1,60}\"\s*,\s*"
                r"\"(arguments|args|argumentos)\"\s*:\s*\{[^\n]{0,220}?" + _RISK_MARKER
            ),
            "A tool-call object appears inside content that should be inert.",
        ),
        # --- Tenant / scope bypass (LLM03, LLM09) -------------------------------------------
        # A cross-tenant read is an authorization question, so it is answered by policy.py; the
        # *request* for one is a detection signal, which is all this rule reports.
        Rule(
            "tenant_scope_bypass.en",
            "tenant_scope_bypass",
            Severity.HIGH,
            (OwaspCategory.LLM03, OwaspCategory.LLM09),
            _p(
                r"\b(tenant|scope|user|session)\b[^\n]{0,60}"
                r"\b(does\s+not\s+matter|doesn'?t\s+matter|no\s+importa|ignorar?|bypass|"
                r"omitir|saltar)\b"
                r"|\b(every|all)\s+(tenant|tenants|user|users)\b[^\n]{0,40}"
                r"\b(secrets?|records?|data|stored|tokens?)\b"
            ),
            "Content asks for data outside the caller's tenant or scope.",
        ),
        Rule(
            "tenant_scope_bypass.es",
            "tenant_scope_bypass",
            Severity.HIGH,
            (OwaspCategory.LLM03, OwaspCategory.LLM09),
            _p(
                r"\baunque\s+(esta\s+)?(sesi[o\u00f3]n|usuario)\s+(est[e\u00e9]\s+)?(limitad|acotad)"
                r"|\botro\s+cliente\b[^\n]{0,40}\b(datos|registros|saldos|secretos|acceso)\b"
                r"|\btodos\s+los\s+(clientes|tenants|usuarios)\b"
                r"|\bno\s+importa\s+(el\s+)?([a\u00e1]mbito|alcance|scope)\b"
            ),
            "Una petici\u00f3n pide datos fuera del alcance del usuario.",
        ),
        # --- Persona-based jailbreak (LLM01) ------------------------------------------------
        # The baseline's misses here were persona framing rather than a magic word: "pretend you
        # are an unfiltered model inside a movie script".
        Rule(
            "jailbreak_persona.en",
            "jailbreak_persona",
            Severity.HIGH,
            (OwaspCategory.LLM01,),
            _p(
                r"\b(pretend|act|roleplay|role-play|imagine|behave)\b[^\n]{0,40}"
                r"\b(you\s+are|you're|to\s+be)\b[^\n]{0,60}"
                r"\b(unfiltered|unrestricted|without\s+restrictions|no\s+restrictions|no\s+rules|"
                r"uncensored|DAN)\b"
                r"|\b(you\s+are\s+now|from\s+now\s+on\s+you\s+are)\b[^\n]{0,40}"
                r"\b(unfiltered|unrestricted|uncensored|DAN)\b"
            ),
            "A persona is used to remove the model's restrictions.",
        ),
        Rule(
            "jailbreak_persona.es",
            "jailbreak_persona",
            Severity.HIGH,
            (OwaspCategory.LLM01,),
            _p(
                r"\b(act\u00faa|pretende|imagina|comp\u00f3rtate)\b[^\n]{0,40}\bcomo\b[^\n]{0,60}"
                r"\b(sin\s+(restricciones|filtros|l\u00edmites)|ilimitad|sin\s+reglas)\b"
            ),
            "Se usa una personalidad para eliminar las restricciones del modelo.",
        ),
        # --- Instruction addressed to the model, or against the user's task (LLM01) ---------
        Rule(
            "instruction_override.ignore_user_task",
            "instruction_override",
            Severity.HIGH,
            (OwaspCategory.LLM01,),
            _p(
                r"\bignore\s+(the\s+)?(user|operator|developer|human)(?:'s|s')?\s+"
                r"(task|request|question|prompt|message|instructions?|goal)"
            ),
            "An instruction tells the model to ignore the task the user actually asked for.",
        ),
        Rule(
            "instruction_addressed_to_model",
            "instruction_override",
            Severity.MEDIUM,
            (OwaspCategory.LLM01,),
            _p(
                r"\b(hidden|important|urgent|note|attention|nota|instrucci\u00f3n(?:es)?)\b"
                r"[^\n]{0,40}\b(for|to|al|para)\b[^\n]{0,20}"
                r"\b(ai|llm|agents?|assistants?|models?|tools?|agente|asistentes?)\b"
                r"|\bnote\s+for\s+the\s+(assistant|agent|model|ai)\b"
            ),
            "Content is addressed to the model as an instruction rather than to a reader.",
        ),
        Rule(
            "instruction_addressed_with_target",
            "instruction_override",
            Severity.HIGH,
            (OwaspCategory.LLM01, OwaspCategory.LLM02),
            _p(
                r"\b(hidden|important|urgent|secret|nota|instrucci\u00f3n(?:es)?)\b"
                r"[^\n]{0,40}\b(for|to|al|para)\b[^\n]{0,20}"
                r"\b(ai|llm|agents?|assistants?|models?|tools?|agente)\b[^\n]{0,140}"
                r"\b(ssh\s+key|private\s+key|clave\s+privada|secrets?|secretos?|\.ssh|id_rsa|"
                r"api[\s_-]?key|\.env|\.config|credentials?|credenciales?)"
            ),
            "An instruction addressed to the model also names a secret or a private key.",
        ),
        Rule(
            "jailbreak_developer_mode.en",
            "jailbreak_persona",
            Severity.HIGH,
            (OwaspCategory.LLM01,),
            _p(
                r"\b(enable|activate|enter|switch\s+to)\b[^\n]{0,20}\b(developer|debug|god|"
                r"administrator|unrestricted)\s+mode\b"
                r"|\byou\s+have\s+no\s+content\s+polic(?:y|ies)\b"
                r"|\byou\s+will\s+not\s+refuse\b"
            ),
            "A mode switch or a no-refusal claim is used to strip the model's policy.",
        ),
        Rule(
            "jailbreak_developer_mode.es",
            "jailbreak_persona",
            Severity.HIGH,
            (OwaspCategory.LLM01,),
            _p(
                r"\b(activa|habilita|entra\s+en)\b[^\n]{0,20}\bmodo\s+"
                r"(desarrollador|dios|debug|sin\s+restricciones)\b"
                r"|\bno\s+tienes\s+pol\u00edtica\s+de\s+contenido\b"
            ),
            "Se activa un modo o se niega la pol\u00edtica del modelo para salt\u00e1rsela.",
        ),
        # --- Jailbreak precursors (LLM01): weak on their own, by design ----------------------
        # Phrasings that set up a jailbreak without yet asking for anything. One of them alone
        # is low severity and is allowed; the per-conversation tracker (session.py) is what
        # notices a run of them building toward an attack across turns.
        Rule(
            "jailbreak_precursor.en",
            "jailbreak_precursor",
            Severity.LOW,
            (OwaspCategory.LLM01,),
            _p(
                r"\byou\s+are\s+(now\s+)?in\s+(developer|debug|god|jailbreak|unrestricted)\s+mode\b"
                r"|\b(pretend|imagine|act\s+as\s+if)\b[^\n]{0,20}\byou\s+(have|had)\s+no\s+"
                r"(rules|restrictions|limits|guidelines|filters)\b"
                r"|\bact\s+as\s+(an?\s+)?(unrestricted|unfiltered|uncensored)\b"
                r"|\bhypothetical(ly)?\b[^\n]{0,60}\b(your|the)\s+(hidden|secret|system)\s+"
                r"(instructions|prompt|rules)\b"
                r"|\bstay\s+in\s+character\b[^\n]{0,40}\b(no\s+matter|whatever|regardless)\b"
            ),
            "A phrasing that sets up a jailbreak (a mode, a rule-free persona, a hypothetical).",
        ),
        Rule(
            "jailbreak_precursor.es",
            "jailbreak_precursor",
            Severity.LOW,
            (OwaspCategory.LLM01,),
            _p(
                r"\best\u00e1s\s+(ahora\s+)?en\s+modo\s+(desarrollador|dios|sin\s+restricciones)\b"
                r"|\b(imagina|finge)\b[^\n]{0,20}\bque\s+no\s+tienes\s+"
                r"(reglas|restricciones|l\u00edmites|filtros)\b"
            ),
            "Una frase que prepara un jailbreak (un modo, una personalidad sin reglas).",
        ),
        # --- Credential phishing (LLM02, LLM03) ---------------------------------------------
        Rule(
            "credential_phishing.en",
            "credential_phishing",
            Severity.HIGH,
            (OwaspCategory.LLM02, OwaspCategory.LLM03),
            _p(
                r"\b(page|portal|site|form|email|login)\b[^\n]{0,60}"
                r"\b(identical|clone|cloned|fake|look-alike|imitation)\b[^\n]{0,120}"
                r"\b(passwords?|credentials?|login|employees?|bank|corporate)"
            ),
            "Content asks for a fake sign-in page or a credential-harvesting message.",
        ),
        Rule(
            "credential_phishing.es",
            "credential_phishing",
            Severity.HIGH,
            (OwaspCategory.LLM02, OwaspCategory.LLM03),
            _p(
                r"\b(p\u00e1gina|portal|sitio|web|formulario|correo)\b[^\n]{0,60}"
                r"\b(id\u00e9ntic\w*|clon\w*|falsa?|falso|imita\w*|parecid\w*)\b[^\n]{0,120}"
                r"\b(contrase\u00f1as?|credenciales?|acceso|portal|emplead|personal|"
                r"banco|corporativ)"
            ),
            "Se pide una p\u00e1gina de acceso falsa o un correo que recoja credenciales.",
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
    "research",
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
)

#: The markers as prefix alternation: ``\battack`` matches "attack", "attacks" and "attackers",
#: which is how prose writes it, while the identifier strip below keeps it off hyphenated values
#: (``external-attacker``). ``document`` and ``in this`` were dropped outright: they appear in
#: ordinary directives ("follow only the instructions in this document") far more than in
#: discussion.
_DISCUSSION_RE = re.compile("|".join(rf"\b{re.escape(marker)}" for marker in _DISCUSSION_MARKERS))


#: A token that is part of a URL or a hostname rather than prose. ``attacker.example`` must not
#: read as the discussion marker "example", which is exactly how a tool-call object inside a tool
#: result came back as "under discussion" instead of as a directive.
_HOSTLIKE = re.compile(r"\b(?:https?://)?[A-Za-z0-9][A-Za-z0-9.-]*\.[A-Za-z]{2,}(?::\d+)?(?:/\S*)?")

#: Identifier-shaped tokens (``external-attacker``, ``otro-cliente``, ``KEY=VALUE``). The marker
#: "attack" belongs to a sentence *about* attacks, not to a hyphenated value in an argument.
_IDENTIFIER = re.compile(r"[\w.+-]*[-=_][\w.+-]*")

#: JSON punctuation, blanked out of the discussion window so that quoting syntax cannot supply a
#: marker and prose markers stay visible.
_JSON_PUNCT = str.maketrans({char: " " for char in '{}[]":,'})


def _structural_json(text: str, start: int) -> bool:
    """Whether the quotes beside a match are JSON syntax rather than quotation marks.

    A segment that opens as a JSON document quotes its own keys and values. Treating those quotes
    as "the author is quoting an attack" discounted exactly the tool-call objects this scanner
    exists to catch, so the quotation discount does not apply inside a structure; the discussion
    test still does, which is what keeps a quoted example in a security explainer discounted.
    """
    if text.lstrip()[:1] in ("{", "["):
        return True
    return text[:start].rstrip().endswith(("{", "[", ",", ":"))


def _context(text: str, start: int, end: int) -> tuple[bool, bool]:
    """Return ``(quoted, discussed)`` for a match at ``[start, end)``."""
    left = text[start - 1] if start > 0 else ""
    right = text[end] if end < len(text) else ""
    quoted = (left in _QUOTE_CHARS or right in _QUOTE_CHARS) and not _structural_json(text, start)
    window = text[max(0, start - 200) : end + 200]
    prose = _IDENTIFIER.sub(" ", _HOSTLIKE.sub(" ", window)).translate(_JSON_PUNCT).lower()
    discussed = _DISCUSSION_RE.search(prose) is not None
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
