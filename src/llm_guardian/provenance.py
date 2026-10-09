"""Where a piece of text came from, and how much it may be trusted.

The gateway assigns provenance at a server-controlled boundary. A client-supplied role label
(``"system"``, ``"user"``, ``"assistant"`` from an OpenAI-shaped payload) is *not* provenance and
must never be accepted as one: an attacker controls the role field of their own message.

Trust is a property of the boundary, not of the text, and every downstream consumer takes it as
an argument rather than reading it off the content.
"""

from __future__ import annotations

from enum import StrEnum


class Provenance(StrEnum):
    """Server-assigned origin of a text segment."""

    USER = "user"
    REPOSITORY_FILE = "repository_file"
    RETRIEVED_CHUNK = "retrieved_chunk"
    TOOL_RESULT = "tool_result"
    TOOL_CALL = "tool_call"
    MODEL_OUTPUT = "model_output"


class TrustLevel(StrEnum):
    """What a provenance is allowed to do when it carries instructions."""

    #: A human at the console. Instructions are expected; injection controls still apply.
    OPERATOR = "operator"
    #: Content fetched from a repository, a document store or a tool. Data, never instructions.
    UNTRUSTED_DATA = "untrusted_data"
    #: Text the model produced. Data on the way back out, subject to egress inspection.
    MODEL_DATA = "model_data"


#: The trust level each provenance carries. Instructions found in anything but ``OPERATOR`` text
#: are a signal, not an instruction to follow.
TRUST_BY_PROVENANCE: dict[Provenance, TrustLevel] = {
    Provenance.USER: TrustLevel.OPERATOR,
    Provenance.REPOSITORY_FILE: TrustLevel.UNTRUSTED_DATA,
    Provenance.RETRIEVED_CHUNK: TrustLevel.UNTRUSTED_DATA,
    Provenance.TOOL_RESULT: TrustLevel.UNTRUSTED_DATA,
    Provenance.TOOL_CALL: TrustLevel.UNTRUSTED_DATA,
    Provenance.MODEL_OUTPUT: TrustLevel.MODEL_DATA,
}

#: Provenance values that mark a *high-impact* action boundary: a mistake here reaches a tool,
#: a filesystem or the network, so a scanner error or an unreadable segment fails closed.
HIGH_IMPACT_PROVENANCE: frozenset[Provenance] = frozenset(
    {Provenance.TOOL_CALL, Provenance.TOOL_RESULT}
)


def trust_of(provenance: Provenance) -> TrustLevel:
    """Trust level of a provenance, failing closed on anything unregistered."""
    try:
        return TRUST_BY_PROVENANCE[provenance]
    except KeyError:  # pragma: no cover - defensive: an enum member without a trust level
        return TrustLevel.UNTRUSTED_DATA


def carries_instructions(provenance: Provenance) -> bool:
    """Whether instructions in this segment are legitimate rather than a signal."""
    return trust_of(provenance) is TrustLevel.OPERATOR


def is_high_impact(provenance: Provenance) -> bool:
    """Whether this boundary must fail closed when inspection cannot complete."""
    return provenance in HIGH_IMPACT_PROVENANCE
