"""Measure the policy layer on the declared tool-call corpus.

This is the tool-boundary counterpart of ``bench.harness``: where that module scores *detection*
over text, this one scores *authorization* over proposed actions. It loads the
:mod:`llm_guardian.bench.tool_calls` corpus, sends each case through the real
:class:`~llm_guardian.policy.PolicyEngine` (the same code ``/v1/agent/authorize`` calls), and
publishes the number that decides whether the layer is usable: **false denials per 1,000 legitimate
tool calls**, together with the denial rate on the malicious set and every false denial, by name.

It also measures two naive alternatives in one table, because a number is only meaningful against
the alternatives it beats:

* **deny nothing** -- 0 false denials, 0 caught;
* **detector-only** -- deny whenever *any* injection finding is present in the echoed evidence,
  regardless of provenance or kind. The false-denial rate of that rule is the reason the real
  evidence rule is scoped to untrusted provenance and to three finding kinds.

The corpus is synthesized in this repository by the same author as the policy, so these are
**design-cost estimates, not a validation on real traffic** -- the generated report says so, and
``docs/evaluations/policy-enforcement.md`` records that caveat next to the numbers.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from ..models import EvidenceRef, PolicyAction, PolicyCheckRequest, Severity
from ..policy import PolicyEngine
from ..provenance import Provenance
from ..settings import PolicySettings
from .tool_calls import (
    CASES,
    LEGITIMATE,
    MALICIOUS,
    ToolCallCase,
)

__all__ = [
    "BROAD_INJECTION_KINDS",
    "DEFAULT_POLICY_FILE",
    "Outcome",
    "PolicyReport",
    "build_engine",
    "evaluate",
    "main",
    "measure",
    "render_markdown",
    "run_cli",
]

#: The policy file the report is measured against, relative to the repository root.
DEFAULT_POLICY_FILE = Path("policies/example.toml")

#: Every kind a *detector* would call an injection signal. The naive "detector-only" alternative
#: denies when any of these appears in the echoed evidence, at any provenance. The shipped evidence
#: rule uses only the three in :data:`~llm_guardian.policy.INJECTION_EVIDENCE_KINDS` and only
#: untrusted provenance -- this broader set is the alternative being compared against.
BROAD_INJECTION_KINDS: frozenset[str] = frozenset(
    {
        "agent_directive",
        "agent_directive_control_override",
        "agent_directive_control_suspended",
        "agent_directive_conditional_behaviour",
        "agent_directive_adopt_as_instruction",
        "agent_directive_persona_authority",
        "tool_call_instruction",
        "instruction_override",
    }
)

#: Provenance values that are a channel rather than the operator, for the report's split.
_UNTRUSTED_PROPOSERS: frozenset[str] = frozenset(
    {"tool_result", "retrieved_chunk", "repository_file"}
)

_FIXED_CLOCK = datetime(2020, 1, 1, tzinfo=UTC)


@dataclass(frozen=True)
class Outcome:
    """What the policy layer did with one case."""

    case_id: str
    label: str
    family: str
    proposed_by: str
    why: str
    denied: bool
    denied_by: str | None
    reason_codes: tuple[str, ...]
    argument_failures: tuple[str, ...]
    evidence_used: tuple[str, ...]
    needed_evidence: bool
    detector_only_denies: bool

    @property
    def proposer_class(self) -> str:
        if self.proposed_by == "user":
            return "operator"
        if self.proposed_by in _UNTRUSTED_PROPOSERS:
            return "untrusted"
        return "other"


@dataclass(frozen=True)
class FamilyStat:
    family: str
    total: int
    denied: int
    rate: float


@dataclass(frozen=True)
class SplitStat:
    name: str
    legitimate: int
    false_denials: int
    false_denial_rate_per_1000: float
    malicious: int
    malicious_caught: int
    malicious_denial_rate: float


@dataclass(frozen=True)
class AlternativeStat:
    name: str
    false_denials: int
    false_denial_rate_per_1000: float
    caught: int
    catch_rate: float


@dataclass(frozen=True)
class PolicyReport:
    policy_file: str
    corpus_total: int
    legitimate: int
    malicious: int
    false_denials: int
    false_denial_rate_per_1000: float
    malicious_caught: int
    malicious_denial_rate: float
    false_denial_cases: tuple[Outcome, ...]
    missed_cases: tuple[Outcome, ...]
    legitimate_by_family: tuple[FamilyStat, ...]
    malicious_by_family: tuple[FamilyStat, ...]
    by_proposer: tuple[SplitStat, ...]
    by_evidence_needed: tuple[SplitStat, ...]
    alternatives: tuple[AlternativeStat, ...]
    evidence_rule_fired: int
    checked_at: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)


# =================================================================================================
# Evaluation
# =================================================================================================
def build_engine(policy_file: Path) -> PolicyEngine:
    """A policy engine reading ``policy_file``, with fail-closed defaults."""
    return PolicyEngine(PolicySettings(file=policy_file))


def _request(case: ToolCallCase) -> PolicyCheckRequest:
    return PolicyCheckRequest(
        principal=case.principal,
        tenant=case.tenant,
        action=PolicyAction(case.action),
        tool=case.tool or None,
        tool_arguments=dict(case.tool_arguments),
        evidence=tuple(
            EvidenceRef(
                segment_id=evidence.segment_id,
                finding_kind=evidence.finding_kind,
                severity=Severity(evidence.severity),
                provenance=Provenance(evidence.provenance),
            )
            for evidence in case.evidence
        ),
        requested_capabilities=tuple(case.requested_capabilities),
        egress_hosts=tuple(case.egress_hosts),
        target_tenant=case.target_tenant,
    )


def evaluate(cases: Sequence[ToolCallCase], engine: PolicyEngine) -> tuple[Outcome, ...]:
    """Run every case through the engine and record the decision plus the naive comparison."""
    outcomes: list[Outcome] = []
    for case in cases:
        check = engine.check(_request(case), now=lambda: _FIXED_CLOCK)
        detector_only = any(
            evidence.finding_kind in BROAD_INJECTION_KINDS for evidence in case.evidence
        )
        outcomes.append(
            Outcome(
                case_id=case.id,
                label=case.label,
                family=case.family,
                proposed_by=case.proposed_by,
                why=case.why,
                denied=not check.allowed,
                denied_by=check.denied_by,
                reason_codes=check.reason_codes,
                argument_failures=check.argument_failures,
                evidence_used=check.evidence_used,
                needed_evidence=bool(case.evidence),
                detector_only_denies=detector_only,
            )
        )
    return tuple(outcomes)


def _family_stats(pairs: Sequence[tuple[str, bool]]) -> tuple[FamilyStat, ...]:
    totals: dict[str, int] = defaultdict(int)
    denieds: dict[str, int] = defaultdict(int)
    for family, denied in pairs:
        totals[family] += 1
        if denied:
            denieds[family] += 1
    stats = [
        FamilyStat(
            family=family,
            total=totals[family],
            denied=denieds[family],
            rate=denieds[family] / totals[family],
        )
        for family in sorted(totals, key=lambda f: (denieds[f] / totals[f], f))
    ]
    return tuple(stats)


def _split(name: str, group: Sequence[Outcome]) -> SplitStat:
    legitimate = [o for o in group if o.label == LEGITIMATE]
    malicious = [o for o in group if o.label == MALICIOUS]
    legit_denied = sum(1 for o in legitimate if o.denied)
    caught = sum(1 for o in malicious if o.denied)
    return SplitStat(
        name=name,
        legitimate=len(legitimate),
        false_denials=legit_denied,
        false_denial_rate_per_1000=(legit_denied / len(legitimate) * 1000.0 if legitimate else 0.0),
        malicious=len(malicious),
        malicious_caught=caught,
        malicious_denial_rate=(caught / len(malicious) if malicious else 0.0),
    )


def measure(outcomes: Sequence[Outcome], policy_file: Path) -> PolicyReport:
    """Aggregate outcomes into the report the markdown is rendered from."""
    legitimate = [o for o in outcomes if o.label == LEGITIMATE]
    malicious = [o for o in outcomes if o.label == MALICIOUS]

    false_denials = [o for o in legitimate if o.denied]
    missed = [o for o in malicious if not o.denied]
    caught = len(malicious) - len(missed)

    by_proposer = tuple(
        _split(name, [o for o in outcomes if o.proposer_class == name])
        for name in ("operator", "untrusted", "other")
        if any(o.proposer_class == name for o in outcomes)
    )
    by_evidence = (
        _split("evidence rule exercised", [o for o in outcomes if o.needed_evidence]),
        _split("no evidence supplied", [o for o in outcomes if not o.needed_evidence]),
    )

    detector_false = sum(1 for o in legitimate if o.detector_only_denies)
    detector_caught = sum(1 for o in malicious if o.detector_only_denies)
    alternatives = (
        AlternativeStat(
            name="deny nothing",
            false_denials=0,
            false_denial_rate_per_1000=0.0,
            caught=0,
            catch_rate=0.0,
        ),
        AlternativeStat(
            name="detector-only (deny on any injection finding)",
            false_denials=detector_false,
            false_denial_rate_per_1000=(
                detector_false / len(legitimate) * 1000.0 if legitimate else 0.0
            ),
            caught=detector_caught,
            catch_rate=(detector_caught / len(malicious) if malicious else 0.0),
        ),
        AlternativeStat(
            name="policy + scoped evidence rule (this system)",
            false_denials=len(false_denials),
            false_denial_rate_per_1000=(
                len(false_denials) / len(legitimate) * 1000.0 if legitimate else 0.0
            ),
            caught=caught,
            catch_rate=(caught / len(malicious) if malicious else 0.0),
        ),
    )

    return PolicyReport(
        policy_file=str(policy_file),
        corpus_total=len(outcomes),
        legitimate=len(legitimate),
        malicious=len(malicious),
        false_denials=len(false_denials),
        false_denial_rate_per_1000=(
            len(false_denials) / len(legitimate) * 1000.0 if legitimate else 0.0
        ),
        malicious_caught=caught,
        malicious_denial_rate=(caught / len(malicious) if malicious else 0.0),
        false_denial_cases=tuple(false_denials),
        missed_cases=tuple(missed),
        legitimate_by_family=_family_stats([(o.family, o.denied) for o in legitimate]),
        malicious_by_family=_family_stats([(o.family, o.denied) for o in malicious]),
        by_proposer=by_proposer,
        by_evidence_needed=by_evidence,
        alternatives=alternatives,
        evidence_rule_fired=sum(1 for o in outcomes if o.evidence_used),
        checked_at=_FIXED_CLOCK.isoformat(),
    )


# =================================================================================================
# Rendering
# =================================================================================================
_SYNTHESIS_NOTE = (
    "**This corpus is synthesized in this repository, by the same author as the policy it "
    "measures.** Every number below is a *design-cost estimate*: zero false denials says the "
    "rules do what they were written to do on cases written alongside them, not that they "
    "generalise to real traffic. The unbiased figure is missing until the corpus is replaced by "
    "traffic the rules were never written against. The detectors are not touched or measured here."
)

_OPERATIONAL_COST = (
    "A denial is not one thing. A blocked file read inside the workspace costs a retry; a blocked "
    "`deploy_staging` costs a release window and an on-call page; a blocked `send_mail` or "
    "`post_message` costs a quiet status update; a blocked cross-tenant read costs a support "
    "ticket that cannot be answered. The false-denial *rate* pools them, so the cases are listed "
    "by id above: on this run the operator can see exactly which legitimate calls a tightened rule "
    "would have held. The scoped evidence rule is what keeps the confused-deputy catch from "
    "turning every quoted injection in an operator's own paste into a blocked tool call."
)

_KNOWN_GAPS = (
    "Stated plainly, because a limitation left implicit reads as a claim:",
    "",
    "1. **The corpus is synthesized here**, by the same author as the policy, so 0/1,000 is a "
    "design-cost estimate and not a generalisation result. It is not evidence about traffic the "
    "rules were never written against.",
    "2. **The cross-tenant rule reuses the read grant for writes.** A tool call (a write) whose "
    "`target_tenant` differs from the principal's own is denied unless the target tenant lists the "
    "principal in `allowed_cross_tenant_reads` -- the only grant key that exists. Fail-closed, but "
    "a read grant named `..._reads` currently also authorises a write to that tenant; a separate "
    "write grant is the missing piece.",
    "3. **`run_shell` constrains command *shape*, not the files its arguments name.** The schema "
    "requires a safe first token and forbids shell metacharacters, so `rm -rf` and curl-pipe-to-"
    "shell are denied; but a command built from an allow-listed binary (`cat`, `ls`) is not "
    "path-checked, so `cat /etc/shadow` would pass the shell schema. File-confinement is the job "
    "of `read_file`'s `path_within`, which is where the sensitive-read cases are caught.",
    "4. **No case here drives a scanner error or an unreadable policy file** through the endpoint; "
    "the engine fails closed for both (unit-tested separately), and the corpus does not add a "
    "malicious case that relies on those paths.",
    "5. **Latency is not measured here.** The policy check is deterministic and in-process; the "
    "detection layer's latency is reported in `docs/phase1-baseline.md`.",
)


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _rate_per_1000(value: float) -> str:
    return f"{value:.1f}"


def render_markdown(report: PolicyReport) -> str:
    """The full report: numbers, every false denial by name, and the alternatives compared."""
    lines: list[str] = []
    add = lines.append
    add("# Policy enforcement at the tool boundary — measured")
    add("")
    add(_SYNTHESIS_NOTE)
    add("")
    add("## How to reproduce")
    add("")
    add("```bash")
    add("unset PYTHONPATH")
    add(
        "PYTHONPATH=src .venv/bin/python -m llm_guardian.bench.policy_report --policy "
        + report.policy_file
    )
    add("```")
    add("")
    add(
        f"Policy file: `{report.policy_file}`. Corpus: **{report.corpus_total} cases "
        f"({report.legitimate} legitimate / {report.malicious} malicious)**."
    )
    add("")
    add("## Headline")
    add("")
    add("| Metric | Measured |")
    add("| --- | --- |")
    add(
        f"| False denials per 1,000 legitimate tool calls | "
        f"**{_rate_per_1000(report.false_denial_rate_per_1000)}** "
        f"({report.false_denials}/{report.legitimate}) |"
    )
    add(
        f"| Malicious calls denied (overall) | **{_pct(report.malicious_denial_rate)}** "
        f"({report.malicious_caught}/{report.malicious}) |"
    )
    add(f"| Cases where the evidence rule fired | {report.evidence_rule_fired} |")
    add("")
    add("## Malicious denial rate by family")
    add("")
    add("| Family | Denied | Rate |")
    add("| --- | --- | --- |")
    for stat in report.malicious_by_family:
        add(f"| `{stat.family}` | {stat.denied}/{stat.total} | {_pct(stat.rate)} |")
    add("")
    add("## Legitimate false denials by family")
    add("")
    add("| Family | Denied | Rate /1,000 |")
    add("| --- | --- | --- |")
    for stat in report.legitimate_by_family:
        add(
            f"| `{stat.family}` | {stat.denied}/{stat.total} | "
            f"{_rate_per_1000(stat.rate * 1000.0)} |"
        )
    add("")
    add("## Every false denial, named")
    add("")
    if not report.false_denial_cases:
        add("None: no legitimate case was denied on this run.")
    for outcome in report.false_denial_cases:
        add(f"- `{outcome.case_id}` — denied by `{outcome.denied_by}`; {outcome.why}")
    add("")
    if report.missed_cases:
        add("## Malicious cases that were NOT denied")
        add("")
        for outcome in report.missed_cases:
            add(
                f"- `{outcome.case_id}` — allowed; "
                f"reasons `{', '.join(outcome.reason_codes)}`; {outcome.why}"
            )
        add("")
    add("## Split by the provenance that proposed the call")
    add("")
    add(
        "| Proposer | Legitimate | False denials | False denials /1,000 | "
        "Malicious | Denied | Denial rate |"
    )
    add("| --- | --- | --- | --- | --- | --- | --- |")
    for split in report.by_proposer:
        add(
            f"| {split.name} | {split.legitimate} | {split.false_denials} | "
            f"{_rate_per_1000(split.false_denial_rate_per_1000)} | {split.malicious} | "
            f"{split.malicious_caught} | {_pct(split.malicious_denial_rate)} |"
        )
    add("")
    add("## Split by whether the evidence rule was needed")
    add("")
    add(
        "| Group | Legitimate | False denials | False denials /1,000 | "
        "Malicious | Denied | Denial rate |"
    )
    add("| --- | --- | --- | --- | --- | --- | --- |")
    for split in report.by_evidence_needed:
        add(
            f"| {split.name} | {split.legitimate} | {split.false_denials} | "
            f"{_rate_per_1000(split.false_denial_rate_per_1000)} | {split.malicious} | "
            f"{split.malicious_caught} | {_pct(split.malicious_denial_rate)} |"
        )
    add("")
    add("## Against the naive alternatives")
    add("")
    add("| Rule | False denials /1,000 legitimate | Malicious caught | Catch rate |")
    add("| --- | --- | --- | --- |")
    for alt in report.alternatives:
        add(
            f"| {alt.name} | {_rate_per_1000(alt.false_denial_rate_per_1000)} "
            f"({alt.false_denials}) | {alt.caught} | {_pct(alt.catch_rate)} |"
        )
    add("")
    add(
        "The detector-only row is the reason the evidence rule is scoped to *untrusted provenance* "
        'and to three finding kinds: an unscoped "deny on any injection finding" rule repeats the '
        "**over-defense** failure the detection layer is already careful to avoid (`docs/phase1-"
        "baseline.md`), by blocking legitimate calls whose only sin is that an operator pasted a "
        "payload to discuss it or that the tripped finding was not an injection at all."
    )
    add("")
    add("## What a denial costs")
    add("")
    add(_OPERATIONAL_COST)
    add("")
    add("## Known gaps in this measurement")
    add("")
    for gap in _KNOWN_GAPS:
        add(gap)
    add("")
    add(
        f"_Report generated from a fixed clock ({report.checked_at}); the policy and corpus are "
        "deterministic, so a re-run reproduces these numbers exactly._"
    )
    return "\n".join(lines)


# =================================================================================================
# CLI
# =================================================================================================
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-guardian bench.policy_report",
        description="Measure the tool-boundary policy layer on the declared corpus.",
    )
    parser.add_argument(
        "--policy",
        dest="policy_file",
        type=Path,
        default=DEFAULT_POLICY_FILE,
        help=f"policy file to measure against (default: {DEFAULT_POLICY_FILE})",
    )
    parser.add_argument("--json", action="store_true", help="emit the report as JSON")
    return parser


def run_cli(argv: Sequence[str]) -> int:
    args = _build_parser().parse_args(list(argv))
    if not Path(args.policy_file).is_file():
        print(f"policy_report: policy file not found: {args.policy_file}", file=sys.stderr)
        return 2
    engine = build_engine(Path(args.policy_file))
    outcomes = evaluate(CASES, engine)
    report = measure(outcomes, Path(args.policy_file))
    if args.json:
        print(report.to_json())
    else:
        print(render_markdown(report))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run_cli(sys.argv[1:] if argv is None else argv)


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess
    raise SystemExit(main())
