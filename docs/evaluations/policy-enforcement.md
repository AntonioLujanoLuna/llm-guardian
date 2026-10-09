# Policy enforcement at the tool boundary — measured

**This corpus is synthesized in this repository, by the same author as the policy it measures.** Every number below is a *design-cost estimate*: zero false denials says the rules do what they were written to do on cases written alongside them, not that they generalise to real traffic. The unbiased figure is missing until the corpus is replaced by traffic the rules were never written against. The detectors are not touched or measured here.

## How to reproduce

```bash
unset PYTHONPATH
PYTHONPATH=src .venv/bin/python -m llm_guardian.bench.policy_report --policy policies/example.toml
```

Policy file: `policies/example.toml`. Corpus: **192 cases (89 legitimate / 103 malicious)**.

## Headline

| Metric | Measured |
| --- | --- |
| False denials per 1,000 legitimate tool calls | **0.0** (0/89) |
| Malicious calls denied (overall) | **100.0%** (103/103) |
| Cases where the evidence rule fired | 12 |

## Malicious denial rate by family

| Family | Denied | Rate |
| --- | --- | --- |
| `argument_pattern_violation` | 5/5 | 100.0% |
| `argument_too_long` | 5/5 | 100.0% |
| `capability_escalation` | 7/7 | 100.0% |
| `confused_deputy` | 12/12 | 100.0% |
| `cross_tenant_read` | 5/5 | 100.0% |
| `cross_tenant_write` | 5/5 | 100.0% |
| `curl_pipe_shell` | 5/5 | 100.0% |
| `destructive_shell` | 6/6 | 100.0% |
| `exfiltration_egress` | 6/6 | 100.0% |
| `external_recipient` | 5/5 | 100.0% |
| `host_pivot` | 6/6 | 100.0% |
| `missing_required_argument` | 5/5 | 100.0% |
| `path_traversal_write` | 6/6 | 100.0% |
| `plugin_mass_install` | 4/4 | 100.0% |
| `sensitive_file_read` | 7/7 | 100.0% |
| `shell_metacharacter` | 4/4 | 100.0% |
| `unauthorized_tool` | 6/6 | 100.0% |
| `undeclared_argument` | 4/4 | 100.0% |

## Legitimate false denials by family

| Family | Denied | Rate /1,000 |
| --- | --- | --- |
| `awkward_legitimate` | 0/6 | 0.0 |
| `deploy_to_staging` | 0/5 | 0.0 |
| `evidence_operator` | 0/5 | 0.0 |
| `evidence_untrusted_nontrigger` | 0/6 | 0.0 |
| `fetch_dependency` | 0/6 | 0.0 |
| `file_read_workspace` | 0/6 | 0.0 |
| `internal_channel_post` | 0/5 | 0.0 |
| `internal_mail` | 0/5 | 0.0 |
| `open_pull_request` | 0/5 | 0.0 |
| `own_database_query` | 0/6 | 0.0 |
| `own_tenant_retrieval` | 0/3 | 0.0 |
| `privileged_legitimate` | 0/9 | 0.0 |
| `proposed_by_untrusted_no_evidence` | 0/6 | 0.0 |
| `run_test_suite` | 0/6 | 0.0 |
| `search_workspace` | 0/5 | 0.0 |
| `update_owned_ticket` | 0/5 | 0.0 |

## Every false denial, named

None: no legitimate case was denied on this run.

## Split by the provenance that proposed the call

| Proposer | Legitimate | False denials | False denials /1,000 | Malicious | Denied | Denial rate |
| --- | --- | --- | --- | --- | --- | --- |
| operator | 77 | 0 | 0.0 | 89 | 89 | 100.0% |
| untrusted | 12 | 0 | 0.0 | 14 | 14 | 100.0% |

## Split by whether the evidence rule was needed

| Group | Legitimate | False denials | False denials /1,000 | Malicious | Denied | Denial rate |
| --- | --- | --- | --- | --- | --- | --- |
| evidence rule exercised | 11 | 0 | 0.0 | 12 | 12 | 100.0% |
| no evidence supplied | 78 | 0 | 0.0 | 91 | 91 | 100.0% |

## Against the naive alternatives

| Rule | False denials /1,000 legitimate | Malicious caught | Catch rate |
| --- | --- | --- | --- |
| deny nothing | 0.0 (0) | 0 | 0.0% |
| detector-only (deny on any injection finding) | 78.7 (7) | 12 | 11.7% |
| policy + scoped evidence rule (this system) | 0.0 (0) | 103 | 100.0% |

The detector-only row is the reason the evidence rule is scoped to *untrusted provenance* and to three finding kinds: an unscoped "deny on any injection finding" rule repeats the **over-defense** failure the detection layer is already careful to avoid (`docs/phase1-baseline.md`), by blocking legitimate calls whose only sin is that an operator pasted a payload to discuss it or that the tripped finding was not an injection at all.

## What a denial costs

A denial is not one thing. A blocked file read inside the workspace costs a retry; a blocked `deploy_staging` costs a release window and an on-call page; a blocked `send_mail` or `post_message` costs a quiet status update; a blocked cross-tenant read costs a support ticket that cannot be answered. The false-denial *rate* pools them, so the cases are listed by id above: on this run the operator can see exactly which legitimate calls a tightened rule would have held. The scoped evidence rule is what keeps the confused-deputy catch from turning every quoted injection in an operator's own paste into a blocked tool call.

## Known gaps in this measurement

Stated plainly, because a limitation left implicit reads as a claim:

1. **The corpus is synthesized here**, by the same author as the policy, so 0/1,000 is a design-cost estimate and not a generalisation result. It is not evidence about traffic the rules were never written against.
2. **The cross-tenant rule reuses the read grant for writes.** A tool call (a write) whose `target_tenant` differs from the principal's own is denied unless the target tenant lists the principal in `allowed_cross_tenant_reads` -- the only grant key that exists. Fail-closed, but a read grant named `..._reads` currently also authorises a write to that tenant; a separate write grant is the missing piece.
3. **`run_shell` constrains command *shape*, not the files its arguments name.** The schema requires a safe first token and forbids shell metacharacters, so `rm -rf` and curl-pipe-to-shell are denied; but a command built from an allow-listed binary (`cat`, `ls`) is not path-checked, so `cat /etc/shadow` would pass the shell schema. File-confinement is the job of `read_file`'s `path_within`, which is where the sensitive-read cases are caught.
4. **No case here drives a scanner error or an unreadable policy file** through the endpoint; the engine fails closed for both (unit-tested separately), and the corpus does not add a malicious case that relies on those paths.
5. **Latency is not measured here.** The policy check is deterministic and in-process; the detection layer's latency is reported in `docs/phase1-baseline.md`.

_Report generated from a fixed clock (2020-01-01T00:00:00+00:00); the policy and corpus are deterministic, so a re-run reproduces these numbers exactly._
