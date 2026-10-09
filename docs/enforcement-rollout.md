# Enforcement rollout

The service ships in **shadow** (`GuardMode`, default `shadow`): every verdict is computed and
returned with `shadow: true`, and nothing is held. This document is the case for flipping any part of
it, using the numbers already measured in this repository — and it deliberately separates what
enforcing would buy from what it would cost, because the second number is the one that decides it.

Nothing here is a promise about real traffic. Every figure below comes from a corpus in this repo,
and the corpus each one came from is named.

## What enforcing would have cost, measured

| Corpus | Cases | Recall | False blocks / 1,000 benign | Source |
| --- | --- | --- | --- | --- |
| Internal regression corpus | 72 attacks / 1,000 benign | 1.000 | **0.0** | `docs/phase1-baseline.md` |
| Public holdout (7 datasets) | 1,270 attacks / 670 benign | 0.136 | **20.9** (14 cases) | `docs/evaluations/external-holdout.md` |
| Tool boundary (policy layer) | 103 malicious / 89 legitimate calls | 1.000 | **0.0** | `docs/evaluations/policy-enforcement.md` |

Read together, those three rows say:

- **On the traffic this detector was built against, enforcing is free** — 0 false blocks in 1,000
  benign internal cases, at p95 0.227 ms against a 30 ms budget.
- **On public benign traffic it would hold 14 cases in 670** (20.9 per 1,000). Read them before
  deciding: most are label disputes, not misfires — a real session JWT inside a response the dataset
  labels `safe`, and phishing prompts JailbreakBench calls benign. Four carry a decodable injection
  payload in the prompt's own text, where holding is defensible even though the dataset labels the
  prompt benign. For a guardian, flagging suspicious material while the gateway answers is the
  intended behaviour.
- **Enforcing does not make the service secure against most of what the public corpora contain.**
  86% of the holdout's attack cases draw no finding at all, because most are harmful-content and
  policy-compliance asks ("give me the steps to do this"), not trust-boundary violations. That is
  scope, decided deliberately, and no threshold change moves it. A moderation component is the answer
  there and it is a different component with a different false-positive budget.

## Order of enforcement, by how much evidence stands behind each layer

1. **The policy layer at the tool boundary** — enforce first. It is deterministic, reads declared
   principals/capabilities/egress/grants, never consults a model, and measured 0 false denials on 89
   legitimate tool calls while denying 103/103 malicious ones. The one thing to configure before
   enabling it is `PolicySettings.file`: with no policy file the engine fails closed and denies every
   tool call, which `/v1/health` reports as `degraded / policy_unavailable`.
2. **Deterministic findings at or above `severity_floor`** (default `critical`) — enforce. These block
   on their own regardless of score, and every false block this layer has produced is listed by name
   in the holdout report.
3. **The scoring band** — `escalate_at` 0.35 to `block_at` 0.75 — escalate, do not block. Note the
   arithmetic that catches people out: a single `high` finding scores 0.75, which *is* `block_at`, so
   "high" is already a blocking severity, not an escalating one.
4. **The optional local classifier (`ml`)** — remains off. The best candidate measured here
   (ProtectAI DeBERTa v3) blocked **87.8% of benign Spanish** against 27.3% of benign English. A
   bilingual deployment cannot enable it on that split, and the repo reports ML coverage as *absent*
   rather than clean while it is off.

## Staging

1. **Shadow** (today), watching: verdict distribution, escalation rate (2.7% on the internal corpus),
   p95 latency, and the named false blocks.
2. **Enforce the deterministic high-severity layer** (items 1 and 2 above) on one principal or one
   route — a canary with real traffic and a short review window.
3. **Enforce broadly, with a declared exception list.** Exceptions belong in the policy file, next to
   the principals they apply to, and not in code: a rule that exists only as a branch is a rule nobody
   can audit.
4. **Only then** consider promoting the scoring band to blocking, and only with the false-block rate
   re-measured after each change.

## Rollback

One setting: `GUARDIAN_MODE=shadow`. Nothing in the enforce path is a one-way door — the verdict is
computed identically in both modes and only the *act* differs, so rollback cannot change a decision
that a downstream system already saw, and it needs no deploy.

## The decision this repository cannot make for itself

The false-block budget. Stated as a number, because "as few as possible" is not a budget:

- **≤2 per 1,000 benign requests**: enforcement covers the deterministic families only; the
  scoring band stays advisory, and roughly 86% of public-corpus attacks remain out of scope by design.
- **≤10 per 1,000**: the same layers, with the near-miss directives promoted to blocking and a wider
  `severity_floor` (for example `high`).
- **Anything above that** is buying recall on families the rules were never written for, and should
  be spent on a moderation component instead.

Two smaller calls worth making at the same time: whether the public false blocks above justify an
exception-list entry each, and whether the risk memory may ever do more than advise. **Today it is
explicitly advisory**, and the measurement is now in: `docs/evaluations/risk-memory.md` finds
**0 of 50 scenarios changed by a precedent** — including the 10 the corpus declared should change,
where a repeat benign pattern escalates exactly as it did on first sight, and the three critical
cases that stay blocked. So the three poisoning attempts that try to suppress a block are refused
trivially: there is no suppression path to reach.

That same measurement is what should decide the second call, and it argues for caution. The memory
is a token-set cosine over stored text with no principal, destination, path or role in the feature —
**seven of twelve near-misses retrieve at ≥ 0.71**, and "export the log to `paste.example`" is
**0.909** similar to "export the log to `transfer.example`". A precedent wired to de-escalate on
similarity would therefore apply across exactly the differences that matter: a reviewed-benign
`staging` deployment would whitelist `production` at 0.875 similarity. Enabling suppression needs a
feature that carries the principal, the destination and the path, not a threshold on a text
embedding; until then, advisory is the correct state and the 0 is a property of the design rather
than a tested claim about a working one.

## What enforcing still could not claim

- **No real-traffic measurement.** Every number above is from a corpus in this repository. A shadow
  period on real traffic is the only thing that produces the number that matters.
- **No concurrency measurement.** The latency figures are single-request, in-process; the block path
  under load has not been measured.
- **No adversarial evaluation of the enforce path itself** — what an attacker does when a call is
  denied, and whether the denial leaks anything (a tool name, a policy key, a principal's
  capabilities), is unmeasured. The policy response returns stable reason codes on purpose; that is a
  design decision with a privacy trade-off that has not been tested.
