# Evaluating CerbIA, Prompt Armor and Vigil against the guardian's own corpus

**Branch:** `feat/eval-cerbia` · **Base:** `main` @ `ed06335` · **Host:** Oracle ARM64 (aarch64), Python 3.12.14, no GPU, ~23 GB RAM.

This is the Phase-1 evaluation the design asks for in §4 ("Existing projects worth evaluating")
and §7 ("Evaluation order"). It answers one question with measurements: **what can be reused from
CerbIA, Prompt Armor and Vigil, and at what cost?** Every security number below was produced by the
repository's own harness (`llm_guardian.bench.harness.run`) over the same 146-case labelled corpus
(72 attack / 74 benign, EN 78 / ES 68, 13 attack families) the rules-only baseline already reports,
so the columns are directly comparable. The baseline to beat is the Phase-1 rules-only run:
**recall 0.375, precision 1.000, 0 false blocks/1,000 benign, escalation 0.055, p95 0.219 ms**
([`docs/phase1-baseline.md`](../phase1-baseline.md)).

> **The corpus caveat still applies.** The cases and the guardian's rules share an author, and now
> so does the comparison: a third-party detector measured on a corpus that quotes attacks
> verbatim in its benign half is being measured, deliberately, on the hard part. That is the point
> — the design's headline usability number is *false blocks per 1,000 benign developer requests*,
> and half this benign half is security work that merely *looks* like an attack.

---

## 1. Headline result

| Configuration | recall | precision | false blocks / 1,000 benign | escalation | p50 ms | p95 ms | p99 ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Guardian rules-only (baseline)** | **0.375** | **1.000** | **0.0** | 0.055 | 0.112 | **0.219** | 0.251 |
| CerbIA 0.3.0 rules-only (core) | 0.556 | 0.741 | 189.2 | 0.000 | 0.507 | 0.713 | 13.56 |
| CerbIA 0.3.0 rules-extended | 0.556 | 0.741 | 189.2 | 0.000 | 0.563 | 0.799 | 13.29 |
| CerbIA + ProtectAI DeBERTa (thr 0.5)† | 0.819 | 0.567 | 608.1 | 0.000 | 298.1 | 508.7 | 3241.4 |
| CerbIA + ProtectAI DeBERTa (thr 0.9)† | 0.764 | 0.567 | 567.6 | 0.000 | 91.4 | 230.7 | 1283.8 |
| Prompt Armor 0.9.1 (strict) | 0.375 | 0.711 | 148.6 | 0.068 | 23.97 | 40.58 | 507.4 |
| Prompt Armor 0.9.1 (warn→block) | 0.431 | 0.646 | 229.7 | 0.000 | 35.45 | 62.11 | 1058.6 |
| **Vigil 0.8.7** | — | — | — | — | — | — | — | *(does not install; see §5)* |

† The ProtectAI classifier is the model CerbIA's optional `cerbia-protectai` extra wires in, run
here directly on ONNX Runtime/CPU because the canonical install is not CPU-only on this host (§4).

**No candidate beats the baseline.** CerbIA roughly doubles recall (0.556 vs 0.375) and Prompt Armor
matches it exactly (0.375) while the ProtectAI classifier reaches 0.819 — but *every* one of them
pays for that with a false-block rate between 148 and 608 per 1,000 benign requests, against a
baseline of **zero**. The baseline doc is explicit that a guardian which flags its own security
documentation "gets switched off"; on this corpus all three third-party detectors do exactly that,
and Prompt Armor and the ProtectAI model also fail the design's 30 ms p95 target by 1–2 orders of
magnitude. The negative result is the deliverable: **keep our pipeline, add nothing at runtime.**

---

## 2. Method and reproducibility

- **Corpus:** `from llm_guardian.bench.corpus import CASES` — 146 cases, unchanged.
- **Harness:** `run(CASES, scan_callable, repeat=3, warmup=1, p95_target_ms=30.0)`. A `scan_callable`
  is `Case -> ScanObservation`; each wrapper times the library's own scan call and reports the
  verdict. `repeat=3` gives 438 pooled latency samples per config (Prompt Armor and CerbIA);
  a rules-only baseline of `repeat=3` was published in `phase1-baseline.md`.
- **Decision mapping.** The harness counts only `block` as a detection and reports `escalate`
  separately. CerbIA has no third state, so `not is_safe -> block`. Prompt Armor has four
  decisions; `strict` maps `block -> block`, `warn`/`uncertain -> escalate`; `aggressive` maps
  `warn -> block` to show the recall/overdefense cost of treating a warning as a catch.
- **All scans are `content_type=TEXT`** (CerbIA) / default input (Prompt Armor); the corpus is text.
- **Isolation.** Each library lives in its own venv (they conflict) under `/home/ubuntu/eval-venvs/`.
  `llm_guardian` is imported by putting the worktree's `src/` on `PYTHONPATH`; the only thing added
  to the eval venvs for the harness is `pydantic-settings` (2 packages, ~2 MB), which the library
  itself does not need.
- **Scripts** (all under `/home/ubuntu/eval-venvs/measure/`, kept out of the repo so the gates stay
  clean): `measure_cerbia.py`, `decompose_cerbia.py`, `measure_promptarmor.py`,
  `measure_protectai.py`. Raw reports: `cerbia-*.json`, `promptarmor-*.json`, `protectai-*.json`,
  `*-summary.json` in the same directory.
- **Latency caveat.** Timings are in-process on a shared, unaccelerated ARM VM; absolute latencies
  moved between runs (e.g. the ProtectAI p50 differed 3× between two adjacent passes). Treat the
  latency columns as order-of-magnitude, not a benchmark. Classification results are deterministic.

Commands:

```bash
# CerbIA (also runs decompose)
cd /home/ubuntu/eval-venvs/cerbia
PYTHONPATH=/home/ubuntu/guardian-eval-a/src ./bin/python /home/ubuntu/eval-venvs/measure/measure_cerbia.py
PYTHONPATH=/home/ubuntu/guardian-eval-a/src ./bin/python /home/ubuntu/eval-venvs/measure/decompose_cerbia.py
# Prompt Armor
cd /home/ubuntu/eval-venvs/prompt-armor
PYTHONPATH=/home/ubuntu/guardian-eval-a/src ./bin/python /home/ubuntu/eval-venvs/measure/measure_promptarmor.py 3
PYTHONPATH=/home/ubuntu/guardian-eval-a/src ./bin/python /home/ubuntu/eval-venvs/measure/measure_protectai.py
```

---

## 3. CerbIA 0.3.0

### Install, licence, size

| | |
| --- | --- |
| Version | `cerbia==0.3.0` (pulls `cerbia-core==0.3.0`) |
| Licence | **Apache-2.0** (repo, `cerbia`, `cerbia-core`; `NOTICE` + `LICENSE` shipped) |
| Install | clean on aarch64 / py3.12: `uv pip install cerbia` → **7 packages, 4 ms, no native build** |
| Size | **8.0 MB** on disk (core only) |
| Drags in | `pydantic~=2.13` and its stack. **No** ONNX / torch / transformers by default. |
| Python | `requires-python >=3.12` |

The core is genuinely light. Everything ML-shaped is behind extras that resolve to *separate*
packages and are not installed by default:

- `cerbia[cli]` → `cerbia-cli` (typer)
- `cerbia[ml]` → `cerbia-ml` → `optimum[onnxruntime]~=2.0` (→ torch + transformers + onnx/onnxruntime)
- `cerbia[presidio]` → `cerbia-presidio` → `presidio-analyzer~=2.2` + `en-core-web-sm` (spaCy model)
- `cerbia[protectai]` → `cerbia-protectai` → `cerbia-ml` (see §4)

### API surface (real signatures, from site-packages)

Entry point is documented and stable (the design already names it):

```python
# cerbia/core/security_gate/_gate.py
class SecurityGate:
    def __init__(
        self,
        name: str,
        scanners: list[Scanner],
        score_aggregator: ScoreAggregator,
        fail_fast: bool = True,
        threshold: float = 0.0,
        on_scanner_error: ScannerErrorPolicy = ScannerErrorPolicy.BLOCK,
    ) -> None: ...
    def scan(
        self, text: str, *, content_type: ContentType = ContentType.UNKNOWN
    ) -> GateVerdict: ...


# cerbia/core/security_gate/_models.py
class GateVerdict(BaseModel):  # is_safe: bool; score: float; rationale: str;
    is_safe: bool  # findings: list[Finding]; skipped_scanners: list[SkippedScanner]
    score: float
    rationale: str
    findings: list[Finding]
    skipped_scanners: list[SkippedScanner]


# cerbia/core/scanners/base.py — the plugin contract (typing.Protocol, @runtime_checkable)
class Scanner(Protocol):
    scanner_id: str
    scanner_name: str
    severity: Severity
    action: Action
    content_types: tuple[ContentType, ...] | None

    def scan(self, text: str) -> ScanOutcome: ...


# ScanOutcome: risk_score: float (0..1); rationale: str; matches: list[MatchSpan]
```

Config model (`cerbia/core/config.py`) — the piece the design calls "a configurable scanning and
verdict pipeline":

```python
class CerbIAConfig(BaseModel):  # model_config = ConfigDict(extra="forbid")
    name: str
    loaders: list[LoaderConfig] = []
    preprocessors: list[PreprocessorConfig] = []
    fail_fast: bool = True
    threshold: float = 0.0  # 0..1; 0.0 => any BLOCK finding fails the gate
    on_scanner_error: ScannerErrorPolicy = ScannerErrorPolicy.BLOCK
    scanners: list[ScannerConfig] = []
    score_aggregator: ScoreAggregatorConfig


# Runner(config).scan() -> ScanResult drives loaders -> preprocessors -> gate
```

Enumerations (`cerbia/core/types.py`):

- `Severity`: `CRITICAL` (weight 1.0), `HIGH` (0.75), `MEDIUM` (0.5), `LOW` (0.25).
- `Action`: `BLOCK` / `WARN` / `PASS`. `ScannerErrorPolicy`: `FAIL` / `BLOCK` / `SKIP`
  (fail-closed by default — a scanner that raises makes the gate unsafe).
- `ContentType`: `TEXT` / `URL` / `CODE` / `UNKNOWN` (a `UNKNOWN` content type runs *every*
  scanner; a typed one skips scanners that don't declare it).

Built-in scanners (`cerbia/core/scanners/`): **`PromptInjectionScanner`**, **`KeywordScanner`**,
**`InvisibleTextScanner`**, `SecretScanner`, `MaliciousUrlScanner`, `UrlAllowlistScanner`,
`XssScanner`, `PiiScanner`, `CanaryLeakScanner`. Aggregators
(`cerbia/core/score_aggregators/`): `MaxScoreAggregator`, `MaxWithBonusScoreAggregator(bonus=0.05)`,
`MeanScoreAggregator`. Preprocessors: `whitespace_normalization`, `speculative_decoding`.
Multilingual via an i18n regex registry (`en`, `es`, `gl`) with a **`defensive_context`** negation
guard — the analogue of our `quoted_discussion` provenance guard:

```python
# cerbia/core/scanners/keyword/_scanner.py
def is_in_defensive_context(text, match_start, defensive_patterns) -> bool:
    # looks at <=80 chars before the match, bounded by the line, for negation words
```

The config-driven `Runner` and the hand-built `SecurityGate` are the same gate; for a like-for-like
rules-only comparison I build the gate directly.

### Measured numbers (146 cases, repeat=3)

| Config | recall | precision | false blocks / 1k | escal. | p50 | p95 | p99 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rules-core: `PromptInjectionScanner(en,es)` + `KeywordScanner(en,es)` + `InvisibleTextScanner` | 0.556 (40/72) | 0.741 (40/54) | 189.2 (14/74) | 0.0 | 0.507 | 0.713 | 13.56 |
| rules-extended: core + `SecretScanner` + `MaliciousUrlScanner` + `XssScanner` | 0.556 | 0.741 | 189.2 | 0.0 | 0.563 | 0.799 | 13.29 |

Adding the secret/URL/XSS scanners changed **nothing** on this corpus — they neither caught a
missed attack nor added a false block (they are URL/CODE-scoped, so a `TEXT` pass mostly routes
around them). The core textual scanners are the whole story.

Per-family recall, against the baseline:

| Family | baseline | CerbIA core | Prompt Armor (strict) | ProtectAI DeBERTa |
| --- | --- | --- | --- | --- |
| instruction_override | 0.375 | **1.000** | 0.625 | **1.000** |
| system_prompt_exfiltration | 0.600 | **1.000** | 0.200 | **1.000** |
| delimiter_role_spoofing | 0.667 | 0.667 | 0.833 | **1.000** |
| tool_call_injection | 0.000 | 0.167 | 0.000 | 0.667 |
| indirect_injection_repo | 0.333 | 0.667 | 0.167 | **1.000** |
| invisible_text_smuggling | 0.600 | **1.000** | 0.600 | 0.800 |
| obfuscated_payload | **0.857** | 0.000 | 0.714 | **1.000** |
| exfiltration_request | 0.143 | 0.429 | 0.000 | 0.571 |
| agent_tool_misuse | 0.000 | 0.000 | 0.000 | 0.500 |
| jailbreak_persona | 0.400 | **1.000** | **1.000** | **1.000** |
| long_context_buried | 0.500 | **1.000** | 0.000 | 0.500 |
| markdown_image_exfiltration | 0.000 | 0.250 | 0.500 | 0.750 |
| credential_phishing | 0.333 | 0.000 | 0.000 | 0.667 |

**Where the false blocks come from.** The 14 benign false blocks are all in the *injection-shaped*
families — `benign_security_explainer` (6), `benign_owasp_checklist` (2), `benign_redteam_template`
(2), `benign_concept_discussion` (4). Scanner decomposition (`decompose_cerbia.py`):

| Scanner alone | recall | false blocks | tp |
| --- | --- | --- | --- |
| `PromptInjectionScanner(en,es)` | 0.194 | 5 | 14 |
| `KeywordScanner(en,es)` | **0.528** | **13** | 38 |
| `InvisibleTextScanner` | 0.069 | **0** | 5 |

Nearly all of CerbIA's recall *and* nearly all of its overdefense come from one scanner. Its
`keyword_suspicious_keywords` set matches bare tokens — `\binstructions?\b`, `\boverrides?\b`,
`\bignores?\b`, `\bsystem\s*prompts?\b`, `\bbypass`, `\bsudo\b`, `\bdeveloper\s*mode\b` — so any
document that *mentions* instructions, prompts or overrides is a hit. That is the naive keyword
matcher the baseline doc names as the overdefense trap (23 benign false blocks for a crude matcher).
CerbIA's `defensive_context` guard fires only on a narrow prefix window (`never`, `don't`, `refuse`,
`check for`, …) and does not fire on "a common prompt-injection payload is the phrase 'ignore
previous instructions'".

### What we would reuse, and what we would not

- **Reuse (design-level, not a dependency):** the *shape* of `CerbIAConfig` — an explicit
  `fail_fast` / `threshold` / `on_scanner_error(BLOCK|SKIP|FAIL)` / weighted-severity aggregation
  model — is a clean articulation of controls our pipeline already implements, and is useful as a
  cross-check that we expose the same knobs (we do: `ScannerErrorPolicy`, aggregation, thresholds).
  The **`PromptInjectionScanner`'s multilingual regex set** (EN/ES/GL, categories
  instruction_override / exfiltration / role_hijack / context_manipulation / privilege_escalation /
  fake_authority / task_deflection) is worth mining as **signature material** for our rules scanner:
  it is broad enough to close the lexical paraphrase gap (instruction_override 1.0) and
  `invisible_text`/`long_context` coverage comes free. Copying patterns is licence-safe (Apache-2.0).
- **Do not reuse:** CerbIA's `KeywordScanner` as a detector (it is the overdefense trap by
  construction), and CerbIA as the **orchestration layer**. Reusing the framework means adding a
  second gate/loader/preprocessor/gate stack in front of a pipeline that already implements
  scanning, aggregation, thresholds, provenance-aware suppression and a hard policy boundary — for
  no measured detection benefit and a large measured precision cost.
- **Integration cost in our terms:** `SecurityGate.scan()` is trivial to wrap (one callable,
  `GateVerdict.is_safe -> decision`), but wrapping it does not make it *useful*: we would have to
  keep the `defensive_context` idea but replace CerbIA's detector with ours, at which point we are
  back to our own pipeline plus a foreign config model. Not worth the dependency.

---

## 4. CerbIA + ProtectAI (the optional classifier)

`cerbia-protectai==0.3.0` (**Apache-2.0**) depends on `cerbia-ml==0.3.0`, which depends on
`optimum[onnxruntime]~=2.0`. A dry-run resolve of `cerbia[protectai]` on this ARM64 host selects
**40 packages including `torch==2.14.1` and the full NVIDIA CUDA runtime**
(`nvidia-cudnn-cu13`, `nvidia-cublas`, `nvidia-cusolver`, `nvidia-nccl-cu13`, `nvidia-nvshmem-cu13`,
…), plus `onnx`, `onnxruntime`, `optimum`, `optimum-onnx`, `transformers` — a multi-gigabyte, GPU
torch stack. The scanner itself is configured CPU-only
(`DEFAULT_MODEL_KWARGS={"provider": "CPUExecutionProvider"}`, `pipeline_kwargs={"device": "cpu"}`)
and its default is `local_files_only=True`, but the *install* is not CPU-only, so enabling it as the
design intends ("if it can be enabled without GPU") is not clean here. **I did not install it and
did not measure `cerbia-protectai` as shipped.**

What I did measure is the exact model and tokenizer it names, run identically on ONNX Runtime/CPU
(the scanner's own inference path), in `measure_protectai.py`:

| | |
| --- | --- |
| Model | `protectai/deberta-v3-base-prompt-injection-v2`, revision `89b085cd…`, `onnx/model.onnx` |
| Licence | **Apache-2.0** (model card `license: apache-2.0`; base `microsoft/deberta-v3-base`) |
| Size | `model.onnx` **738.6 MB** (fp32, not quantised) + `onnx/tokenizer.json` 8.6 MB |
| Labels | `id2label = {"0": "SAFE", "1": "INJECTION"}` (verified; label orientation sanity-checked:
  a benign coding request scores 0.000, a direct override 1.000) |

| Threshold | recall | precision | false blocks / 1k | p50 | p95 | p99 |
| --- | --- | --- | --- | --- | --- | --- |
| 0.5 (model's decision boundary) | 0.819 (59/72) | 0.567 | **608.1 (45/74)** | 298.1 | 508.7 | 3241.4 |
| 0.9 | 0.764 (55/72) | 0.567 | 567.6 (42/74) | 91.4 | 230.7 | 1283.8 |

Raising the threshold barely helps — the false blocks are confident, not marginal. **The model is
strongly over-defensive on exactly our benign half:** it flags all ten `benign_security_explainer`
cases (it scores the verbatim quote *"A common prompt-injection payload is the phrase 'ignore
previous instructions'"* at **1.000**), all ten `benign_spanish_engineering` cases, most
`benign_repo_readme`/`benign_concept_discussion`/`benign_cve_description` cases. Two problems stand
out:

1. **No provenance awareness.** The classifier cannot tell a document *about* an attack from an
   attack; our baseline's zero false blocks come from exactly that distinction.
2. **A Spanish/multilingual blind spot.** The benign *Spanish engineering chatter* family
   (`bench-b-se-02…10`, no attack content at all) is almost entirely flagged, while recall on the
   attack side stays high — a signal that the model treats non-English text as suspicious. This is
   a disqualifying behaviour for an EN/ES corporate gateway.

**What to reuse:** the classifier *is* the kind of local, CPU-only, Apache-2.0 `ml` scanner the
design wants, and our existing `OnnxMlScanner` seam (an ONNX session + an `AutoTokenizer`, CPU
provider, one sigmoid score) would load it with **no code change**. But on this corpus it does not
beat the rules-only baseline on the tradeoff that matters (recall +0.444, false blocks +608/1,000),
so the honest next experiment is *rules + this classifier as advisory behind the provenance /
`quoted_discussion` guard* — not switching it on. **Do not make the raw DeBERTa a default.**

---

## 5. Prompt Armor 0.9.1

### Install, licence, size

| | |
| --- | --- |
| Version | `prompt-armor==0.9.1` |
| Licence | **Apache-2.0**; PyPI classifier **`Development Status :: 3 - Alpha`** |
| Install | clean on aarch64 / py3.12: **37 packages**, no native build |
| Size | **367 MB** as installed; the wheel ships rule/signature data (19 MB) but **not the models** |
| Drags in | `onnxruntime`, `faiss-cpu`, `scikit-learn`, `numpy`, `scipy`, `tokenizers`, `huggingface-hub`, `regex`, `rich` |

**Runtime model downloads.** First use downloads model artifacts *into site-packages*:

- L3 embedding model `prompt-armor/l3-contrastive-onnx` (Apache-2.0): `model_quant.onnx` **113 MB**
  + `tokenizer.json` **17 MB**.
- L5 anomaly model `prompt-armor/l5-negative-selection` (Apache-2.0): `l5_negative_selection.pkl`
  **3.1 MB** (loaded via `joblib`, sha256-verified before deserialisation).
- L2 classifier `aldenb/scout-prompt-injection-classifier-22m`, pinned revision `1b28ceb5…`:
  **download fails (HTTP 401 / "Repository Not Found")** — the pinned model is gone or gated. L2
  therefore silently falls back to a keyword heuristic. This is a real defect for anyone trusting
  the default pipeline.

Net on disk after first run: **537 MB** (≈367 MB install + 133 MB models + ~2 MB harness glue).

### API surface (real signatures)

```python
# prompt_armor/__init__.py
def analyze(text: str, session_id: str | None = None) -> ShieldResult: ...


# prompt_armor/engine.py
class LiteEngine:
    def __init__(self, config: ShieldConfig | None = None) -> None: ...
    def analyze(self, text: str, session_id: str | None = None) -> ShieldResult: ...
    @property
    def active_layers(self) -> list[str]: ...


# prompt_armor/models.py (dataclasses)
@dataclass(frozen=True, slots=True)
class ShieldResult:
    risk_score: float
    confidence: float
    decision: Decision
    categories: tuple[Category, ...]
    evidence: tuple[Evidence, ...]
    needs_council: bool
    latency_ms: float
    cost_usd: float
    layer_results: tuple[LayerResult, ...]


class Decision(str, Enum):
    ALLOW = "allow"
    WARN = "warn"
    BLOCK = "block"
    UNCERTAIN = "uncertain"
```

Layers are auto-discovered and fail-open: L1 regex + L4 structural always; L2 (ONNX), L3 (`faiss`),
L5 (`sklearn`) load if importable. Fusion is a hard-coded logistic-regression meta-classifier
(`prompt_armor/fusion.py`, `_META_THRESHOLD = 0.50`), then `_decide`: `score < 0.5 -> ALLOW`,
`0.5..0.8 -> WARN`, `>= 0.8 -> BLOCK`, with a `hard_block=0.95` path requiring ≥2 corroborating
layers. Thresholds live in `ShieldConfig.thresholds` (`allow_below=0.55`, `block_above=0.7`,
`hard_block=0.95`, `min_confidence=0.5`, `jitter_sigma=0.0`) and layer weights in `LayerWeights`.

Config model (`prompt_armor/config.py`):

```python
class ShieldConfig(BaseModel):
    weights: LayerWeights  # l1_regex .20, l2_classifier .30, l3_similarity .30, l4_structural .20
    thresholds: ThresholdConfig
    convergence_boost: float = 0.10
    divergence_penalty: float = 0.15
    rules_path: Path | None = None  # override data/rules/default_rules.yml
    attacks_path: Path | None = None  # override the known-attack corpus
    analytics: AnalyticsConfig
    council: CouncilConfig
```

Active layers on this host: `['l1_regex', 'l4_structural', 'l3_similarity', 'l2_classifier',
'l5_negative_selection']` (L2 heuristic because the download 401'd).

### Measured numbers (146 cases, repeat=3)

| Mapping | recall | precision | false blocks / 1k | escal. | p50 | p95 | p99 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| strict (`WARN/UNCERTAIN → escalate`) | 0.375 (27/72) | 0.711 | 148.6 (11/74) | 0.068 | 23.97 | 40.58 | 507.4 |
| aggressive (`WARN → block`) | 0.431 (31/72) | 0.646 | 229.7 (17/74) | 0.000 | 35.45 | 62.11 | 1058.6 |

Prompt Armor's headline recall equals the baseline's **exactly** (0.375) while adding 148.6 false
blocks per 1,000 and ~40 ms at p95 — it is dominated by the baseline on both axes. It does move
detections *between* families (better on the lexical/obfuscation families, worse on the systemic
ones), which is the useful part of the picture:

- **Better than baseline:** obfuscated_payload 0.714, delimiter_role_spoofing 0.833,
  jailbreak_persona 1.000, markdown_image_exfiltration 0.500, instruction_override 0.625.
- **Worse than baseline:** long_context_buried **0.000**, system_prompt_exfiltration 0.200,
  exfiltration_request **0.000**, indirect_injection_repo 0.167. It misses the long-context
  payloads entirely (the ~3 KB benign document with a late payload).
- **Still zero:** agent_tool_misuse, credential_phishing, tool_call_injection.

Its 11–17 false blocks are the familiar benign security set: `benign_security_explainer`,
`benign_concept_discussion`, `benign_redteam_template`, `benign_incident_writeup`,
`benign_cve_description`, `benign_spanish_engineering`. Its bundled data is nonetheless the most
directly reusable artefact of the three libraries:

- `data/rules/default_rules.yml` — a labelled L1 rule set (`PI-001 "ignore … previous
  instructions"` weight 0.92, etc.).
- `data/attacks/known_attacks_v2.jsonl` (1,509 curated attack entries, used for the 1,488-vector
  bundled FAISS index) and `data/attacks/benign_exemplars.jsonl` (80 injection-shaped-but-benign
  exemplars with a **benign-margin gate**: suppress the score when the input is at least as similar
  to a benign exemplar as to any attack). That margin idea is the closest thing here to our
  provenance guard.

**What to reuse:** the **`default_rules.yml` regex set** and the **benign-exemplar margin idea**
(signature material), and the *concept* of a prebuilt FAISS similarity index keyed on a content
signature. **What not to reuse:** the engine as a dependency — 37 packages + 133 MB of
runtime-downloaded models for recall equal to the baseline and a 148/1,000 false-block rate is a
bad trade; and the L2 classifier contract is currently broken (pinned model absent). **Integration
cost:** it is a drop-in callable (`analyze() -> ShieldResult`), but a drop-in that must run in its
own process/venv because of `onnxruntime`+`faiss`.

---

## 6. Vigil (deadbits/vigil-llm) 0.8.7

**Not on PyPI.** Upstream is a source repo; the install the design contemplates does not work here.

```
$ uv pip install "git+https://github.com/deadbits/vigil-llm"
  ...
  × Failed to build `pandas==2.0.0`
  ╰─▶ Call to `setuptools.build_meta:__legacy__.build_wheel` failed (exit status: 1)
      ModuleNotFoundError: No module named 'pkg_resources'
  hint: `pandas` (v2.0.0) was included because `vigil-llm` (v0.8.7) depends on `pandas`
```

**Exact reason.** `vigil-llm==0.8.7` (`setup.py`) pins a 2023 stack for Python ≤ 3.11:
`pandas==2.0.0`, `numpy==1.25.2`, `pyarrow==14.0.1`, `openai==1.0.0`, `transformers==4.36.0`,
`pydantic==1.10.7`, `yara-python==4.3.1`, `sentence-transformers==2.2.2`, `chromadb==0.4.17`,
`streamlit==1.26.0`, `nltk==3.8.1`, `datasets==2.15.0`. `pandas==2.0.0` has no cp312 wheel, so uv
builds it from source, and its legacy build backend imports `pkg_resources`, which current
setuptools no longer ships — the build dies. Even past that, `pydantic==1.10.7` predates py3.12
wheels and `yara-python==4.3.1` must compile against libyara. This is not a route-around problem
worth solving for a library its own README calls **"in an alpha state … experimental / for research
purposes"**; the failure is itself the finding, and it matches the design's own caution.

- **Licence:** Apache-2.0 (repo `LICENSE`).
- **Footprint:** the clone is 576 KB; a working install would pull torch/transformers/chromadb/
  sentence-transformers — the same multi-GB class as `cerbia[protectai]`.
- **API surface (from source, for reference):**
  `Vigil(config_path: str)` builds `input_scanner` / `output_scanner` `Manager`s from a
  `ScannerRegistry`; scanners are modular (`vigil/scanners/`: `yara.py`, `similarity.py`,
  `vectordb.py`, `transformer.py`, `sentiment.py`, `relevance.py`) with optional vector DB
  (`chromadb`), embedder (`sentence-transformers`), canary tokens and an auto-updating vector DB.
- **What to reuse (this is the useful bit):** its **YARA signature files**,
  `data/yara/*.yar` — ten files including `instruction_bypass.yar`, `system_instructions.yar`,
  `mdexfil.yar` (markdown-image exfiltration), `generic_secret.yar`, `apitokens.yar`, `ssh.yar`,
  `guidance.yar`, `react.yar`. They are plain signature text, licence-safe (Apache-2.0), and map
  directly onto our gaps (markdown-image beacon, exfiltration phrasing). Treat as **reference
  material to port**, not a dependency.

---

## 7. Which failure clusters each library closes

The baseline doc types the 45 misses into four causes. Mapping the measured per-family recall onto
them:

| Cluster (baseline cause) | CerbIA rules | Prompt Armor | ProtectAI DeBERTa | Verdict |
| --- | --- | --- | --- | --- |
| **1. Paraphrase gap** (lexical rules miss reworded overrides) | **closed** — instruction_override 1.0 via broader lexical patterns, but by bare-word matching | partial — instruction_override 0.625, jailbreak 1.0 | **closed** — instruction_override 1.0 | Closed *at the cost of* precision; no source closes it without over-flagging. |
| **2. Tool-call shape** (structural, 0/6) | not closed — tool_call 0.167, agent_tool_misuse 0.0 | not closed — 0.0 / 0.0 | partial — tool_call 0.667, agent_tool_misuse 0.5 | None models the *structural* signal; a declared structural rule + policy check is still the fix. |
| **3. Markdown image beacon** (0/4) | 0.25 | 0.5 | 0.75 | Partial only; no deterministic URL-shape rule — the contained fix in `scanners/urls.py` still stands. |
| **4. Exfiltration phrasing** (1/7) | partial — exfiltration_request 0.429 | 0.0 | 0.571 | Partial; the (sensitive source + external destination) pair remains the right signal. |

Nobody closes clusters 2–4 with a classifier; those are the cheap deterministic fixes the baseline
already recommends.

---

## 8. Recommendation

**Keep our pipeline and add nothing at runtime.** The evidence:

1. **CerbIA rules-only** roughly doubles recall (0.556 vs 0.375) but at **189 false blocks per
   1,000 benign** (baseline 0), and the recall comes almost entirely from a `KeywordScanner` that
   matches bare words (`instructions`, `override`, `bypass`, `sudo`) — literally the naive keyword
   matcher the baseline names as the trap. Its framework is clean and Apache-2.0, and fast
   (p95 0.71 ms), but our pipeline already implements the same gate concepts and the same knobs.
   **Adopting it as the orchestration layer buys no detection benefit and reinstates overdefense.**
2. **Prompt Armor** matches the baseline's recall exactly (0.375) and adds 148.6 false blocks/1,000
   and ~40 ms p95. Its L2 model download is broken. **Not an improvement.**
3. **CerbIA + ProtectAI DeBERTa** reaches recall 0.819 but flags **608/1,000** benign requests
   (60%+), including all benign security explainers (verbatim quotes score 1.000) and almost all
   benign Spanish engineering text, at 90–300 ms p50 on CPU. **High recall, unusable alone.**
4. **Vigil** does not install on py3.12/ARM64 and is self-described alpha.

No candidate *clearly beats* the rules-only baseline on the metric the design prioritises
(recall gained vs false blocks gained). Per the task's rule, **`src/llm_guardian/scanners/ml.py` is
left unchanged**; the ML seam already fits a local ONNX classifier if a future experiment clears the
bar.

**Borrow, don't depend — as documentation-level signature material for our rules scanner** (all
Apache-2.0, no new runtime dependency):

- CerbIA's EN/ES/GL `PromptInjectionScanner` patterns and the `defensive_context` negation idea
  (we already do this better with provenance, but its Spanish patterns are a useful cross-check).
- Prompt Armor's `data/rules/default_rules.yml` and `data/attacks/benign_exemplars.jsonl`
  (the benign-margin idea).
- Vigil's `data/yara/instruction_bypass.yar`, `mdexfil.yar`, `system_instructions.yar`.

**Next experiment (not this ticket):** rules-only + the ProtectAI DeBERTa classifier as an *advisory*
`ml` scanner **behind the existing provenance/`quoted_discussion` guard**, measured on the same
corpus. The classifier's raw recall (0.82, including obfuscation and tool-call shaping our rules
miss) makes it the only candidate worth re-testing; the false blocks above are a guard problem, not
a classifier problem.

---

## 9. What could not be measured, and why

- **`cerbia-protectai` as shipped.** Not installed: its dependency chain (`cerbia-ml ->
  optimum[onnxruntime]~=2.0`) resolves a full GPU torch + NVIDIA CUDA stack on this ARM64 host
  (dry-run: 40 packages incl. `torch==2.14.1`, `nvidia-*`), i.e. not "without GPU". The exact model
  and tokenizer it names were run directly on ONNX Runtime/CPU instead, and labelled as such.
- **CerbIA/L5 `PiiScanner` and the Presidio extra** (`cerbia[presidio]`: `presidio-analyzer`,
  `en-core-web-sm`): out of the corpus' scope (no PII cases) and heavy; not installed.
- **Prompt Armor's L2 ONNX classifier**: its pinned model is unreachable (HTTP 401 / not found), so
  the run reflects the heuristic fallback. A future measuring run with a valid model could change
  L2's contribution.
- **Prompt Armor's `council`** (LLM judge) and **inflammation cascade** (needs a `session_id`): not
  exercised; the harness is stateless per case.
- **Latency is not a benchmark**: in-process, on a shared unaccelerated VM; magnitudes moved between
  passes. Classification numbers are deterministic and reproducible.
- **Absolute install sizes include harness glue**: each eval venv also received `pydantic-settings`
  (~2 MB) purely so the in-repo harness could import; that is not a dependency of the library.
