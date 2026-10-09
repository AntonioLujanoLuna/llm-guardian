# Hugging Face prompt-injection detectors — CPU/ARM64 evaluation

The missing measurement the design asks for in section 4: run the named model baselines against the
guardian's own 146-case corpus and the deterministic rule layer, on the target host class (a CPU-only
ARM64 VPS), and decide whether any of them earns a place behind the `ml` scanner seam.

**Headline:** both measurable models raise recall on attack **families** the blind rules could not see
at all, but on this corpus neither is a net win — each false-blocks 378–608 of every 1,000 benign
requests and blows the 30 ms p95 target by 6–100×. Between the two, **PIGuard** is the better
*advisory* scanner (fewer false blocks, MIT, ungated); **neither should ship as a blocking control**.
Prompt Guard 2 could not be measured at all — both checkpoints are gated.

> **Read this before the numbers.** The corpus is a **saturated regression suite**, not a validation
> set. The deterministic rules were tuned on these exact 72 attack cases after the first blind run
> and now score **1.000 recall / 0 false blocks** (`docs/phase1-baseline.md`, "Second measurement").
> A model that also scored ~1.000 here would have shown nothing. The value of this run is therefore
> **per-family behaviour and the false-block cost**, measured with cases neither model was written
> against — not the headline recall.

---

## 1. Environment (measured)

| | |
| --- | --- |
| Host | Oracle free-tier **ARM64** (aarch64) VPS, **CPU only, no GPU** |
| Cores | `nproc` = **4** |
| Python | 3.12.14 (isolated venv `/home/ubuntu/eval-venvs/hf`) |
| Runtime | `onnxruntime` **1.31.0**, `transformers` **5.19.0**, `onnx` **1.23.2**, `tokenizers` **0.23.3**, `numpy` **2.5.3** |
| `torch` | 2.14.1 installed **only** to export PIGuard to ONNX, then **removed** |
| Inference | ONNX Runtime, `CPUExecutionProvider`, DeBERTa-v2/v3 graph, fp32 |

Disk: the first `torch` install pulled the **full CUDA stack** (`nvidia-*` cu13, nccl, cusolver,
cusparselt, triton) onto a GPU-less host and bloated the venv to **5.4 GB**. After uninstalling the
CUDA packages and the post-export `model.safetensors`, the venv is **360 MB** and the two ONNX
checkpoints are **715 MB** (base) and **711 MB** (PIGuard). `df -h /`: **23 GB free** of 45 GB.
README for the next person: `uv pip install torch --index-url https://download.pytorch.org/whl/cpu`
if you only need CPU.

## 2. Download verdicts

| Model | id | Licence | Gating | Verdict |
| --- | --- | --- | --- | --- |
| ProtectAI DeBERTa v3 **base** v2 | `protectai/deberta-v3-base-prompt-injection-v2` | Apache-2.0 | ungated | **OK** (19.8 s, ONNX weights published in-repo) |
| ProtectAI DeBERTa v3 **small** v2 | `protectai/deberta-v3-small-prompt-injection-v2` | Apache-2.0 | `gated=auto` | **BLOCKED — HTTP 401** |
| PIGuard | `leolee99/PIGuard` | MIT | ungated | **OK** (22.5 s, safetensors + custom code) |
| Llama Prompt Guard 2 22M | `meta-llama/Llama-Prompt-Guard-2-22M` | Llama 4 Community License (`license: other`) | `gated=manual` | **BLOCKED — no credentials** |
| Llama Prompt Guard 2 86M | `meta-llama/Llama-Prompt-Guard-2-86M` | Llama 4 Community License (`license: other`) | `gated=manual` | **BLOCKED — no credentials** |

**Small DeBERTa v2 (401, exact reason):** `Cannot access gated repo … Access to model
protectai/deberta-v3-small-prompt-injection-v2 is restricted. You must have access to it and be
authenticated to access it. Please log in.` `gated=auto` accepts the terms automatically but still
requires an authenticated HF account; there is no public path. Not worked around.

**Prompt Guard 2 (blocked only far enough to be actionable).** Both checkpoints are `gated=manual`.
To run them a person must, on huggingface.co: (1) sign in, (2) open each model page and **accept the
Llama 4 Community License / acceptable-use terms**, then (3) authenticate `huggingface-cli login`.
For **corporate use** the governing terms are the **Llama 4 Community License Agreement** (not
Apache-2.0): it permits commercial use below **700 million monthly active users**, requires
attribution ("Built with Llama"), carries an acceptable-use policy, and — importantly for a
detector — the licence grant terminates if you bring a patent/ip claim over the models. Review with
legal before shipping. No unofficial mirror was used and no credential was requested.

### Input format (a first-class result)

| | base v2 | PIGuard |
| --- | --- | --- |
| Tokenizer | DeBERTa-v2 SentencePiece (`tokenizer.json`) | same family |
| Prompt template | **none** — raw text is the input | **none** — raw text |
| Max window | **512** tokens, right-truncation | **512** tokens, right-truncation |
| Head | 2-class softmax | 2-class softmax |
| Labels (`id2label`) | `{0: SAFE, 1: INJECTION}` | `{0: benign, 1: injection}` |
| Positive class index | 1 | 1 |
| ONNX I/O | `input_ids`+`attention_mask` (int64, dynamic) → `logits[B,2]` | same |

The base card is explicit that it is **English-only, does not detect jailbreaks, and false-positives
on system prompts** — a real limitation, and visible below (it false-blocks the Spanish benign cases).
PIGuard ships tokenizer files but **no ONNX**: it was exported to ONNX here from the safetensors
using its own `modeling_piguard.py` custom class (`PIGuard` forward = DeBERTa last-hidden-state
`[:,0,:]` → linear). Exported logits match the PyTorch reference to **max abs diff 9.5e-7**.

## 3. Results on the 146-case corpus (72 attack / 74 benign, EN 78 / ES 68)

Harness: `llm_guardian.bench.harness.run(CASES, scan_callable, repeat=3, warmup=1)`; each model is a
callable returning a `ScanObservation` (block iff `p(injection) ≥ threshold`, else allow). Threshold
**0.5** (the model's own argmax) — not tuned.

| Metric | rules-only (current, **saturated**) | DeBERTa base v2 | PIGuard |
| --- | --- | --- | --- |
| Recall (block) | **1.000** (72/72)† | 0.819 (59/72) | 0.778 (56/72) |
| Precision | 1.000 | 0.567 | 0.667 |
| False blocks / 1,000 benign | **0.0** | 608.1 (45/74) | 378.4 (28/74) |
| Escalation rate | 0.014 | 0.000 | 0.000 |
| p50 latency (ms) | 0.112 | 82.6 | 279.0 |
| p95 latency (ms) | **0.219** | 189.7 | 494.4 |
| p99 latency (ms) | 0.251 | 1202 | 2905 |
| Model load (session + tokenizer) | — | 3.5 s | 5.3 s |
| Truncated cases (512-token window) | — | 4 | 4 |

†The rules were **tuned on these cases**; the models were not. At the pipeline's own effective block
point (the existing severity map emits `high ≥ 0.85`, which alone scores 0.75 = `block_at`), base v2
moves to **0.792 recall / 567.6 false-blocks-per-1000** and PIGuard to **0.653 / 324.3** — the
false-block cost is not a 0.5-threshold artefact; both models are confidently wrong on benign security
text.

### Per-family recall — where the models earn their keep (and where the rules couldn't see)

Blind rules = the first (pre-tuning) run; current rules = 1.000 on every family by construction.

| Attack family | blind rules | DeBERTa base v2 | PIGuard |
| --- | --- | --- | --- |
| instruction_override (paraphrase gap) | 0.375 | **1.000** | **1.000** |
| indirect_injection_repo | 0.333 | **1.000** | **1.000** |
| jailbreak_persona | 0.400 | **1.000** | **1.000** |
| delimiter_role_spoofing | 0.667 | **1.000** | **1.000** |
| obfuscated_payload | 0.857 | 1.000 | 1.000 |
| system_prompt_exfiltration | 0.600 | **1.000** | 0.800 |
| invisible_text_smuggling | 0.600 | 0.800 | 0.600 |
| tool_call_injection | **0.000** | 0.667 | **0.833** |
| markdown_image_exfiltration | **0.000** | 0.750 | 0.750 |
| exfiltration_request | 0.143 | 0.571 | 0.429 |
| agent_tool_misuse | **0.000** | 0.500 | 0.667 |
| credential_phishing | 0.333 | 0.667 | **0.000** |
| long_context_buried | 0.500 | 0.500‡ | 0.500‡ |

‡**Spurious.** All four `long_context_buried` cases exceed 512 tokens and are truncated; the payload
sits at the *end* of the document, past the window. Scoring the benign prefix alone reproduces the
same verdicts — both models flag the ES truncated prefix (`p≈0.99` base, `0.57` PIGuard) and clear
the EN prefix, so the "0.5" is the model reacting to the truncated benign text, **not** to a detected
payload. Neither model can see a payload buried past 512 tokens; long-context detection needs
chunking *above* the scanner, and a truncated scan must never be read as clean.

**What the models actually close:** the paraphrase gap in `instruction_override` (0.375 → 1.000) and
the three families the blind rules scored 0.000 on — `tool_call_injection`, `markdown_image_exfiltration`,
`agent_tool_misuse` — plus `exfiltration_request`. That is real, untuned signal. But the tuned rules
now also reach 1.000 on those families **with zero false blocks and three orders of magnitude less
latency**, so on this corpus the model adds cost, not coverage.

## 4. Overdefense — the design's stated concern (PIGuard's own NotInject set)

NotInject (`leolee99/PIGuard`, `datasets/NotInject_{one,two,three}.json`, downloaded from the upstream
GitHub repo — the HF *dataset* answers 401 and is not public) is **339 benign sentences enriched with
prompt-injection trigger words**; every one is benign, so any `injection` prediction is a false
positive. This is not our corpus and neither model was tuned on it.

| | DeBERTa base v2 | PIGuard |
| --- | --- | --- |
| NotInject false positives | **147/339 = 43.4 %** | **39/339 = 11.5 %** |
| — 1 trigger word | 25/113 | 6/113 |
| — 2 trigger words | 60/113 | 12/113 |
| — 3 trigger words | 62/113 | 21/113 |
| Our `benign_security_explainer` (10 cases) | 8/10 | 8/10 |

PIGuard's over-defense mitigation is the clearest result here: **11.5 % vs 43.4 %** on trigger-word
benign text, and its errors scale far more slowly with trigger-word count. Both still flag 8/10 of
our hardest cases (documents that quote a full attack phrase) — neither is safe to switch on.

## 5. Latency and throughput (CPU)

**Harness, full corpus, `repeat=3`** (pooled; the host is shared with a parallel build):

| | DeBERTa base v2 | PIGuard |
| --- | --- | --- |
| p50 / p95 / p99 / max (ms) | 82.6 / 189.7 / 1202 / 1532 | 279.0 / 494.4 / 2905 / 3081 |

**Controlled, `intra_op_num_threads` pinned, median of 30** — isolates the model from the tokenizer
and from contention:

| Input | base 1 thr | base 4 thr | PIGuard 1 thr | PIGuard 4 thr |
| --- | --- | --- | --- | --- |
| short (12 tok) | 84.9 ms | **30.2 ms** | 85.7 ms | **30.4 ms** |
| medium (80 tok) | 454 ms | 130.7 ms | 452 ms | 130.1 ms |
| long (~480 tok) | 3097 ms | 874 ms | 3079 ms | 851 ms |

The two models are **architecturally identical** (both ~184 M-param DeBERTa), and the controlled
bench confirms it: base ≈ PIGuard at every length and thread count. The 3× gap in the harness table
is an **export artefact** — ProtectAI publishes an optimised ONNX, PIGuard was exported here naïvely
— not a property of the model. Cost is quadratic in tokens (disentangled attention): a 512-token
document is ~29× a 12-token prompt.

**Production throughput on this box (4 cores):** short requests ≈ **33 req/s** (≈ 8/core-s), p95 far
above the 30 ms target; 512-token documents ≈ **1.1 req/s**. To approach the SLO you need ONNX
quantization (int8 dynamic is the obvious next step — re-measure; do not assume), then batching, then
GPU. On a GPU both models are sub-millisecond–to–few-ms and the SLO is not the question; on **CPU,
per the design's own "compare GPU only if measured CPU throughput fails" rule, CPU throughput fails**
for anything but short inputs and even then misses p95.

## 6. Recommendation

**Which model:** **PIGuard** (`leolee99/PIGuard`, MIT, ungated) is the better `ml`-seam candidate —
essentially the same untuned recall as the DeBERTa baseline (0.778 vs 0.819), **38 % fewer false
blocks** on our corpus (378 vs 608) and **~4× fewer** on NotInject (11.5 % vs 43.4 %), and it is the
checkpoint built specifically to reduce overdefense, which is the failure mode that gets a guardian
switched off. It is MIT and needs no account. Its only measured downside is a naïve ONNX export;
re-export with graph optimisations (and quantize) before quoting a latency number.

**How it should ship:** behind the existing `ml` seam, **off by default**
(`GUARDIAN_SCANNERS__ENABLE_ML=false`), **advisory only, shadow/escalate — never the sole reason to
block**. The seam reads the checkpoint's `id2label` (so it works for either model), softmaxes the
logits, truncates at 512, and raises rather than reporting an unscored input as clean. The pipeline's
existing severity map means only a `≥0.85` score blocks on its own, which is the conservative
operating point already used above.

**What this run does *not* justify:** adding the model to the *default* configuration, or removing
any deterministic rule. On this corpus the tuned rules dominate on every axis (recall, false blocks,
latency); the model's value is as a **second, independent vote on paraphrases the rules were not
written for**, and that value must be re-measured on a held-out split the rules were not tuned on
(the design's own requirement) before it is turned on. Reported honestly, this is **a negative result
for "ship a model now" and a positive result for "PIGuard is the one to revisit", with Prompt Guard 2
still unmeasured.**

## 7. What could not be measured

- **Prompt Guard 2 (22M and 86M)** — `gated=manual`; blocked at the licence wall. No numbers exist
  here, only the blocker in §2.
- **ProtectAI DeBERTa v3 small v2** — `gated=auto`; HTTP 401 without an authenticated account.
- **NotInject's HF dataset** — 401; the GitHub subset files were used instead.
- **GPU latency** — no GPU on this host.
- **Quantized / re-exported PIGuard latency** — not run; the 279 ms p50 above is a naïve export.
- **Multi-segment chunking** of >512-token documents — out of scope; noted as the reason long-context
  results are spurious.

## 8. Caveats

- The 146-case corpus is **synthesized in-repo by the same author as the detectors** and is now a
  **saturated regression suite** (rules 1.000). Model recall on it is a small, unflattered signal;
  the false-block figures and NotInject are the load-bearing results.
- Latency was measured on a **shared** host (a parallel build ran concurrently); treat the absolute
  milliseconds as an upper bound and the model-vs-model shape as the finding. The controlled
  pinned-thread table is the fair comparison.
- A threshold of **0.5** was used for every model. No threshold was searched to improve a number.

## 9. Reproduce

```bash
uv venv /home/ubuntu/eval-venvs/hf --python 3.12
uv pip install --python /home/ubuntu/eval-venvs/hf/bin/python onnxruntime transformers onnx \
    pydantic pydantic-settings
# base v2 ONNX:   huggingface-cli download protectai/deberta-v3-base-prompt-injection-v2 \
#                   --include 'onnx/*' --local-dir <dir>
# PIGuard:        download leolee99/PIGuard, then export to ONNX with its modeling_piguard.py
# scripts used here: /home/ubuntu/eval-scratch/{eval_models,percase,notinject_eval,diag,prove_seam}.py
PYTHONPATH=src python /home/ubuntu/eval-scratch/eval_models.py   # writes results/summary.json
```

Gate status on this branch: `ruff format --check` clean, `ruff check` clean, `mypy src/llm_guardian`
clean, `pytest --cov` **376 passed, 1 skipped, 97.2 %** (the skipped test is the real-model test,
which runs green in the eval venv when `GUARDIAN_ML_TEST_MODEL` points at a checkpoint).
