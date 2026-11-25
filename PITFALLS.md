# Pitfalls

Every problem hit while building this, what caused it, and how to avoid
repeating it. Ordered roughly by how long each one cost.

---

## 1. Tau measured without a chat template — results halved

**Symptom.** A draft that clearly worked scored tau ≈ 1.0–2.0. The reference
pretrained model scored 2.03 where the model card implied ~5.

**Cause.** `3_bench/measure_tau.py` sent `system + "\n\n" + user` as raw text.
SGLang's `/generate` tokenises text as-is; it does **not** apply a chat
template. The draft had been trained on
`<|im_start|>system ... <|im_end|><|im_start|>user ... <|im_end|><|im_start|>assistant`
plus Qwen3's empty thinking block, so raw text put it far out of distribution.
Qwen3 also needs `enable_thinking=False` to match training.

**Fix.** Always render through
`tokenizer.apply_chat_template(convs[:-1], add_generation_prompt=True, enable_thinking=False)`.
Measured effect: reference model 2.03 → **4.51**.

**Rule.** Never benchmark a draft on prompts that were not rendered the same
way the training data was. `3_bench/measure_tau.py` now does this by default;
`--raw-text` reproduces the broken path for diagnosis only.

---

## 2. A stage that "succeeded" while measuring nothing

**Symptom.** The pipeline reported `bench: OK` for eight models in under two
minutes each, and the report showed `tau=None` everywhere.

**Cause.** Two layers of swallowed failure:

* `3_bench/bench_one.py` wrote `{"status": "OK", "sets": {...}}` where every set had
  `"status": "FAIL"` — the outer status did not reflect the inner ones.
* The queue script piped its success check through `tee`, which masks the exit
  code, so a failed check still counted as `DONE`.

**Rule.** A stage's status must be derived from its actual payload, never
assumed. When you pipe a check through `tee`, `$?` is tee's status.

---

## 3. Orchestrator looped 40 times without doing work

**Symptom.** The unattended loop (`run_all.sh`, since removed) reached its
`MAX_PASSES` limit while the same three runs kept failing.

**Cause.** The training stage set `progress=1` because it *attempted*
training, not because a checkpoint appeared. A permanently failing config
therefore looked like progress forever.

**Fix.** Only count progress when the artefact actually exists
(`is_trained <name>` after the attempt).

---

## 4. `block_size: null` silently trained at the wrong size

**Symptom.** dspark-warmstart was trained, exported, and measured — all
"successfully" — but scored tau 1.34 instead of ~5.

**Cause.** `block_size` was only inside `dflash_config`, not at the top level.
The model builder reads `config.block_size` first and falls back to the
`dflash_config` value for *dflash*, but dspark's path defaulted to 7 without
error. Training ran at gamma 7 while serving used 16.

**How it was caught.** Initial training accuracy: the bad run started at 0.036,
a correct one at 0.610. Comparing step-10 accuracy against a known-good run is
the fastest sanity check.

**Rule.** After any config surgery, verify the *effective* value, not the file:
check the draft config's top-level `block_size`, the `dflash_config.block_size`,
and (for dspark) that `num_draft_tokens == block_size + 1`.

---

## 5. Serving flags overridden by a stale harness table

**Symptom.** `ValueError: DSpark speculative_num_draft_tokens must equal
gamma + 1 (= 8 for gamma=7), but got 16`.

**Cause.** `3_bench/bench_one.py` hardcoded `--speculative-dspark-block-size 7`.
The suite manifest said 16. Hardcoded values in the harness silently beat the
manifest.

**Fix.** dspark now passes **no** block-size flag; gamma comes from the draft
config, which is the only thing that can be right for both b7 and b16 drafts.

**Rule.** Parameters that describe the model belong in the model's config, not
in the harness. If a harness must know one, read it from the artefact.

---

## 6. Argument-list bug: `unrecognized arguments: 16`

**Symptom.** Every SGLang server died instantly with
`sglang serve: error: unrecognized arguments: 16`.

**Cause.** Two independent bugs stacked:

1. In bash, `${var:+--flag "$var"}` unquoted word-splits, turning
   `--draft-tokens 16` into three tokens with `16` left dangling.
2. In Python, `[a for a in extra if not a.startswith("--speculative-num-draft-tokens")]`
   dropped the flag but kept its **value**, leaving a bare `16` in the list.

**Fix.** Build optional flags as bash arrays (`EXTRA+=(--flag "$val")`), and
when filtering a flag/value pair, skip the value too.

**Rule.** Filtering `--flag value` pairs needs pair-aware logic. A string
filter on the flag alone leaves an orphan value.

---

## 7. Python loads only ONE `sitecustomize`

**Symptom.** After adding a second shim directory to `PYTHONPATH`, the
CUDA-13 workarounds stopped applying and the server died on
`TreeSpeculativeSamplingTargetOnly ... CUDA driver version is insufficient`.

**Cause.** `sitecustomize` is imported once, from the first directory on
`PYTHONPATH` that has one. The new shim shadowed `sglang_compat` entirely.

**Fix.** Each shim that needs another chains to it explicitly:

```python
_shim = "<abs path>/sglang_compat/sitecustomize.py"
exec(compile(open(_shim).read(), _shim, "exec"), {"__name__": ...})
```

**Rule.** Only the first `sitecustomize` on `PYTHONPATH` runs. Chain, never
assume both load.

---

## 8. `torchcodec` raises `OSError`, which SGLang does not catch

**Symptom.** `OSError: Could not load this library:
.../torchcodec/libtorchcodec_image.so` killed the server during model loading.

**Cause.** SGLang guards the optional import with
`except (ImportError, RuntimeError)`, but this build raises `OSError`.

**Two wrong fixes, then the right one.**

1. *Blocking the module* (`raise ModuleNotFoundError` from a finder) breaks
   `transformers`, whose audio path imports torchcodec eagerly and then loses
   `AutoProcessor`.
2. *A bare `types.ModuleType` stub* breaks `importlib.util.find_spec`, which
   raises `ValueError: torchcodec.__spec__ is None`.

**Working fix.** Register stub modules **with a real spec**:

```python
pkg.__spec__ = importlib.machinery.ModuleSpec("torchcodec", loader=None)
decoders.__spec__ = importlib.machinery.ModuleSpec("torchcodec.decoders", loader=None)
sys.modules["torchcodec"] = pkg
sys.modules["torchcodec.decoders"] = decoders
```

Imports succeed, `find_spec` works, and nothing tries to decode audio.

**Rule.** When stubbing an optional dependency, give it a `ModuleSpec`. A
hand-made module without `__spec__` breaks every availability probe.

---

## 9. Shims must load *before* the library that needs them

**Symptom.** Regeneration crashed with the torchcodec `OSError` even though
the shim function was present.

**Cause.** `from vllm import LLM` sat above `_block_torchcodec()`.

**Rule.** Any import-time workaround must run before the first import of the
library it protects. In `1_data/build_all8.py` that means shim → cu13 → vllm.

---

## 10. vLLM aborts the whole batch on one over-long prompt

**Symptom.** Regeneration stopped at 1000 / 30212 rows with
`VLLMValidationError: ... prompt contains at least 8193 input tokens`.

**Cause.** One prompt exceeded `max_model_len`. vLLM raises rather than
skipping it.

**Fix.** Filter by token count before `generate()`:
`len(tok.encode(text)) <= max_model_len - max_new_tokens`.

**Consequence to be aware of.** Filtering *drops* those rows. data4 averages
35k characters per prompt, so it lost 3015 of 3106 rows and ended up with 77
training samples. If a source needs long prompts, raise `--max-model-len` or
truncate instead of skipping.

---

## 11. Warm start silently not applied

**Symptom.** A "warm start" run had the same step-10 accuracy as its scratch
counterpart (0.036 vs 0.040), i.e. the pretrained weights did nothing.

**Cause.** `draft_checkpoint_path` pointed at a directory whose config listed
an architecture the trainer accepts, but whose **tensor names** were the
serving layout (`selected_prefix_bridge.*`) instead of the training layout
(`injection.*`). Loading found nothing to map and the model stayed at init.

**How to catch it.** Compare step-10 accuracy between the warm-start and
scratch runs. They must differ substantially. As a second check, compare final
weights:

```python
# warm start should be much closer to the pretrained weights than to scratch
mean_abs_diff(warm, pretrained)  # ~0.004
mean_abs_diff(scratch, pretrained)  # ~0.18
```

**Rule.** "It trained without error" does not mean the weights loaded.
Always verify the initial accuracy differs from scratch.

---

## 12. Export used the wrong config, silently

**Symptom.** A run trained at block 16 exported a checkpoint with
`block_size: 8`.

**Cause.** `stage_export` derived the draft config from the algorithm name
(`configs/qwen3-8b-{algo}.json`) instead of using the config the run was
actually trained with (`configs/qwen3-8b-{algo}-b16.json`). Worse, the
generated run YAMLs under `artifacts/` and the checked-in ones under
`configs/runs/` had drifted apart, so the export read the stale copy.

**Fix.** Export reads `model.draft_model_config` from the run's own YAML; for
warm-start runs (which have no such key) it uses the init checkpoint's
`config.json`.

**Rule.** Never re-derive a training parameter at export time. Carry it in the
run's config, or read it from the checkpoint. And when generated files are
copied into a checked-in location, they **will** drift — copy, don't hand-edit
both.

---

## 13. Architecture strings are framework-specific

Checkpoints arrive in one framework's naming and must be renamed for another.
All of these were hit:

| checkpoint | said | must say | because |
| --- | --- | --- | --- |
| `Qwen3-8B-Domino-b16` | `DFlashDraftModel` | `DominoDraftModel` | SpecForge validates the strategy's architecture |
| `Qwen3-8B-DSpine-b16` | `DFlashDraftModel` | `DSpineDraftModel` (train) / `DFlashDraftModel` (serve) | training and serving want different names |
| `dspark_qwen3_8b_block7` | `Qwen3DSparkModel` | `DSparkDraftModel` | `Qwen3DSparkModel` is not registered |
| `Qwen3-8B-speculator.peagle` | `PEagleDraftModel` | `LlamaForCausalLMEagle3` | SGLang hard-requires this exact string (`weight_utils.py:467`) |

**Rule.** One checkpoint may need *two* configs — one for training, one for
serving. Keep them in separate directories with symlinked weights rather than
flip-flopping a single file.

---

## 14. Serving has hard validation the trainer does not

Config keys that training ignores but serving requires at the **top level**
(not inside `dflash_config`):

| key | who needs it |
| --- | --- |
| `markov_rank` | dspark: `ValueError: DSPARK requires markov_rank > 0` |
| `block_size` | dspark: gamma derivation |
| `target_layer_ids` | dspine / dflash aux-layer resolution |
| `max_position_embeddings` | context-length derivation |
| `target_hidden_size` | EAGLE3 FC input width (`NoneType * int`) |
| `num_hidden_layers` | counted from weights; wrong value re-inits layers |

Also `peagle` needs `eagle_config.eagle_aux_hidden_state_layer_ids` as a nested
dict — SGLang reads `config.eagle_config["..."]`, not the flat key.

**Rule.** After exporting, diff the exported `config.json` against a known-good
model of the same architecture before spending time on a benchmark run.

---

## 15. `spec_accepted_tokens` / `spec_drafted_tokens` are always 0

In this SGLang build those response fields are never populated. Only
`spec_accept_length` and `spec_accept_rate` are meaningful.

**Rule.** Trust tau and accept rate; ignore the token counters.

---

## 16. tau is not comparable across block sizes

| strategy | draft tokens per step |
| --- | --- |
| dflash / domino / dspine | 16 |
| dflash2 / dspark | 8 (upstream weights) |
| eagle3 / peagle | 4 (1×1 tree) or 16 (4×2 tree) |

A tau of 8 with a 16-token budget is *worse* than a tau of 5 with an 8-token
budget. Two normalisations help:

* **accept rate** — fraction of drafted tokens accepted, already in the report
* **(tau - 1) / (block - 1)** — expected accepted fraction of the budget

**Rule.** Compare tau only within the same block size. Across algorithms, use
accept rate.

---

## 17. Training-side accuracy overstates serving tau by ~3.5×

One run reported `unary_greedy_accepted_length` 11.4 while the served tau was
3.22. Training metrics are teacher-forced per-position estimates; serving is a
chain of self-generated tokens with exposure bias.

**Rule.** Judge drafts by served tau only. Training accuracy is useful for
spotting non-convergence and failed warm starts, nothing else.

---

## 18. Fine-tuning can *destroy* a strong draft

Measured, with weights verified as loaded:

| algorithm | pretrained (OOD) | fine-tuned on data1 (OOD) | change |
| --- | --- | --- | --- |
| dflash | 3.53 | 4.61 | +31% |
| dspark | 4.84 | 5.04 | +4% |
| domino | 7.39 | 7.05 | −5% |
| dflash2 | 4.07 | 4.04 | −1% |
| **dspine** | **8.18** | **5.68** | **−31%** |
| **eagle3** | 2.29 | 1.18 | **−48%** |

DSpine lost **both** in-domain (5.33 → 2.82) and out-of-domain accuracy. Its
weights changed by only 0.004 mean-absolute, so it is genuine catastrophic
forgetting, not a loading bug.

**Rule.** More pretrained prior ≠ better fine-tuning. Benchmark the
pretrained, fine-tuned and scratch variants; do not assume fine-tuning helps.

---

## 19. In-domain numbers hide which task a draft serves

The 8 generic benchmarks (gsm8k, mbpp, …) are a *different distribution* from
the business data. All-8-source training looked worse on them (−11%) but was
better on 7 of 8 business sources (mean 3.14 → 4.62).

**Rule.** Benchmark on the distribution you will serve. Keep the generic sets
as a secondary signal.

---

## 20. Environment notes worth not rediscovering

* `git` is not installed on this box. `codeload.github.com/<repo>/tar.gz/refs/heads/main`
  works where `raw.githubusercontent.com` and the API are blocked.
* `pip download` works; the HF CLI works through `HF_ENDPOINT=https://hf-mirror.com`
  and is ~10× faster than `curl` from the mirror. Plain `curl` gets throttled
  to ~16 KB/s on large files.
* `setsid nohup ... &` frequently loses the child in this environment. Use the
  tool's own background execution or `timeout` in the foreground.
* Killing a training run leaks `mooncake_master`, `capture_server` and
  `EngineCore`; the leaked processes hold GPU memory and make the *next* vLLM
  start fail with "free memory less than desired". Check `nvidia-smi` reaches
  0 MiB before relaunching.
* A failed training attempt leaves `control/` and `consumer-state/` in the run
  directory and SpecForge refuses to reuse them
  (`has artifacts from an existing attempt`). Delete the run directory before
  retrying.
