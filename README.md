# specforge-experiments

> 中文版:[README_zh.md](README_zh.md) · 踩坑记录:[PITFALLS.md](PITFALLS.md)

Tooling built around [SpecForge](https://github.com/sgl-project/SpecForge):
build a dataset from raw request dumps, train speculative-decoding drafts, and
measure their accepted length on real traffic.

**This repo does not vendor SpecForge.** It drives a SpecForge checkout, which
you install separately; point `SF_SPECFORGE` at it (see Setup). That keeps this
toolkit independent of the framework's internals and free to track upstream.

---

## Layout

```
specforge-experiments/
├── paths.py               every filesystem location (the only place one is written)
├── local.env.example      copy to local.env to override paths.py
├── run.sh                 entry point: build-data | train | export | bench | report
│
├── 1_data/                raw dumps -> train/test JSONL
│   ├── conversations.py       parsing helpers (the four raw schemas)
│   ├── build_all8.py          the pipeline: flatten -> regenerate -> window
│   └── convert_speculators_dflash.py
│
├── 2_train/               configs -> trained checkpoints
│   ├── gen_configs.py         algorithm table -> run YAMLs
│   ├── run_pipeline.py        five-stage driver
│   └── run_online.sh          capture server + `specforge train`
│
├── 3_bench/               checkpoints -> accepted length (tau)
│   ├── bench_one.py           one draft, all eval sets
│   ├── bench_dspine.py        DSpine (separate venv)
│   ├── measure_tau.py         the measurement client
│   ├── load_prompts.py        generic sets + the 8 business sources
│   ├── run_bench_queue.sh     sequential runner with timeouts
│   └── convert_dspine_serve.py
│
├── serving/
│   └── capture_server.py      stand-in for SpecForge's SGLang capture plugin
│
└── shims/                 environment workarounds (PYTHONPATH)
    ├── sglang_compat/     CUDA-13 ops on a 12.9 driver
    ├── no_torchcodec/     broken torchcodec build
    └── cudnn_off/         Domino's GRU head
```

Documents:

| file | contents |
| --- | --- |
| `README.md` / `README_zh.md` | this file -- the tooling |
| `PITFALLS.md` / `PITFALLS_zh.md` | 20 problems hit, and how to avoid them |
| `RESULTS.md` / `RESULTS_zh.md` | environment, data prep, training setup, measured numbers |
| `docker/` | Dockerfile reproducing the base image + `ENVIRONMENT.md` / `ENVIRONMENT_zh.md` |

The three numbered directories are the three stages, in order. Read them in
that order; nothing in `2_train` is needed to understand `1_data`.

Run configs are in `configs/runs/`; the suite manifest is `configs/suite.yaml`;
the Qwen3-8B draft configs are in `configs/`.
Outputs go to `artifacts/`, which is git-ignored.

---

## Setup

No file here hardcodes a path. Every location comes from `paths.py`:

1. an environment variable (table at the top of `paths.py`)
2. `local.env` if present (git-ignored)
3. a default derived from the repo location

The defaults assume:

```
<root>/
├── SpecForge/               framework this drives ($SF_SPECFORGE)
├── specforge-experiments/   this repo ($SF_TOOLKIT)
├── model/Qwen3-8B/          target model
├── data/specforge/          raw dumps + processed datasets
├── venv-specforge/          python with specforge installed
├── venv-sglang/             python with SGLang
└── venv-dspine/             python with the sglang-dspine plugin
```

**First, get SpecForge.** Clone the framework and install it into an
interpreter; that interpreter is `SF_PY_TRAIN`.

```bash
git clone https://github.com/sgl-project/SpecForge
# ... create venv-specforge with SpecForge installed (see docker/ENVIRONMENT.md)
```

Then, for a layout other than the one above:

```bash
cp local.env.example local.env     # set SF_SPECFORGE / SF_ROOT at minimum
$SF_PY_TRAIN -c 'import specforge' # the trainer must be importable
python paths.py                    # every resolved location + ok/MISSING
```

Two roots matter. `SF_TOOLKIT` is this repo; `SF_SPECFORGE` is the framework
checkout. `specforge train` runs *inside* the framework checkout, so artifact
paths and `configs/runs/*.yaml` resolve against it — that is why outputs land
in `$SF_SPECFORGE/artifacts/`.

**Using your own models and data:** two edits —

* `2_train/gen_configs.py` → `WARM_DRAFTS` (one pretrained checkpoint per
  algorithm) and `ALGORITHMS` (draft config, batch size, aux layers).
* `configs/suite.yaml` → `runs` and `eval_sets`.

---

## Usage

```bash
./run.sh build-data          # raw dumps -> train/test JSONL
./run.sh train all           # every run in the suite
./run.sh export all          # trained -> servable HF
./run.sh bench all           # serve + measure tau
./run.sh report              # markdown + JSON summary

./run.sh pipeline            # the five stages in order
./run.sh bench-dspine <dir>  # DSpine needs its own venv
```

`run.sh` picks the interpreter per stage: `venv-specforge` for training,
`venv-sglang` for serving, `venv-dspine` for DSpine. That split is required —
see `PITFALLS.md` §20.

A single stage can also be run directly:

```bash
python 1_data/build_all8.py --stage flatten
python 2_train/run_pipeline.py --stages report
./3_bench/run_bench_queue.sh bench-dspark-all8
```

---

## 1. Data — `1_data/`

### `conversations.py`

Parsing helpers, no pipeline logic. The raw dumps use four schemas:

| schema | example sources |
| --- | --- |
| `messages` + `choices` | data3–data8 |
| `model_prompt_tmpl` + `model_prompt_placeholder` + `choices` | data1, data2 |
| multi-turn agent traces with `tool_calls` | data8 |
| `messages` only, no target | data7 |

`to_turns()` normalises all four into `[{role, content}]`, appending the
`choices[0]` answer as the final assistant turn. Note that data1/data2 share
**one** `model_prompt_tmpl` with `{{context}}`; the per-row text lives in
`model_prompt_placeholder`, and the `messages[].content` copy is the
*unsubstituted* template — use it and every prompt becomes identical.

`flatten_to_single_turn()` emits one sample per assistant turn, with the
earlier turns as context, so a multi-turn trace becomes several training rows.
`apply_window()` enforces the length rules. `sample_key()` is the dedup
identity: `(last user turn, answer)`.

### `build_all8.py`

Three resumable stages:

```bash
python 1_data/build_all8.py --stage flatten
python 1_data/build_all8.py --stage regenerate
python 1_data/build_all8.py --stage window
```

| stage | what it does | why |
| --- | --- | --- |
| `flatten` | one sample per assistant turn; cross-source dedup; per-source cap | keeps each business domain represented without letting one dump dominate |
| `regenerate` | re-answer with Qwen3-8B, `enable_thinking=False`, `temperature=0` | the draft must imitate the target's *current* distribution, not an older system's output |
| `window` | drop answers > 1024 tokens; right-truncate the rendered conversation to 2048 | matches the window the draft sees when serving |

`regenerate` records `finish_reason`; an answer cut at `max_new_tokens` is an
answer **larger** than 1024 that we could not see, and gets dropped. Prompts
that cannot fit the context are skipped *before* generation — vLLM aborts the
whole batch on one over-long input.

Output: `data/specforge/v5-all8/{train,test}.jsonl`.

**data4 caveat.** Its prompts average 35k characters; the skip rule removes
3015 of 3106 rows, leaving 77 training samples. Raise `--max-model-len` or
truncate instead if data4 matters.

### `convert_speculators_dflash.py`

Rewrites a `speculators`-native DFlash checkpoint into SpecForge's layout:
drops speculators-only config keys, lifts `aux_hidden_state_layer_ids` into
`dflash_config.target_layer_ids`, removes `embed_tokens` / `lm_head` (a
SpecForge draft borrows the target's). The source is never modified.

---

## 2. Training — `2_train/`

### `gen_configs.py`

Generates one run YAML per (algorithm, initialisation). Per-algorithm
differences live in two tables:

| algorithm | batch | vocab mapping | aux layers |
| --- | --- | --- | --- |
| dflash | 2 | no | 1, 9, 17, 25, 33 |
| dflash2 | 2 | no | 2, 10, 18, 26, 34 |
| domino | 2 | no | 1, 9, 17, 25, 33 |
| dspark | 2 | no | 1, 9, 17, 25, 33 |
| dspine | 2 | no | 1, 9, 17, 25, 33 |
| eagle3 | 2 | **yes** | 1, 17, 32 |
| peagle | **1** | **yes** | 1, 17, 32 |

`peagle` requires `batch_size=1`; eagle3/peagle need a vocab mapping because in
disaggregated mode the producer and consumer cannot derive a shared one. Both
are validated at config-load time, so a mistake fails loudly.

`WARM_DRAFTS` maps each algorithm to its pretrained checkpoint. **Edit this to
use your own weights** — the architecture has to match the algorithm.

### `run_pipeline.py`

Five stages, each writing a `.done` marker so a re-run resumes:

| stage | action |
| --- | --- |
| `configs` | regenerate run YAMLs from `gen_configs.py` |
| `train` | `run_online.sh <yaml>` for every run without a checkpoint |
| `export` | `specforge export --to hf` into `artifacts/checkpoints/<run>` |
| `bench` | serve + measure (delegates to `3_bench/`) |
| `report` | markdown + JSON summary |

`configs/suite.yaml` says **what** runs; the driver never invents a
parameter that the manifest or the run YAML already states.

### `run_online.sh <config.yaml>`

Starts `mooncake_master` and the capture server, waits for readiness, runs
`specforge train`, tears down on exit.

SpecForge's official capture server is a patched SGLang that cannot be built
here (Rust 1.92 / CUDA 13 vs this driver's 12.9). `serving/capture_server.py`
reproduces the *contract* — two endpoints, features written to Mooncake keys —
in plain PyTorch. It does **not** reproduce SGLang's radix-cache isolation or
chunked-prefill, so its checkpoints are pipeline-correct but not
bit-comparable with a real SGLang capture run.

### Hyperparameters

Identical for every run so rows are comparable:

| setting | value | rationale |
| --- | --- | --- |
| epochs | 3 | 9 epochs on the same 3000 rows gave only +12% tau |
| learning rate | 2e-4 | measured +11% vs 6e-4 for scratch training |
| batch × accum | 2 × 4 | fits one A800 at length 2048 |
| warmup / grad-norm / seed | 0.04 / 1.0 / 42 | |

---

## 3. Benchmark — `3_bench/`

### `bench_one.py`

Serves one draft behind SGLang and measures tau on every evaluation set with a
single server (loading the target costs minutes; reuse it).

The algorithm → SGLang mapping is **not uniform** and is the most error-prone
part of this repo:

| strategy | SGLang algorithm | notes |
| --- | --- | --- |
| dflash | `DFLASH` | block size from the draft config |
| dflash2 | `DFLASH` | block 8 or 16, from the draft config |
| domino | `DFLASH` | routed by `projector_type: domino`; needs cuDNN off |
| dspark | `DSPARK` | gamma = draft block_size; `num_draft_tokens = gamma + 1` |
| dspine | *(separate)* | `bench_dspine.py` |
| eagle3 | `EAGLE3` | tree = `num-steps` × `eagle-topk` |
| peagle | `EAGLE3` | served through EAGLE3 after a config rewrite |

### `measure_tau.py`

The client. **Renders prompts through the chat template** — SGLang's
`/generate` does not apply one, and raw text roughly halves tau. Reads
`spec_accept_length` and `spec_accept_rate`; the token counters are always 0 in
this build.

### `bench_dspine.py`

DSpine needs the `sglang-dspine` plugin, pinned to sglang 0.5.22 and torch
2.14.1, which cannot coexist with the 0.5.20 venv. This script launches from
`venv-dspine`; only prompt files and result format are shared.

### `run_bench_queue.sh`

Sequential runner with a per-model `timeout`. A broken server hangs on
`/health` while holding the GPU, so each model is timed out and all sglang
processes are killed between models, polling until the card returns to 0 MiB.

### `load_prompts.py`

Two families of evaluation set:

* `datasets/<name>/` — seven generic benchmarks (aime25, math500, gsm8k,
  humaneval, mbpp, lcb, mtbench), five different schemas.
* `src1..src8` — the eight business sources, split out of the merged test file,
  for "which task does this draft serve best". Not all eight are usable: `src4`
  keeps only 3 rows after the length filter, so per-task reporting covers the
  other seven (labelled A–G in `RESULTS.md`).

---

## 4. Serving and shims

`serving/capture_server.py` — the stand-in described above. Contract points:
`store_id` must echo the request; feature names must match the algorithm
(`hidden_state`, `attention_mask`, `loss_mask`, `input_ids`, `target`);
`hidden_states[i+1]` is layer *i*'s output (index 0 is the embedding).

`shims/` are loaded by putting the directory first on `PYTHONPATH`. Python
imports exactly **one** `sitecustomize`, so a shim needing another chains to it
explicitly.

| shim | problem | mechanism |
| --- | --- | --- |
| `sglang_compat` | CUDA-13 ops abort on this 12.9 driver | hides the non-greedy verify op; Triton replacements for EAGLE tree ops under `SGLANG_COMPAT_TRITON_TREE=1` |
| `no_torchcodec` | torchcodec raises `OSError` at import, escaping SGLang's `except (ImportError, RuntimeError)` | stub modules **with a real `ModuleSpec`** |
| `cudnn_off` | Domino's GRU head cannot init cuDNN here | `torch.backends.cudnn.enabled = False`, then chains to `sglang_compat` |

See `PITFALLS.md` §7–§9.

---

## 5. Where things live

```
configs/
├── suite.yaml                 manifest: every run, every eval set
├── runs/bench-<algo>-<init>.yaml   15 run configs (framework input)
└── qwen3-8b-*.json            draft model configs (the trainer's draft_model_config)

configs/qwen3-8b-*.json        draft architectures

artifacts/                     git-ignored, ~205 GB
├── datasets/v4-data1/         single-source dataset
├── checkpoints/<run>/         exported HF drafts (servable)
├── runs/<run>/                raw training output (resume only)
└── pipeline/{configs,logs,reports,prompts}
```

Run configs hold **repo-relative** paths (SpecForge resolves from the current
directory; the launchers `cd` to the repo root), so they work on any machine
with the same relative layout.

---

## 6. Framework version note

This toolkit was built and measured against an intermediate SpecForge
revision that had two defects, both since removed by upstream refactors:

1. `PEagleDraftModel.from_pretrained` did not handle
   `output_loading_info=True` (transformers returns `(model, info)` then), so
   every PEagle warm start crashed with
   `'tuple' object has no attribute '_rebuild_rotary_embedding'`. The class now
   inherits `DraftPreTrainedModel`, which handles it.
2. EAGLE3's `build_server_collator` used `concatenate_features`, which demands
   equal sequence lengths and raised `Sizes of tensors must match` as soon as
   `batch_size > 1`. That collator was deleted; the padding collator is used
   everywhere.

**This repo therefore ships no SpecForge patch and assumes neither of those
internals.** If you run an older SpecForge, verify both; the numbers in
`RESULTS.md` were measured on a revision carrying those two fixes.

---

## 7. Results

Measured results are kept in a separate document,
[RESULTS.md](RESULTS.md) (中文:[RESULTS_zh.md](RESULTS_zh.md)), together with
the environment, the data-preparation recipe and the exact training/testing
setup. This README describes the tooling only.
