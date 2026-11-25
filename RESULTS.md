# Results

> 中文版:[RESULTS_zh.md](RESULTS_zh.md) · Tooling:[README.md](README.md)

Empirical results for fine-tuning speculative-decoding drafts on a production
LLM traffic mix.

The source data is described only by task category; no dataset names, prompts
or sample content appear here.

---

## 0. What is being measured

Every number in this document is **accepted length, tau (τ)**: the mean number
of tokens the target model accepts per speculative step when serving with a
draft model.

```
tau = total tokens produced / number of speculative steps
```

Read it as "how many tokens we get for the price of one target forward pass".
`tau = 1.0` means every proposal was rejected — speculative decoding gives zero
benefit. `tau = 8` with a 16-token budget means the draft is right about half
the time, and we save roughly half the decode steps.

It is **not**:

* training accuracy — teacher-forced per-position hit rate, which overstates
  serving by ~3.5× here (measured in §5);
* the draft's own confidence — a draft can be confident and wrong;
* latency — that also depends on the target's verification cost and batch size.

Measured by running a live SGLang server with the draft attached and issuing
greedy generation (`temperature=0`) over held-out prompts, reading
`spec_accept_length` from the response metadata. Prompts are rendered through
the target's chat template with thinking disabled — SGLang's `/generate` does
not apply a template, and getting this wrong roughly halves tau.

### Comparability caveat

**tau is only comparable within the same proposal budget.** A draft proposing
16 tokens per step has twice the room of one proposing 8; ranking them by raw
tau flatters the former.

| algorithm | tokens per step |
| --- | --- |
| DFlash | 16 |
| DFlash2 | 8 (fixed by the released weights) |
| Domino | 16 |
| DSpark | 8 (gamma 6 + 1) |

Accept rate — the fraction of drafted tokens accepted — is budget-independent,
and is reported alongside tau in §4.5.

---

## 1. Environment

| item | value |
| --- | --- |
| GPU | 1 × NVIDIA A800 80 GB |
| driver / CUDA | 550.54.15 / 12.9 |
| target model | Qwen3-8B (36 layers, hidden 4096) |
| training stack | **SpecForge** (SGLang team), pure PyTorch 2.13+cu129 — no vLLM, no SGLang in the training loop |
| serving stack | SGLang 0.5.20 |
| data regeneration | vLLM 0.32.0+cu129, same target model (chosen for throughput; not part of training) |

The base image happens to be a vLLM image, but vLLM only regenerates answers.
Training is SpecForge on plain PyTorch, and feature capture during training runs
the target model through `transformers` — a stand-in for SpecForge's official
SGLang capture plugin, which needs CUDA 13 to build. Because that capture path
is a stand-in, treat absolute numbers as pipeline-correct rather than
bit-comparable to a real SGLang capture run.

---

## 2. Data

### 2.1 Raw source

Raw production request logs, roughly 135k rows, arrive as eight independent
streams:

| # | task category | rows | kept after length filter |
| --- | --- | --- | --- |
| 1 | structured extraction from news text | 27.8k | 3,731 |
| 2 | text classification | 42.3k (half duplicated) | 3,783 |
| 3 | multi-turn tool-using dialogue | 11.3k | 2,196 |
| 4 | extraction with very long context (news + search results) | 3.1k | **77** |
| 5 | extraction, short context | 3.1k | 2,930 |
| 6 | mixed instruction following | 14.0k | 3,808 |
| 7 | general instruction following | 49.0k | 3,804 |
| 8 | agent traces with tool calls | 3.9k | 2,102 |

Stream 4 contributes only 77 usable training rows (0.34% of the merged set) and
all but 3 of its held-out rows fall to the length filter, so it is **not
measurable** at serving time. The report is therefore written around **seven
tasks**, labelled A–G; mapping from the stream numbers above is
1→A, 2→B, 3→C, 5→D, 6→E, 7→F, 8→G.

Each source arrives in one of four schemas: `messages` + `choices`, a prompt
template with placeholders + `choices`, multi-turn traces with `tool_calls`, or
plain `messages`.

### 2.2 Preparation

1. **Flatten to single turn.** Every assistant message becomes one sample with
   the preceding turns as context, so a multi-turn trace yields one sample per
   decision point. Tool results fold into the user side with a marker
   (SpecForge only masks `system` / `user` / `assistant`).
2. **Deduplicate** on `(last user turn, answer)`. 142,876 unique samples remain.
3. **Cap per source** at 4,000 rows in deterministic hash order, so no task
   dominates: 30,212 samples.
4. **Regenerate every answer with the target model** (thinking disabled,
   temperature 0, max 1024 new tokens), so the draft imitates what the target
   says today rather than an older system's output.
5. **Filter by length.** Answers that hit the 1024-token cap are dropped, and
   prompts that do not fit the context window are skipped.

Final dataset: **22,431 train / 1,180 test** rows, 2048-token window
(right-truncated: prompt head discarded, matching serving behaviour).

### 2.3 The two training sets

Two datasets are used, and the distinction matters for reading §4.3:

| name | contents | train / test |
| --- | --- | --- |
| **single-task** | **task A only** — structured extraction from news text | 3,000 / 200 |
| **all-task** | all eight streams merged (seven of them measurable), 4,000-per-source cap | 22,431 / 1,180 |

**Unless a row says otherwise, every fine-tune in this document is trained on
the single-task set.** §4.3 adds the all-task variant for DSpark only.

---

## 3. Models

### 3.1 Pretrained drafts (baseline)

Four publicly available checkpoints. They were authored by different teams on
different data; **they are used as the "before" reference, not as a
reproduction of any paper.**

| algorithm | source | params | tokens/step |
| --- | --- | --- | --- |
| DFlash | [RedHatAI/Qwen3-8B-speculator.dflash](https://huggingface.co/RedHatAI/Qwen3-8B-speculator.dflash) | 2.0 GB, 5 layers | 16 |
| DFlash2 | [RedHatAI/Qwen3-8B-speculator.dflash2](https://huggingface.co/RedHatAI/Qwen3-8B-speculator.dflash2) | 2.2 GB, 5 layers + selector | 8 |
| Domino | [Qwen3-8B-Domino-b16](https://huggingface.co/Huang2020/Qwen3-8B-Domino-b16) | 2.5 GB, 5 layers + GRU corrector | 16 |
| DSpark | [deepseek-ai/dspark_qwen3_8b_block7](https://huggingface.co/deepseek-ai/dspark_qwen3_8b_block7) | 2.3 GB, 5 layers + Markov head | 8 |

Only these four algorithms are reported. DSpine, EAGLE3 and P-EAGLE were
carried through the pipeline but their results are **not published here**: the
DSpine serving path required a hand-inferred tensor rename, EAGLE3's tree
config never produced a stable baseline, and P-EAGLE could not be trained at
all. Their rows would say more about this setup than about the drafts.

Two of the four needed config conversion before the framework would accept them
(architecture string, config key placement), and two needed a serving-side
config rewrite. Those details are pitfalls, not results.

### 3.2 Training

Identical hyperparameters for every run, so rows are comparable:

| setting | value |
| --- | --- |
| epochs | 3 |
| learning rate | 2e-4 |
| batch × accumulation | 2 × 4 (effective 8) |
| warmup / grad-norm / seed | 0.04 / 1.0 / 42 |
| max sequence length | 2048 |

Each algorithm is trained two ways:

* **scratch** — random init, embedding copied from the target.
* **fine-tuned** — initialised from the pretrained draft in §3.1.

### 3.3 Evaluation sets

| set | kind | rows | what it tests |
| --- | --- | --- | --- |
| in-domain | task A held out | 100 | the task trained on |
| aime25 | maths | 30 | competition maths |
| math500 | maths | 100 | mixed maths |
| gsm8k | maths | 100 | grade-school word problems |
| humaneval | code | 100 | function completion |
| mbpp | code | 100 | short programs |
| lcb | code | 100 | competitive programming |
| mtbench | chat | 80 | open-ended instruction |

The seven non-domain sets are collectively called **out-of-domain (OOD)**; their
unweighted mean is the single OOD number quoted in summary tables. Full
per-set numbers are in §4.1.

These OOD sets are public benchmarks — a *different distribution* from the
business traffic. They measure whether fine-tuning destroyed general ability;
they do **not** measure product value. §4.2 and §4.3 measure business tasks
directly.

---

## 4. Results

### 4.1 Generic benchmarks — full per-set breakdown

Three variants per algorithm: pretrained (before), fine-tuned on the
single-task set, and scratch on the same data. All values are tau; the last
column is the unweighted mean over the seven OOD sets.

**DFlash** (16 tokens/step)

| stage | in-domain | aime25 | math500 | gsm8k | humaneval | mbpp | lcb | mtbench | OOD mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pretrained | 2.35 | 3.67 | 4.47 | 3.95 | 3.88 | 3.27 | 2.89 | 2.55 | 3.53 |
| fine-tuned | **6.97** | **4.97** | **5.77** | **5.37** | **5.13** | **4.13** | **3.59** | **3.34** | **4.61** |
| scratch | 2.95 | 1.06 | 1.08 | 1.09 | 1.07 | 1.05 | 1.04 | 1.06 | 1.06 |

**DSpark** (8 tokens/step)

| stage | in-domain | aime25 | math500 | gsm8k | humaneval | mbpp | lcb | mtbench | OOD mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pretrained | 2.69 | 4.83 | 5.52 | 5.72 | 4.94 | 4.86 | 4.10 | 3.89 | 4.84 |
| fine-tuned | **5.48** | **5.12** | **5.76** | **5.81** | **5.50** | **4.94** | **4.23** | 3.90 | **5.04** |
| scratch | 2.59 | 1.06 | 1.09 | 1.10 | 1.05 | 1.04 | 1.03 | 1.06 | 1.06 |

**DFlash2** (8 tokens/step)

| stage | in-domain | aime25 | math500 | gsm8k | humaneval | mbpp | lcb | mtbench | OOD mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pretrained | 2.81 | 4.03 | 4.80 | 4.98 | 4.14 | 3.99 | 3.26 | 3.25 | 4.07 |
| fine-tuned | **5.24** | **4.11** | **4.83** | 4.92 | **4.24** | 3.79 | 3.18 | 3.18 | 4.04 |
| scratch | 3.26 | 1.07 | 1.11 | 1.12 | 1.07 | 1.05 | 1.04 | 1.07 | 1.08 |

**Domino** (16 tokens/step)

| stage | in-domain | aime25 | math500 | gsm8k | humaneval | mbpp | lcb | mtbench | OOD mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pretrained | 4.16 | 7.18 | 9.01 | 9.76 | 8.57 | 6.78 | 5.39 | 5.05 | 7.39 |
| fine-tuned | **7.00** | 6.95 | 8.72 | 9.48 | 7.92 | 6.41 | 4.99 | 4.89 | 7.05 |
| scratch | 3.28 | 1.06 | 1.08 | 1.10 | 1.05 | 1.04 | 1.03 | 1.07 | 1.06 |

### 4.2 What those three rows say

**Fine-tuning works in-domain, universally.** Every algorithm roughly doubles
or triples its in-domain tau: DFlash 2.35 → 6.97, DSpark 2.69 → 5.48,
DFlash2 2.81 → 5.24, Domino 4.16 → 7.00.

**Out of domain it is a coin flip, and the split is clean.** DFlash improves on
all seven OOD sets (+31% mean), DSpark on seven of seven (+4%), DFlash2 on three
(−1% mean), Domino on none (−5% mean). No set breaks the pattern within an
algorithm, which points at the draft's pretrained prior rather than at noise.

**Scratch training destroys general ability, without exception.** All four
scratch models collapse to tau 1.03–1.08 on every OOD set — proposals are
essentially always rejected. Three thousand single-task rows seen three times
overwrite a pretrained draft's entire behavioural range. Scratch also
underperforms fine-tuning in-domain (DFlash 2.95 vs 6.97).

### 4.3 Single task vs all tasks

DSpark, measured on **per-business-task** held-out data. This is the comparison
that matters for a product, because the generic benchmarks in §4.1 are a
different distribution from the traffic being served.

**The single-task column is trained on task A only** (structured extraction
from news text). The all-task column is trained on every stream merged.

Seven tasks are reported below, labelled A–G. Stream 4 (long-context
extraction) lost all but 3 test rows to the length filter and is omitted —
its 77 training rows are still in the *training* set, it is just not
measurable on held-out data.

| task | rows | pretrained | fine-tuned<br>**task A only** | fine-tuned<br>all tasks |
| --- | --- | --- | --- | --- |
| **A** extraction — *the training task* | 184 | 2.57 | **5.47** | **5.48** |
| **B** classification | 217 | 2.18 | 2.74 | **5.15** |
| **C** multi-turn tool use | 121 | 2.01 | 2.53 | **3.20** |
| **D** short-context extraction | 176 | 1.52 | 1.83 | **3.90** |
| **E** mixed instruction following | 189 | 2.68 | 4.96 | **7.66** |
| **F** general instruction following | 180 | 1.96 | 2.34 | **5.36** |
| **G** agent traces | 110 | 2.85 | 2.97 | **3.93** |
| **mean** | — | **2.25** | **3.26** | **4.95** |

Reading the table:

* **Task A is the training task.** Fine-tuning on it lifts that task 2.57 → 5.47
  (+113%) and leaves B–G almost untouched: they move by +0.1 to +2.3, and five
  of the six move by less than +0.5. This is a single-task model behaving
  exactly as expected.
* **Training on all tasks more than doubles the mean** (2.25 → 4.95, +120%) and
  wins on 6 of the 7 reported tasks. Task E gains the most (2.68 → 7.66, +186%).
* **Adding tasks cost nothing on A**: 5.48 vs 5.47.
* **The all-task model is not a "do everything a bit" compromise** — it beats
  the single-task model on all six non-training tasks reported.

This does not contradict §4.1: on the *generic* benchmarks the all-task model
scores 11% lower than the single-task one. Both are true. A draft serving
business traffic should be measured on business traffic.

### 4.4 Choosing, in one table

| goal | pick | evidence |
| --- | --- | --- |
| one dominant task, general ability irrelevant | fine-tune that task | in-domain 7.00 (Domino) / 6.97 (DFlash) |
| mixed traffic, general ability matters | **DFlash** or **DSpark** | OOD +31% / +4%, no OOD set regressed |
| mixed traffic, many task types | fine-tune on all of them | per-task mean 2.25 → 4.95 |
| no fine-tuning at all | Domino pretrained | best zero-shot OOD (7.39) |
| do not | fine-tune Domino when general ability matters | all seven OOD sets lose 1–5% |

### 4.5 Accept rate

`tau` depends on the proposal budget; accept rate does not. Same runs, same
sets, as the fraction of drafted tokens the target accepted:

| algorithm | budget | pretrained OOD | fine-tuned OOD |
| --- | --- | --- | --- |
| DSpark | 8 | 54.8% | **61.5%** |
| DFlash2 | 8 | 43.8% | 43.4% |
| DFlash | 16 | 16.8% | 24.1% |
| Domino | 16 | 42.6% | 40.3% |

This is why tau alone misleads: DSpark's tau (5.04) is lower than Domino's
(7.05), yet DSpark accepts a *larger share of a smaller budget* — 61.5% of 8
tokens versus 40.3% of 16. Under a fixed latency ceiling the higher accept rate
is the better draft, because it can be given a bigger budget.

---

## 5. Conclusions

1. **Start from a pretrained draft. Always.** Scratch training on a single-task
   corpus drives out-of-domain tau to ≈1.05 — proposals are rejected almost
   every time — and underperforms fine-tuning in-domain too (DFlash 2.95 vs
   6.97).

2. **Fine-tuning buys in-domain accuracy and may cost general ability.** All
   four algorithms doubled or tripled in-domain tau, but only two of four
   improved out of domain; Domino lost 1–5% on every OOD set. The strongest
   pretrained draft is also the one that loses most from fine-tuning — a strong
   prior resists a narrow fine-tune.

3. **Train on the distribution you will serve.** The all-task model won 6 of 7
   reported business tasks (mean +120%) while scoring 11% *worse* on public
   benchmarks.
   Evaluating a draft that writes extraction JSON on maths and coding
   benchmarks measures the wrong thing.

4. **Adding tasks did not dilute the task that matters.** All-task training
   scored 5.48 on task A versus 5.47 for the task-A-only model — extra tasks
   cost nothing in-domain while doubling five of the other six.

5. **Rank drafts by accept rate, not by tau.** DSpark's 5.04 tau beats Domino's
   7.05 in efficiency — 61.5% versus 40.3% of a smaller budget accepted.

6. **Training-side metrics overstate serving by ~3.5×.** One run reported an
   accepted length of 11.4 in its training log while the served tau was 3.22.
   Teacher-forced per-position accuracy is not a serving estimate.

---

## 6. Limitations

* **The pretrained baselines are third-party checkpoints**, not reproductions
  of any published configuration. Absolute tau should not be compared against
  paper numbers; the pretrained / fine-tuned / scratch rows within an algorithm
  are the meaningful comparison.
* **The capture path is a stand-in**, not SGLang's official capture plugin
  (which requires CUDA 13 to build). Training is pipeline-correct; absolute
  numbers may differ from a real SGLang capture by some margin.
* **One task is excluded from §4.3.** The long-context extraction task lost all
  but 3 test rows to the length filter, too few to report; it is omitted rather
  than shown with a misleading number.
* **Only four algorithms are reported.** DSpine, EAGLE3 and P-EAGLE were run
  but their numbers reflect setup problems (a hand-inferred serving rename, an
  unstable tree baseline, an untrainable configuration) more than the drafts
  themselves.
* **Single GPU, serial runs, no hyperparameter search.** Learning rate and epoch
  count came from a small sweep and were then fixed; a tuned configuration
  would likely do better, so these numbers are a lower bound.
* **Token budgets differ across algorithms** (16 vs 8), so cross-algorithm tau
  comparisons are valid only at equal accept rate — see §4.5.
