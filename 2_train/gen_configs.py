"""Generate one SpecForge run config per (algorithm, initialisation) pair.

Every algorithm differs in ways that are easy to get wrong, so they are encoded
once here rather than repeated across hand-written YAML files:

| algorithm | draft config | batch size | vocab mapping | extra |
| --- | --- | --- | --- | --- |
| dflash | qwen3-8b-dflash.json | 2 | no | 5 aux layers |
| domino | qwen3-8b-domino.json | 2 | no | 5 aux layers, projector_type=domino |
| dspark | qwen3-8b-dspark.json | 2 | no | 5 aux layers, block 7 |
| eagle3 | qwen3-8b-eagle3.json | 2 | **yes** | 3 aux layers |
| peagle | qwen3-8b-peagle.json | **1** | **yes** | 3 aux layers, 4 draft layers |

`peagle` needs batch_size=1 and eagle3/peagle need a vocab mapping in
disaggregated mode (producer and consumer cannot derive a shared mapping).

Two initialisations are supported:

* ``scratch``     - ``draft_model_config`` only; weights are N(0, 0.02) with the
                    embedding copied from the target.
* ``warmstart``   - ``draft_checkpoint_path`` pointing at an exported draft. Only
                    offered where an exported checkpoint with the same layer set
                    exists, because the projection weights are tied to the
                    captured layers.

Usage:
    python gen_configs.py --out-dir <dir> [--algos dflash,domino,...]
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
_sys.path.insert(1, str(_Path(__file__).resolve().parent))
import paths as _p

import argparse
import json
import os
import sys

ROOT = str(_p.ROOT)
REPO = str(_p.REPO)
ART = f"{REPO}/artifacts"

# Emit paths RELATIVE TO THE FRAMEWORK CHECKOUT: SpecForge resolves config paths
# against the current directory, and run_online.sh always runs from the
# SpecForge root. This keeps the committed configs machine-independent.
def _rel(path) -> str:
    import os
    return os.path.relpath(str(path), str(_p.SPECFORGE))


def _draft_config(name: str) -> str:
    """Path to a draft config, relative to the framework checkout.

    All Qwen3-8B draft configs this suite needs ship with the toolkit under
    ``configs/``; fall back to the framework's own ``configs/`` if a name is
    missing there. Relative for the same reason as every other emitted path:
    committed configs must not embed this machine's layout.
    """
    local = _p.DRAFT_CONFIGS / name
    if not local.is_file():
        upstream = _p.SPECFORGE / "configs" / name
        if upstream.is_file():
            local = upstream
    return _rel(local)


TARGET = _rel(_p.TARGET_MODEL)
TRAIN_DATA = _rel(_p.DATASETS / "v4-data1" / "train.jsonl")
VOCAB_MAPPING = _rel(_p.VOCAB_MAPPING)
CACHE_DIR = _rel(_p.CACHE_DIR)
# Per-algorithm knobs. `layers` documents what the draft config asks for; the
# trainer resolves them from the config itself, so this is informational.
ALGORITHMS = {
    "dflash": dict(draft_config="qwen3-8b-dflash.json", batch_size=2,
                   vocab=False, layers=[1, 9, 17, 25, 33]),
    "domino": dict(draft_config="qwen3-8b-domino.json", batch_size=2,
                   vocab=False, layers=[1, 9, 17, 25, 33]),
    "dspark": dict(draft_config="qwen3-8b-dspark-b16.json", batch_size=2,
                   vocab=False, layers=[1, 9, 17, 25, 33]),
    "eagle3": dict(draft_config="qwen3-8b-eagle3.json", batch_size=2,
                   vocab=True, layers=[1, 17, 32]),
    "peagle": dict(draft_config="qwen3-8b-peagle.json", batch_size=1,
                   vocab=True, layers=[1, 17, 32]),
    "dspine": dict(draft_config="qwen3-8b-dspine.json", batch_size=2,
                   vocab=False, layers=[1, 9, 17, 25, 33]),
    # DFlash2 has no strategy name of its own: SpecForge drives it through
    # `strategy: dflash` and picks the DFlash2 class from the draft config's
    # architecture. Its block size is 8 and it carries a candidate selector, so
    # it is a separate row rather than a dflash variant.
    "dflash2": dict(draft_config="qwen3-8b-dflash2-b16.json", batch_size=2,
                    vocab=False, layers=[2, 10, 18, 26, 34], strategy="dflash"),
}

# The pretrained initialisation for each algorithm. These are the checkpoints
# the user nominated; three arrived in a foreign layout and were repaired in
# place (see REPAIR_NOTES below).
WARM_DRAFTS = {
    # Plain full-attention dflash export (layers [2,10,18,26,34]), distinct
    # from the sliding-window *.dflash-sf-sliding used by the older data1 runs.
    # Names are relative to $SF_ROOT/model (see paths.py). Point these at your
    # own pretrained drafts; the algorithm has to match the architecture the
    # checkpoint was trained as.
    "dflash": _rel(_p.ROOT / "model" / "Qwen3-8B-speculator.dflash-sf"),
    "dflash2": _rel(_p.ROOT / "model" / "Qwen3-8B-speculator.dflash2-sf"),
    "domino": _rel(_p.ROOT / "model" / "Qwen3-8B-Domino-b16"),
    "dspark": _rel(_p.ROOT / "model" / "dspark_qwen3_8b_block7-sf"),
    "eagle3": _rel(_p.ROOT / "model" / "Qwen3-8B-eagle3-AngelSlim"),
    "peagle": _rel(_p.ROOT / "model" / "Qwen3-8B-speculator.peagle"),
    "dspine": _rel(_p.ROOT / "model" / "Qwen3-8B-DSpine-b16"),
}

# What had to change so SpecForge would accept each checkpoint:
#
# * Qwen3-8B-Domino-b16   -- `architectures` said DFlashDraftModel; its 62
#   tensors and `projector_type: domino` are identical to a working Domino
#   draft. Renamed to DominoDraftModel (config.json.bak kept).
# * dspark_qwen3_8b_block7 -- `architectures` said Qwen3DSparkModel (not
#   registered), geometry sat at the top level instead of dflash_config, and it
#   shipped embed_tokens/lm_head which a SpecForge draft does not own. Converted
#   into a side-car directory, source untouched.
# * Qwen3-8B-speculator.dflash2 -- speculators-native; rewritten as a
#   DFlash2DraftModel side-car with target_layer_ids lifted out of
#   aux_hidden_state_layer_ids and the speculators-only keys dropped.
REPAIR_NOTES = "see gen_configs.py WARM_DRAFTS comment"


def render(algo: str, init: str, run_name: str) -> str:
    spec = ALGORITHMS[algo]
    out_dir = _rel(_Path(ART) / "runs" / run_name)

    if init == "warmstart":
        model_lines = [
            f"  target_model_path: {TARGET}",
            "  # draft_model_config omitted: the checkpoint carries config.json and",
            "  # the two are mutually exclusive as init sources.",
            f"  draft_checkpoint_path: {WARM_DRAFTS[algo]}",
        ]
    else:
        model_lines = [
            f"  target_model_path: {TARGET}",
            f"  draft_model_config: {_draft_config(spec['draft_config'])}",
        ]

    # eagle3 always needs the vocab mapping (disaggregated producer/consumer
    # cannot derive a shared one). peagle takes it too when its checkpoint does
    # not carry a draft-vocab mapping of its own.
    if spec["vocab"]:
        model_lines.append(f"  vocab_mapping_path: {VOCAB_MAPPING}")

    model_lines += [
        "  target_backend: sglang",
        "  embedding_key: model.embed_tokens.weight",
        "  torch_dtype: bfloat16",
        "  sglang_mem_fraction_static: 0.3",
    ]

    return f"""# Generated by gen_configs.py (specforge-experiments) -- do not edit by hand.
#
# algorithm:       {algo}
# initialisation:  {init}
# draft layers:    {spec['layers']}
# batch size:      {spec['batch_size']}
# vocab mapping:   {'required' if spec['vocab'] else 'not used'}
#
# Regenerate with:
#   ./run.sh pipeline configs     (or: python 2_train/gen_configs.py --out-dir <dir>)

model:
{chr(10).join(model_lines)}

data:
  train_data_path: {TRAIN_DATA}
  max_length: 2048
  chat_template: qwen
  cache_dir: {CACHE_DIR}

training:
  strategy: {spec.get('strategy', algo)}
  num_epochs: 3
  batch_size: {spec['batch_size']}
  accumulation_steps: 4
  learning_rate: 2.0e-4
  warmup_ratio: 0.04
  max_grad_norm: 1.0
  attention_backend: flex_attention
  num_anchors: 512
  loss_decay_gamma: 7.0
  save_interval: 1000
  max_checkpoints: 2
  log_interval: 10
  dist_timeout: 60
  seed: 42

runtime:
  in_flight_high_watermark: 32
  in_flight_low_watermark: 16

tracking:
  report_to: tensorboard

run_id: {run_name}
output_dir: {out_dir}

deployment:
  mode: disaggregated
  trainer:
    nnodes: 1
    nproc_per_node: 1
  disaggregated:
    control_dir: {out_dir}/control
    consumer_state_dir: {out_dir}/consumer-state
    backend: mooncake
    server_urls:
      - http://127.0.0.1:30000
    mooncake_metadata_server: http://127.0.0.1:35880/metadata
    mooncake_master_server_addr: 127.0.0.1:35551
    mooncake_protocol: tcp
    receive_buffers: pageable
    client_buffer_size: 4294967296
    receive_pool_bytes: 1073741824
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=f"{ART}/pipeline/configs")
    ap.add_argument("--algos", default=",".join(ALGORITHMS),
                    help=f"comma separated, from {sorted(ALGORITHMS)}")
    ap.add_argument("--inits", default="scratch,warmstart",
                    help="scratch, warmstart, or both")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    wanted_inits = [i.strip() for i in args.inits.split(",") if i.strip()]
    index = []

    for algo in [a.strip() for a in args.algos.split(",") if a.strip()]:
        if algo not in ALGORITHMS:
            sys.exit(f"unknown algorithm {algo!r}; pick from {sorted(ALGORITHMS)}")
        for init in wanted_inits:
            if init == "warmstart":
                ck_rel = WARM_DRAFTS[algo]
                # WARM_DRAFTS holds repo-relative strings for the YAML; check
                # the real location, not a path relative to our cwd.
                if not (_p.SPECFORGE / ck_rel).is_dir():
                    print(f"[skip] {algo}/warmstart: missing {ck_rel}",
                          file=sys.stderr)
                    continue
            run_name = f"bench-{algo}-{init}"
            path = os.path.join(args.out_dir, f"{run_name}.yaml")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(render(algo, init, run_name))
            index.append({"run_name": run_name, "algorithm": algo,
                          "init": init, "config": path})
            print(f"[gen] {path}", file=sys.stderr)

    with open(os.path.join(args.out_dir, "index.json"), "w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=2)
    print(f"[gen] {len(index)} configs -> {args.out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
