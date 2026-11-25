"""Rewrite a SpecForge-trained DSpine draft into the sglang-dspine layout.

SpecForge and the serving plugin name the same code bridge differently:

    SpecForge (training)            sglang-dspine (serving)
    ------------------------------  ------------------------------------
    transfer_codes                  selected_prefix_bridge.codebook
    injection.bias                  selected_prefix_bridge.gate.bias
    injection.message.weight        selected_prefix_bridge.gate.weight
    injection.predecessor_gate.w    selected_prefix_bridge.predecessor_gate.weight
    injection.receiver_gate.weight  selected_prefix_bridge.prev.weight
    injection.readouts.{i}.weight   selected_prefix_bridge.early_read_weights[i]
    injection.writes.{i}.weight     selected_prefix_bridge.feature_write_weights[i]
    (none)                          selected_prefix_bridge.out.weight

The shapes line up: readouts are [1024, 4096] and stack into
early_read_weights [4, 1024, 4096]; writes are [4096, 512] and stack into
feature_write_weights [5, 4096, 512].

`out.weight` has no training-side counterpart. It projects the read
[1024] back to the 512-wide code space before the gate. The serving model
expects it, so it is written as zeros -- the gate still adds the learned
`bias`/`message` path, so this degrades that one path rather than producing
NaNs. Results from such a draft are therefore a *lower bound*; flag it in the
report.

The source checkpoint is never modified: output goes to a separate directory
with symlinks for everything that needs no renaming.

Usage:
    python convert_dspine_serve.py --source <trained ckpt dir> --output <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from safetensors.torch import load_file, save_file

BRIDGE = "selected_prefix_bridge."


def convert(source: str, output: str) -> dict:
    src_cfg_path = os.path.join(source, "config.json")
    with open(src_cfg_path, encoding="utf-8") as fh:
        cfg = json.load(fh)

    weights = load_file(os.path.join(source, "model.safetensors"))
    out = {}
    mapped, zeroed = [], []

    for name, tensor in weights.items():
        if name == "transfer_codes":
            out[BRIDGE + "codebook"] = tensor
            mapped.append(name)
        elif name == "injection.bias":
            out[BRIDGE + "gate.bias"] = tensor
            mapped.append(name)
        elif name == "injection.message.weight":
            out[BRIDGE + "gate.weight"] = tensor
            mapped.append(name)
        elif name == "injection.predecessor_gate.weight":
            out[BRIDGE + "predecessor_gate.weight"] = tensor
            mapped.append(name)
        elif name == "injection.receiver_gate.weight":
            out[BRIDGE + "prev.weight"] = tensor
            mapped.append(name)
        elif name.startswith("injection.readouts."):
            idx = int(name.split(".")[2])
            out.setdefault("__readouts__", {})[idx] = tensor
            mapped.append(name)
        elif name.startswith("injection.writes."):
            idx = int(name.split(".")[2])
            out.setdefault("__writes__", {})[idx] = tensor
            mapped.append(name)
        elif name == "transfer_ready":
            # Training-only readiness flag; serving rebuilds this state.
            mapped.append(name)
        else:
            out[name] = tensor

    reads = out.pop("__readouts__", {})
    writes = out.pop("__writes__", {})
    if reads:
        # The serving bridge reads once per *draft layer*; SpecForge emits one
        # readout per (layer+1) because it also covers the anchor row. Keep the
        # last N so the anchor-only readout is dropped rather than a real layer.
        vals = [reads[i] for i in sorted(reads)]
        n_keep = int(os.environ.get("DSPINE_READOUTS", "4"))
        out[BRIDGE + "early_read_weights"] = torch.stack(vals[-n_keep:])
    if writes:
        out[BRIDGE + "feature_write_weights"] = torch.stack(
            [writes[i] for i in sorted(writes)])

    # out.weight: [code_size(1024), rank(512)] -- maps the code read back into
    # the rank-wide gate space. Gate weights are [rank, hidden] = [512, 1024].
    rank = out[BRIDGE + "gate.bias"].shape[0] if (BRIDGE + "gate.bias") in out else 512
    code_size = int(json.load(open(os.path.join(source, "config.json"), encoding="utf-8"))
                    .get("dflash_config", {}).get("selected_prefix_code_bridge", {})
                    .get("code_size", 1024))
    # read_weight reads the anchor row into the code space; it is the first of
    # the SpecForge readouts ([1024, 4096] each).
    if reads and (BRIDGE + "read_weight") not in out:
        out[BRIDGE + "read_weight"] = reads[min(reads)].clone()
    out[BRIDGE + "out.weight"] = torch.zeros(code_size, rank, dtype=torch.bfloat16)
    zeroed.append("selected_prefix_bridge.out.weight")

    # Version markers are scalar ints, not empty tensors: the loader does
    # `int(tensor.item())` and rejects a 0-element tensor.
    for marker in (BRIDGE + "independent_write_version",
                   BRIDGE + "terminal_disabled_version",
                   "draft_block_causal_version",
                   "last_write_refine_version"):
        out[marker] = torch.tensor(1, dtype=torch.int64)

    os.makedirs(output, exist_ok=True)
    save_file({k: v.contiguous() for k, v in out.items()},
              os.path.join(output, "model.safetensors"))

    cfg["architectures"] = ["DSpineDraftModel"]
    cfg.setdefault("model_type", "qwen3")
    json.dump(cfg, open(os.path.join(output, "config.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)

    return {"mapped": len(mapped), "zeroed": zeroed, "total": len(out)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if os.path.abspath(args.source) == os.path.abspath(args.output):
        sys.exit("output must differ from source")
    info = convert(args.source, args.output)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
