"""Convert a vLLM `speculators` DFlash checkpoint into SpecForge's legacy layout.

Why this exists
---------------
`Qwen3-8B-speculator.dflash` was trained with vLLM's `speculators` (0.7.0) and
stores that framework's `config.json` plus 60 tensors. SpecForge's legacy DFlash
decoder validates the config strictly and rejects the speculators-only keys, and
SpecForge's own DFlash draft has no `embed_tokens` / `lm_head` (it borrows the
target's), so the two tensors speculators ships have nowhere to go.

This script produces a *side-car* directory: a rewritten `config.json` and a
weights file containing only the tensors SpecForge owns. The source checkpoint is
never modified.

What changes
------------
1. `config.json`: drop the speculators-specific keys, move the five capture layer
   ids down into `dflash_config.target_layer_ids`, keep the geometry keys
   SpecForge's decoder reads. `aux_hidden_state_layer_ids` in the speculators
   config is `[2,10,18,26,34]` -- one higher than SpecForge's own default
   `[1,9,17,25,33]` -- so the run config for training MUST set the same layers,
   otherwise the `fc` projection mixes features from the wrong depth.
2. `model.safetensors`: drop `embed_tokens.weight` and `lm_head.weight`.

Usage:
    python -I convert_speculators_dflash.py \
        --source /path/to/Qwen3-8B-speculator.dflash \
        --output /path/to/converted
"""

import argparse
import json
import os
import shutil
import sys

# Keys the speculators config carries that SpecForge's legacy decoder rejects.
DROP_KEYS = {
    "aux_hidden_state_layer_ids",
    "mask_token_id",
    "sample_from_anchor",
    "sliding_window_non_causal",
    "speculators_config",
    "speculators_model_type",
    "speculators_version",
    "target_hidden_size",
    # its contents are lifted to the top level, so the nested wrapper itself
    # must not survive (the legacy decoder rejects unknown keys).
    "transformer_layer_config",
    "auto_map",
    "speculators_metadata",
    "speculators_name_or_path",
    "compressed_tensors",
    "quantization_config",
}

# speculators nests the decoder geometry under transformer_layer_config; the
# legacy decoder expects those fields at the top level and validates that they
# are present.
LIFT_FROM_TRANSFORMER_CONFIG = (
    "hidden_size",
    "intermediate_size",
    "num_attention_heads",
    "num_key_value_heads",
    "num_hidden_layers",
    "vocab_size",
    "head_dim",
    "hidden_act",
    "rms_norm_eps",
    "max_position_embeddings",
    "initializer_range",
    "attention_bias",
    "attention_dropout",
    "bos_token_id",
    "eos_token_id",
    "rope_parameters",
    "layer_types",
    "tie_word_embeddings",
    "use_sliding_window",
    "sliding_window",
    "max_window_layers",
    "use_cache",
)

# Tensors SpecForge's DFlash draft does not own: it calls target.lm_head and
# target.model.embed_tokens directly, so these would be unexpected keys.
DROP_TENSOR_PREFIXES = ("embed_tokens.", "lm_head.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--num-target-layers",
        type=int,
        default=None,
        help="target depth, needed because the legacy decoder interprets "
        "target_layer_ids against it (Qwen3-8B = 36)",
    )
    ap.add_argument(
        "--keep-sliding-window",
        action="store_true",
        help="preserve the source's sliding-window attention geometry instead "
        "of rewriting it to SpecForge's full-attention layout",
    )
    args = ap.parse_args()

    src = os.path.abspath(args.source)
    out = os.path.abspath(args.output)
    if src == out:
        raise SystemExit("output must differ from source; the source is never modified")

    with open(os.path.join(src, "config.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)

    layers = cfg.get("aux_hidden_state_layer_ids")
    if not layers:
        raise SystemExit("source config has no aux_hidden_state_layer_ids")
    layers = [int(x) for x in layers]

    new_cfg = {k: v for k, v in cfg.items() if k not in DROP_KEYS}

    # Lift the nested decoder geometry to the top level where SpecForge reads it.
    inner = cfg.get("transformer_layer_config") or {}
    lifted = {}
    for key in LIFT_FROM_TRANSFORMER_CONFIG:
        if key in inner and inner[key] is not None:
            lifted[key] = inner[key]
    new_cfg.update(lifted)
    if "rope_parameters" in new_cfg:
        # SpecForge's decoder reads rope_theta directly.
        new_cfg.setdefault("rope_theta", (new_cfg["rope_parameters"] or {}).get("rope_theta"))

    # speculators trains the draft with sliding-window attention on every layer
    # (sliding_window=2048, layer_types=sliding_attention), while SpecForge's
    # DFlash draft is full-attention only. The tensor set is identical either
    # way, but the attention mask during training differs, so a checkpoint
    # trained with a 2048-token window cannot be presented to SpecForge as a
    # full-attention model without changing what the weights mean.
    src_sliding = bool(cfg.get("use_sliding_window")) or "sliding_attention" in (
        (cfg.get("transformer_layer_config") or {}).get("layer_types") or []
    )
    print(
        f"[convert] WARNING: source uses sliding-window attention "
        f"(window={new_cfg.get('sliding_window')}); SpecForge's DFlash draft is "
        f"full-attention. Rewriting layer_types to full_attention changes the "
        f"attention mask the weights were trained under; results may degrade. "
        f"Pass --keep-sliding-window to preserve the source geometry instead.",
        file=sys.stderr,
    ) if src_sliding and not args.keep_sliding_window else None
    if not args.keep_sliding_window:
        new_cfg["layer_types"] = ["full_attention"] * int(new_cfg["num_hidden_layers"])
        new_cfg["use_sliding_window"] = False
        new_cfg["sliding_window"] = None
        new_cfg.pop("max_window_layers", None)

    # The legacy decoder requires the target's depth to interpret target_layer_ids.
    if new_cfg.get("num_target_layers") is None:
        tgt_layers = args.num_target_layers
        if tgt_layers is None:
            raise SystemExit(
                "source config has no num_target_layers; pass --num-target-layers "
                "(Qwen3-8B = 36)"
            )
        new_cfg["num_target_layers"] = int(tgt_layers)
        print(f"[convert] num_target_layers -> {tgt_layers}", file=sys.stderr)

    # SpecForge reads the capture layers from dflash_config.target_layer_ids.
    dflash_cfg = dict(new_cfg.get("dflash_config") or {})
    dflash_cfg["target_layer_ids"] = layers
    dflash_cfg.setdefault("mask_token_id", cfg.get("mask_token_id"))
    new_cfg["dflash_config"] = dflash_cfg
    new_cfg["architectures"] = ["DFlashDraftModel"]

    print(f"[convert] capture layers -> {layers}", file=sys.stderr)
    print(f"[convert] dropped config keys: {sorted(set(cfg) & DROP_KEYS)}", file=sys.stderr)

    from safetensors import safe_open
    from safetensors.torch import save_file

    src_weights = os.path.join(src, "model.safetensors")
    kept = {}
    dropped = []
    with safe_open(src_weights, "pt") as fh:
        for key in fh.keys():
            if key.startswith(DROP_TENSOR_PREFIXES):
                dropped.append(key)
                continue
            kept[key] = fh.get_tensor(key)

    print(f"[convert] tensors kept={len(kept)} dropped={len(dropped)} {dropped}", file=sys.stderr)

    # Sanity: every remaining tensor must be one SpecForge's DFlash draft owns.
    allowed_prefixes = ("layers.", "fc.", "hidden_norm.", "norm.")
    stray = [k for k in kept if not k.startswith(allowed_prefixes)]
    if stray:
        raise SystemExit(f"unexpected tensors remain: {stray}")

    if args.dry_run:
        print("[convert] dry run, nothing written", file=sys.stderr)
        return

    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(new_cfg, fh, indent=2, ensure_ascii=False)
    save_file(kept, os.path.join(out, "model.safetensors"))
    for extra in ("config.py",):
        p = os.path.join(src, extra)
        if os.path.isfile(p):
            shutil.copy2(p, os.path.join(out, extra))
    print(f"[convert] wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
