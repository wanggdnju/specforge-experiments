"""Build one SpecForge dataset from all eight raw dump directories.

Same recipe as data1_pipeline.py, applied across data1..data8 and merged:

1. flatten every assistant turn into its own single-turn sample
2. regenerate the answer with Qwen3-8B (thinking disabled, temperature 0)
3. drop answers over 1024 tokens
4. right-truncate the rendered conversation to a 2048-token window

Differences from the data1-only pipeline:

* **More sources, so a per-source cap.** 133k raw rows would take many hours
  to regenerate. Each source contributes at most `--per-source` samples
  (default 4000), taken in a deterministic hash order so the subset is stable
  across runs. Pass `--per-source 0` for no cap.
* **Cross-source dedup.** data2 ships the same rows twice (raw + _formatted),
  and several sources overlap; identical (last user turn, answer) pairs are
  collapsed before regeneration, earlier sources winning the tie.

Stages are separate so the slow one (regenerate) can be rerun alone:

    python build_all8.py --stage flatten
    python build_all8.py --stage regenerate
    python build_all8.py --stage window
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
_sys.path.insert(1, str(_Path(__file__).resolve().parent))
import paths as _p

import argparse
import glob
import hashlib
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import conversations as bd  # noqa: E402

RAW = str(_p.RAW_DIR)
OUT = str(_p.DATASET_DIR / "v5-all8")
TOKENIZER = str(_p.TOKENIZER)
MAX_WINDOW = 2048
MAX_ANSWER = 1024


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def stage_flatten(args):
    os.makedirs(args.out_dir, exist_ok=True)
    by_source = {}
    stats = {}

    for d in sorted(glob.glob(os.path.join(args.raw_dir, "data*"))):
        if not os.path.isdir(d):
            continue
        src = os.path.basename(d)
        table, order = {}, []
        st = {"raw": 0, "unusable": 0, "samples": 0, "dup": 0}
        for path in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
            for row in bd.iter_jsonl(path):
                st["raw"] += 1
                if row is None:
                    st["unusable"] += 1
                    continue
                turns = bd.to_turns(row)
                if turns is None:
                    st["unusable"] += 1
                    continue
                for conv in bd.flatten_to_single_turn(turns):
                    st["samples"] += 1
                    k = bd.sample_key(conv)
                    if k in table:
                        st["dup"] += 1
                        continue
                    table[k] = conv
                    order.append(k)
        by_source[src] = (table, order)
        stats[src] = st
        log(f"[flatten] {src}: {st}  unique={len(order)}")

    # Cross-source dedup: earlier source wins.
    merged = {}
    for src in sorted(by_source):
        for k, conv in by_source[src][0].items():
            merged.setdefault(k, (src, conv))
    log(f"[flatten] merged unique across sources: {len(merged)}")

    # Deterministic per-source cap.
    rng = random.Random(args.seed)
    per_source = {}
    for k, (src, conv) in merged.items():
        per_source.setdefault(src, []).append((k, conv))
    picked = []
    for src in sorted(per_source):
        items = per_source[src]
        items.sort(key=lambda kv: hashlib.sha1(
            (str(args.seed) + kv[0]).encode()).hexdigest())
        cap = args.per_source or len(items)
        picked.extend((src, c) for _k, c in items[:cap])
        log(f"[flatten] {src}: picked {min(cap, len(items))}/{len(items)}")

    out = os.path.join(args.out_dir, "flattened.jsonl")
    with open(out, "w", encoding="utf-8") as fh:
        for i, (src, conv) in enumerate(picked):
            fh.write(json.dumps({"source": src, "_idx": i, "conversations": conv},
                                ensure_ascii=False) + "\n")
    log(f"[flatten] wrote {out}: {len(picked)} samples")


def stage_regenerate(args):
    # torchcodec and the CUDA-13 runtime must be handled BEFORE vllm imports.
    _block_torchcodec()
    _ensure_cu13()
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    src = os.path.join(args.out_dir, "flattened.jsonl")
    rows = [json.loads(l) for l in open(src, encoding="utf-8")]
    log(f"[regen] {len(rows)} prompts")

    prompts, keep = [], []
    skipped = 0
    budget = args.max_model_len - args.max_new_tokens
    for i, r in enumerate(rows):
        text = tok.apply_chat_template(r["conversations"][:-1], tokenize=False,
                                       add_generation_prompt=True,
                                       enable_thinking=False)
        if len(tok.encode(text, add_special_tokens=False)) > budget:
            skipped += 1
            continue
        prompts.append(text); keep.append(i)
    if skipped:
        log(f"[regen] skipping {skipped} prompts over {budget} tokens")
    rows = [rows[i] for i in keep]
    llm = LLM(model=args.tokenizer, dtype="bfloat16",
              max_model_len=args.max_model_len, max_num_seqs=128,
              gpu_memory_utilization=args.gpu_mem_frac, trust_remote_code=True,
              enable_prefix_caching=True)
    params = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens,
                            skip_special_tokens=True)

    out = os.path.join(args.out_dir, "regenerated.jsonl")
    done = 0
    if args.resume and os.path.exists(out):
        done = sum(1 for _ in open(out, encoding="utf-8"))
        log(f"[regen] resuming after {done}")
    with open(out, "a" if done else "w", encoding="utf-8") as fh:
        for start in range(done, len(rows), args.chunk):
            chunk = rows[start:start + args.chunk]
            outs = llm.generate(prompts[start:start + args.chunk], params)
            for r, o in zip(chunk, outs):
                ans = o.outputs[0].text.strip() if o.outputs else ""
                finish = o.outputs[0].finish_reason if o.outputs else "empty"
                conv = (r["conversations"][:-1] + [{"role": "assistant", "content": ans}]
                        if ans else r["conversations"])
                fh.write(json.dumps({"source": r["source"], "_idx": r["_idx"],
                                     "regenerated": bool(ans), "finish_reason": finish,
                                     "conversations": conv}, ensure_ascii=False) + "\n")
            fh.flush(); os.fsync(fh.fileno())
            log(f"[regen] {min(start+args.chunk, len(rows))}/{len(rows)}")


def stage_window(args):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    src = os.path.join(args.out_dir, "regenerated.jsonl")
    rows = [json.loads(l) for l in open(src, encoding="utf-8")]
    kept, dropped, seen = [], {}, set()
    for r in rows:
        if not r.get("regenerated"):
            dropped["empty"] = dropped.get("empty", 0) + 1
            continue
        if r.get("finish_reason") == "length":
            dropped["over_1024"] = dropped.get("over_1024", 0) + 1
            continue
        conv, why = bd.apply_window(r["conversations"], tok,
                                    args.max_window, args.max_answer)
        if conv is None:
            dropped[why] = dropped.get(why, 0) + 1
            continue
        users = [t["content"] for t in conv if t["role"] == "user"]
        k = hashlib.sha1(((users[-1] if users else "") + "\x00" +
                          conv[-1]["content"]).encode()).hexdigest()
        if k in seen:
            dropped["dup"] = dropped.get("dup", 0) + 1
            continue
        seen.add(k)
        kept.append((r["source"], conv))

    rng = random.Random(args.seed)
    rng.shuffle(kept)
    n_test = max(1, int(len(kept) * args.test_frac))
    test, train = kept[:n_test], kept[n_test:]

    for name, part in (("train", train), ("test", test)):
        path = os.path.join(args.out_dir, f"{name}.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            for i, (src, conv) in enumerate(part):
                fh.write(json.dumps({
                    "id": hashlib.sha1((str(i)+json.dumps(conv, ensure_ascii=False))
                                       .encode()).hexdigest()[:16],
                    "source": src, "conversations": conv}, ensure_ascii=False) + "\n")
        log(f"[window] {name}: {len(part)} -> {path}")
    log(f"[window] dropped: {dropped}")


def _block_torchcodec():
    import importlib.machinery, types
    if "torchcodec" in sys.modules:
        return
    d = types.ModuleType("torchcodec.decoders")
    d.AudioDecoder = None; d.VideoDecoder = None
    p = types.ModuleType("torchcodec"); p.decoders = d
    p.__spec__ = importlib.machinery.ModuleSpec("torchcodec", loader=None)
    d.__spec__ = importlib.machinery.ModuleSpec("torchcodec.decoders", loader=None)
    sys.modules["torchcodec"] = p; sys.modules["torchcodec.decoders"] = d


def _ensure_cu13():
    """vLLM needs the CUDA-13 runtime; re-exec once with it on the loader path."""
    lib = str(_p.sglang_site_packages() / "nvidia" / "cu13" / "lib")
    if os.environ.get("ALL8_LDPATH") or not os.path.isdir(lib):
        return
    parts = [x for x in os.environ.get("LD_LIBRARY_PATH", "").split(":") if x]
    if lib in parts:
        return
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = ":".join([lib] + parts)
    env["ALL8_LDPATH"] = "1"
    os.execve(sys.executable, [sys.executable] + sys.argv, env)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=RAW)
    ap.add_argument("--out-dir", default=OUT)
    ap.add_argument("--tokenizer", default=TOKENIZER)
    ap.add_argument("--stage", default="all",
                    choices=("flatten", "regenerate", "window", "all"))
    ap.add_argument("--seed", type=int, default=20261124)
    ap.add_argument("--per-source", type=int, default=4000,
                    help="max samples per source dir; 0 = no cap")
    ap.add_argument("--max-window", type=int, default=MAX_WINDOW)
    ap.add_argument("--max-answer", type=int, default=MAX_ANSWER)
    ap.add_argument("--max-new-tokens", type=int, default=MAX_ANSWER)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-mem-frac", type=float, default=0.85)
    ap.add_argument("--chunk", type=int, default=500)
    ap.add_argument("--test-frac", type=float, default=0.05)
    ap.add_argument("--resume", action="store_true", default=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.stage in ("flatten", "all"):
        stage_flatten(args)
    if args.stage in ("regenerate", "all"):
        stage_regenerate(args)
    if args.stage in ("window", "all"):
        stage_window(args)


if __name__ == "__main__":
    main()
