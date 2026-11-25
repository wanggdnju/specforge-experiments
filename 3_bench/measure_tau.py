"""Measure spec-decode acceptance against an already-running SGLang server.

Kept separate from the launcher so the same script can be pointed at any port.
Reports mean tau (accepted length per step) and the raw accepted/drafted totals;
tau alone is not enough to tell "the drafter is good" from "the drafter proposed
almost nothing".

IMPORTANT - prompt rendering
----------------------------
Earlier versions sent `system + "\n\n" + user` as raw text. That is wrong for
two reasons, and both destroy the acceptance rate:

* SGLang's /generate tokenises the text as-is; it does not apply a chat
  template. The draft model was trained on "<|im_start|>system ... <|im_end|>
  <|im_start|>user ... <|im_end|><|im_start|>assistant" (plus Qwen3's empty
  thinking block), so raw text puts it far out of distribution.
* Qwen3 also needs enable_thinking=False to match training. The default
  template ends at "<|im_start|>assistant\n" with thinking enabled.

So this script renders each prompt through the target tokenizer's chat
template, exactly as SpecForge's regeneration step did, then sends that text.
Pass --raw-text to reproduce the old behaviour for diagnosis.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
_sys.path.insert(1, str(_Path(__file__).resolve().parent))
import paths as _p


import argparse
import json
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor


def render_prompts(path, limit, tokenizer_path):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    rows = [json.loads(line) for line in open(path, encoding="utf-8")][:limit]

    # Two input shapes reach this function:
    #   * sf-pipeline/load_prompts.py output -- already rendered, has "prompt"
    #   * a dataset row with "conversations" -- needs templating here
    # Honouring the pre-rendered form is what lets one server measure eight
    # differently-shaped evaluation sets.
    out = []
    for row in rows:
        if row.get("prompt"):
            out.append(row["prompt"])
        else:
            out.append(tok.apply_chat_template(
                row["conversations"][:-1],   # drop the assistant target
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            ))
    return out


def raw_prompts(path, limit):
    rows = [json.loads(line) for line in open(path, encoding="utf-8")][:limit]
    out = []
    for row in rows:
        conv = row["conversations"]
        system = next((m["content"] for m in conv if m["role"] == "system"), "")
        user = next((m["content"] for m in conv if m["role"] == "user"), "")
        out.append(f"{system}\n\n{user}" if system else user)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--prompts", default=(
        str(_p.DATASETS / "v4-data1" / "test.jsonl")))
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--label", default="")
    ap.add_argument("--tokenizer",
                    default=str(_p.TOKENIZER))
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--raw-text", action="store_true",
                    help="Diagnostic: send system+user as raw text instead of the "
                         "rendered chat template (reproduces the old path).")
    args = ap.parse_args()

    import requests

    if args.raw_text:
        prompts = raw_prompts(args.prompts, args.limit)
        print("[measure] WARNING: --raw-text reproduces the old broken path",
              file=sys.stderr)
    else:
        prompts = render_prompts(args.prompts, args.limit, args.tokenizer)

    url = f"http://127.0.0.1:{args.port}/generate"

    def one(text):
        body = {"text": text,
                "sampling_params": {"temperature": 0,
                                    "max_new_tokens": args.max_new_tokens}}
        try:
            return requests.post(url, json=body, timeout=600).json()
        except Exception as exc:
            print(f"  failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            return None

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(one, prompts))

    taus, rates = [], []
    accepted = drafted = tokens = skipped = 0
    for d in results:
        if not d:
            skipped += 1
            continue
        m = d.get("meta_info") or {}
        tau = m.get("spec_accept_length")
        if tau is None:
            skipped += 1
            continue
        taus.append(float(tau))
        rates.append(float(m.get("spec_accept_rate") or 0))
        accepted += int(m.get("spec_accepted_tokens") or 0)
        drafted += int(m.get("spec_drafted_tokens") or 0)
        tokens += int(m.get("completion_tokens") or 0)

    if not taus:
        print(json.dumps({"label": args.label, "n": 0, "skipped": skipped,
                          "error": "no spec metrics"}))
        return

    print(json.dumps({
        "label": args.label,
        "n": len(taus),
        "skipped": skipped,
        "completion_tokens": tokens,
        "tau_mean": round(statistics.mean(taus), 4),
        "tau_median": round(statistics.median(taus), 4),
        "tau_min": round(min(taus), 4),
        "tau_max": round(max(taus), 4),
        "accept_rate_mean": round(statistics.mean(rates), 4),
        "accepted_tokens": accepted,
        "drafted_tokens": drafted,
    }))


if __name__ == "__main__":
    main()
