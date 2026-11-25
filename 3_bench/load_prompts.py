"""Turn each held-out evaluation set into chat prompts.

Two families live under the workspace and they need different handling:

* ``code/SpecForge/artifacts/datasets/v4-data1/test.jsonl`` -- already
  ``{"conversations": [...]}`` and in the same schema the drafts were trained on.
  This is the in-domain set.
* ``datasets/<name>/`` -- seven out-of-domain sets in five different schemas
  (math word problems, code completion, open-ended chat). This is the
  generalisation set: the drafts never saw any of it.

Every loader returns a list of ``{"id", "prompt"}`` where ``prompt`` is the full
chat-templated string (thinking disabled), ready for SGLang's /generate.

Usage:
    python load_prompts.py --set gsm8k --out /tmp/gsm8k.jsonl [--limit N]
    python load_prompts.py --list
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
DATA = str(_p.ROOT / "datasets")
TOKENIZER = str(_p.TOKENIZER)

# Instruction text is kept short and task-appropriate: these sets measure how
# well the draft tracks the *target's* continuations on unseen domains, not how
# good the target is at solving them.
SETS = {
    "data1": dict(path=str(_p.DATASETS / "v4-data1" / "test.jsonl"),
                  kind="conversations"),
    # The eight business sources, split out of the merged all8 test split.
    # Useful for asking "which task does this draft serve best".
    **{f"src{n}": dict(path=str(_p.DATASET_DIR / "v5-all8" / "by_source" / f"data{n}.jsonl"),
                       kind="conversations") for n in range(1, 9)},
    "aime25": dict(path=f"{DATA}/aime25/test.jsonl", kind="problem"),
    "math500": dict(path=f"{DATA}/math500/test.jsonl", kind="problem"),
    "gsm8k": dict(path=f"{DATA}/gsm8k/test.jsonl", kind="question"),
    "humaneval": dict(path=f"{DATA}/humaneval/test.jsonl", kind="code_prompt"),
    "mbpp": dict(path=f"{DATA}/mbpp/test.jsonl", kind="mbpp"),
    "lcb": dict(path=f"{DATA}/lcb/test5.jsonl", kind="lcb"),
    "mtbench": dict(path=f"{DATA}/mtbench/question.jsonl", kind="mtbench"),
}

MATH_SYSTEM = "You are a helpful assistant solving math problems. Show your reasoning and give the final answer."
CODE_SYSTEM = "You are an expert programmer. Write correct, complete code."
CHAT_SYSTEM = "You are a helpful, knowledgeable assistant."


def _rows(path, limit):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows[:limit] if limit else rows


def build(set_name, limit=None):
    spec = SETS[set_name]
    kind = spec["kind"]
    out = []

    if kind == "conversations":
        # Already chat-shaped; hand back the rendered conversation.
        for i, row in enumerate(_rows(spec["path"], limit)):
            out.append({"id": f"{set_name}-{i}", "messages": row["conversations"][:-1]})
        return out

    for i, row in enumerate(_rows(spec["path"], limit)):
        if kind == "problem":
            msgs = [{"role": "system", "content": MATH_SYSTEM},
                    {"role": "user", "content": row["problem"]}]
        elif kind == "question":
            msgs = [{"role": "system", "content": MATH_SYSTEM},
                    {"role": "user", "content": row["question"]}]
        elif kind == "code_prompt":
            # HumanEval's `prompt` is already a signature + docstring + examples:
            # the model is expected to continue it, so send it verbatim.
            msgs = [{"role": "system", "content": CODE_SYSTEM},
                    {"role": "user", "content": row["prompt"]}]
        elif kind == "mbpp":
            msgs = [{"role": "system", "content": CODE_SYSTEM},
                    {"role": "user", "content": row["text"] + "\n\nWrite the function."}]
        elif kind == "lcb":
            body = row["question_content"]
            starter = (row.get("starter_code") or "").strip()
            if starter:
                body += "\n\nStart from this code:\n```\n" + starter + "\n```"
            msgs = [{"role": "system", "content": CODE_SYSTEM},
                    {"role": "user", "content": body}]
        elif kind == "mtbench":
            # prompt is a list: first turn is the question, later turns are
            # follow-ups. Only the first turn is used -- multi-turn needs the
            # model's own prior answer, which turns this into a chat trace, not
            # a spec-decode measurement.
            p = row["prompt"]
            text = p[0] if isinstance(p, list) and p else str(p)
            msgs = [{"role": "system", "content": CHAT_SYSTEM},
                    {"role": "user", "content": text}]
        else:
            raise SystemExit(f"unknown kind {kind}")
        out.append({"id": f"{set_name}-{i}", "messages": msgs})
    return out


def render(records, tokenizer_path=TOKENIZER):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    for r in records:
        r["prompt"] = tok.apply_chat_template(
            r["messages"], tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", dest="set_name")
    ap.add_argument("--out")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for name, spec in SETS.items():
            n = sum(1 for _ in open(spec["path"], encoding="utf-8"))
            print(f"{name:10s} {n:6d} rows  {spec['path']}")
        return

    if not args.set_name or not args.out:
        sys.exit("--set and --out are required (or use --list)")

    records = render(build(args.set_name, args.limit))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[prompts] {args.set_name}: {len(records)} rows -> {args.out}")


if __name__ == "__main__":
    main()
