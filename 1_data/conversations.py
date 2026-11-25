"""Shared helpers for turning raw request dumps into SpecForge conversations.

These are the only parts of the original `build_dataset.py` still in use; the
rest of that script (its own sampling and regeneration loops) was superseded by
`build_data.py`. Kept separate so the pipeline logic reads without 500 lines of
parsing helpers inline.

The four shapes seen in the raw dumps:

* `messages` + `choices`  (data3-8, most sources)
* `model_prompt_tmpl` + `model_prompt_placeholder` + `choices`  (data1, data2)
* multi-turn agent traces with `tool_calls`
* plain `messages` with no target

`to_turns` normalises all of them; `flatten_to_single_turn` then emits one
sample per assistant turn.
"""

from __future__ import annotations

import hashlib
import json


ROLE_MAP = {"human": "user", "user": "user", "gpt": "assistant",
            "assistant": "assistant", "system": "system", "developer": "system",
            "tool": "tool"}
SKIP_ROLES = {"assistant_reasoning_effort"}
TOOL_PREFIX = "[tool result] "


def flatten(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(b.get("text", "")) for b in content if isinstance(b, dict))
    if isinstance(content, dict):
        return str(content.get("text", ""))
    return str(content)

def render_tool_calls(calls):
    if not isinstance(calls, list) or not calls:
        return ""
    parts = []
    for call in calls:
        if not isinstance(call, dict):
            continue
        fn = call.get("function") or {}
        name = fn.get("name") or call.get("name") or "tool"
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                pass
        try:
            arg_text = json.dumps(args, ensure_ascii=False, sort_keys=True)
        except Exception:
            arg_text = str(args)
        parts.append(f"[tool_call] {name}({arg_text})")
    return "\n".join(parts)

def render_template(row):
    tmpl = row.get("model_prompt_tmpl") or ""
    ph = row.get("model_prompt_placeholder") or []
    if isinstance(ph, list):
        for item in ph:
            if isinstance(item, dict) and "key" in item:
                tmpl = tmpl.replace("{{" + str(item["key"]) + "}}", str(item.get("value", "")))
    return tmpl

def _rendered_prompt(row):
    """Per-row prompt with placeholders substituted.

    Dumps like data1 share ONE ``model_prompt_tmpl`` (with ``{{context}}``) across
    every row; the per-row text lives in ``model_prompt_placeholder``. The
    ``messages[].content`` copy is the unsubstituted template, so using it makes
    every prompt identical.
    """
    tmpl = row.get("model_prompt_tmpl")
    ph = row.get("model_prompt_placeholder")
    if not isinstance(tmpl, str) or not tmpl.strip():
        return ""
    if not isinstance(ph, list) or not ph:
        return ""
    return render_template(row).strip()

def _choice_text(row):
    choices = row.get("choices")
    if not isinstance(choices, list):
        return ""
    chosen = next((c for c in choices if isinstance(c, dict) and c.get("index", 0) == 0),
                  None)
    if chosen is None:
        chosen = next((c for c in choices if isinstance(c, dict)), None)
    if chosen is None:
        return ""
    msg = chosen.get("message")
    if not isinstance(msg, dict):
        return ""
    text = flatten(msg.get("content")).strip()
    if not text:
        text = render_tool_calls(msg.get("tool_calls")).strip()
    return text

def to_turns(row):
    """Normalize one raw row into [{'role','content'}] ending in an assistant turn."""
    raw_messages = row.get("messages")
    turns = []

    if isinstance(raw_messages, list) and raw_messages:
        for msg in raw_messages:
            if not isinstance(msg, dict):
                continue
            role = ROLE_MAP.get(msg.get("role") or msg.get("from") or "")
            if role is None or msg.get("role") in SKIP_ROLES:
                continue
            text = flatten(msg.get("content")).strip()
            calls = render_tool_calls(msg.get("tool_calls"))
            if role == "assistant":
                body = text or calls
                if not body:
                    continue
                turns.append({"role": "assistant", "content": body})
            elif role == "tool":
                if text:
                    turns.append({"role": "tool", "content": TOOL_PREFIX + text})
            elif role == "user":
                if text:
                    turns.append({"role": "user", "content": text})
            else:
                if text:
                    turns.append({"role": "system", "content": text})
        rendered = _rendered_prompt(row)
        if rendered:
            idx = max((i for i, t in enumerate(turns) if t["role"] == "user"), default=None)
            if idx is not None:
                turns[idx] = {"role": "user", "content": rendered}
        target = _choice_text(row)
        if target:
            turns.append({"role": "assistant", "content": target})
    else:
        target = _choice_text(row)
        if not target:
            return None
        prompt = _rendered_prompt(row)
        if not prompt:
            return None
        turns = [{"role": "user", "content": prompt},
                 {"role": "assistant", "content": target}]

    if not turns or turns[-1]["role"] != "assistant":
        return None
    if not any(t["role"] == "user" for t in turns):
        return None
    return turns

def fold_tool(turns):
    """tool -> user side; merge consecutive user-side turns (templates reject two)."""
    out = []
    for t in turns:
        role, content = t["role"], t["content"]
        if role == "tool":
            role = "user"
        if role == "user" and out and out[-1]["role"] == "user":
            out[-1] = {"role": "user",
                       "content": out[-1]["content"] + "\n\n" + content}
            continue
        out.append({"role": role, "content": content})
    return out


# --------------------------------------------------------------------------
# step 1: flatten to single-turn samples
# --------------------------------------------------------------------------

def flatten_to_single_turn(turns):
    """One sample per assistant turn: everything before it is context."""
    samples = []
    for i, t in enumerate(turns):
        if t["role"] != "assistant":
            continue
        context = turns[:i]
        if not any(c["role"] == "user" for c in context):
            continue
        samples.append(fold_tool(context + [t]))
    return samples


# --------------------------------------------------------------------------
# steps 1-3: gather, sample, deduplicate
# --------------------------------------------------------------------------

def iter_jsonl(path):
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                yield None

def sample_key(conv):
    """Identity for dedup: last user turn + target.

    Context length varies with the window, but the (question, answer) pair is
    what must not be duplicated across train/test.
    """
    users = [t["content"] for t in conv if t["role"] == "user"]
    target = conv[-1]["content"]
    payload = (users[-1] if users else "") + "\x00" + target
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()

def apply_window(conv, tokenizer, max_window, max_answer):
    """Keep the last ``max_window`` tokens. Returns None if the answer is too long.

    Right-anchored because a DFlash draft only attends to a recent window of the
    target's tokens at serving time; discarding the prompt head matches that.
    """
    answer = conv[-1]["content"]
    ans_ids = tokenizer.encode(answer, add_special_tokens=False)
    if len(ans_ids) > max_answer:
        return None, "answer_too_long"

    # Budget for context = window minus answer minus template markup.
    markup = 16 + 6 * len(conv)
    ctx_budget = max_window - len(ans_ids) - markup
    if ctx_budget < 64:
        return None, "no_room_for_context"

    kept = []
    for t in reversed(conv[:-1]):
        ids = tokenizer.encode(t["content"], add_special_tokens=False)
        if len(ids) <= ctx_budget:
            kept.append(t)
            ctx_budget -= len(ids)
        else:
            if ctx_budget > 64:
                kept.append({"role": t["role"],
                             "content": tokenizer.decode(ids[-ctx_budget:])})
            break
    kept.reverse()
    if not kept or not any(t["role"] == "user" for t in kept):
        return None, "no_user_left"

    out = kept + [conv[-1]]
    rendered = tokenizer.apply_chat_template(out, tokenize=False,
                                             add_generation_prompt=False)
    if len(tokenizer.encode(rendered, add_special_tokens=False)) > max_window + 32:
        return None, "still_too_long"
    return out, "ok"


# --------------------------------------------------------------------------

