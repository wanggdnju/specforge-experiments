"""End-to-end: generate configs -> train -> export -> serve+bench -> report.

One command trains and benchmarks every draft architecture SpecForge can build
for Qwen3-8B on the v4-data1 event-extraction set.

Stages (each skippable, each resumable):

    configs   write one YAML per (algorithm, init) via gen_configs.py
    train     `specforge train` through the online capture server
    export    `specforge export --to hf` -> servable checkpoint
    bench     serve with SGLang, measure accepted length on the test set
    report    markdown + json summary under artifacts/pipeline/reports

Everything is serial: there is one GPU and both training and serving need the
whole card. A completed stage leaves a marker file so a re-run resumes instead
of repeating hours of work.

    python run_pipeline.py                    # everything
    python run_pipeline.py --stages report    # just rebuild the summary
    python run_pipeline.py --runs bench-dflash-scratch   # one run
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
import re
import shutil
import subprocess
import sys
import time

ROOT = str(_p.ROOT)
REPO = str(_p.REPO)
ART = f"{REPO}/artifacts"
PIPE = str(_p.PIPELINE_WORK)
CONFIGS = f"{PIPE}/configs"
LOGS = f"{PIPE}/logs"
REPORTS = f"{PIPE}/reports"
RUNS = str(_p.RUNS)
CHECKPOINTS = str(_p.CHECKPOINTS)

ONLINE = str(_p.PIPELINE / "2_train" / "run_online.sh")
VENV_PY = str(_p.PY_TRAIN)
PIPELINE_DIR = str(_p.PIPELINE)


# --------------------------------------------------------------------- helpers
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def marker(run_name, stage):
    return os.path.join(LOGS, f"{run_name}.{stage}.done")


def is_done(run_name, stage):
    return os.path.exists(marker(run_name, stage))


def mark(run_name, stage, payload=None):
    os.makedirs(LOGS, exist_ok=True)
    with open(marker(run_name, stage), "w", encoding="utf-8") as fh:
        json.dump(payload or {"at": time.time()}, fh)


def index_entries(run_filter=None, args=None):
    """Load the suite manifest (configs/suite.yaml).

    The manifest is the source of truth for what gets trained and measured; the
    per-run YAMLs it points at are what `specforge train` actually consumes.
    Falls back to the generated index.json only if no suite was given.
    """
    suite_path = getattr(args, "suite", None) if args else None
    if suite_path and os.path.exists(suite_path):
        import yaml
        suite = yaml.safe_load(open(suite_path, encoding="utf-8"))
        base = os.path.dirname(os.path.abspath(suite_path))
        entries = []
        for r in suite.get("runs", []):
            entries.append({
                "run_name": r["name"],
                "algorithm": r["algorithm"],
                "init": r["init"],
                # Pretrained baselines name a ready-made `draft` and have no
                # generated config; keep `config` empty for them.
                "config": os.path.join(base, r["config"]) if r.get("config") else "",
                "draft": r.get("draft"),
                "init_from": r.get("init_from"),
                "sglang_algo": r.get("sglang_algo"),
                "draft_tokens": r.get("draft_tokens"),
            })
    else:
        path = os.path.join(CONFIGS, "index.json")
        if not os.path.exists(path):
            sys.exit(f"no suite manifest and no {path}; run the `configs` stage first")
        entries = json.load(open(path, encoding="utf-8"))
    if run_filter:
        entries = [e for e in entries if e["run_name"] in run_filter]
    return entries


def final_step_dir(run_name):
    """Locate the last checkpoint directory `specforge train` wrote."""
    base = os.path.join(RUNS, run_name)
    if not os.path.isdir(base):
        return None
    steps = []
    for name in os.listdir(base):
        m = re.match(rf"{re.escape(run_name)}-step(\d+)$", name)
        if m and os.path.isdir(os.path.join(base, name)):
            steps.append((int(m.group(1)), os.path.join(base, name)))
    return max(steps)[1] if steps else None


# ---------------------------------------------------------------------- stages
def stage_configs(args):
    os.makedirs(CONFIGS, exist_ok=True)
    cmd = [sys.executable, f"{PIPELINE_DIR}/2_train/gen_configs.py", "--out-dir", CONFIGS]
    subprocess.run(cmd, check=True)


def stage_train(args):
    for e in index_entries(args.runs, args):
        name, algo = e["run_name"], e["algorithm"]
        if is_done(name, "train"):
            log(f"train {name}: already done, skipping")
            continue
        # A failed attempt leaves control/refs.jsonl.* and consumer-state/
        # behind, and SpecForge refuses to reuse them:
        #   "control path ... has artifacts from an existing attempt"
        #   "consumer metadata database must use a fresh attempt path"
        # Since a run that got this far has no checkpoint and is being retried
        # from scratch, wiping the directory is the correct reset.
        run_dir = os.path.join(RUNS, name)
        if os.path.isdir(run_dir) and not final_step_dir(name):
            log(f"train {name}: clearing stale attempt dir")
            shutil.rmtree(run_dir, ignore_errors=True)
        log(f"train {name} ({algo}, {e['init']}) ...")
        log_path = os.path.join(LOGS, f"{name}.train.log")
        os.makedirs(LOGS, exist_ok=True)
        cmd = [ONLINE, e["config"]]
        with open(log_path, "w", encoding="utf-8") as fh:
            rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode
        if rc != 0:
            log(f"train {name}: FAILED rc={rc} (see {log_path})")
            continue
        step_dir = final_step_dir(name)
        if not step_dir:
            log(f"train {name}: no checkpoint directory produced")
            continue
        mark(name, "train", {"step_dir": step_dir})
        log(f"train {name}: done -> {os.path.basename(step_dir)}")


def _draft_config_from_run(entry):
    """Read model.draft_model_config out of the run's own YAML.

    The export must use the config the run trained with -- dflash2/dspark were
    retrained at block 16 via *-b16.json, and guessing the path from the
    algorithm name reverted them to the 8/7 defaults.
    """
    path = entry.get("config")
    if not path or not os.path.exists(path):
        return None
    try:
        import yaml
        cfg = yaml.safe_load(open(path, encoding="utf-8"))
        return (cfg.get("model") or {}).get("draft_model_config")
    except Exception:
        return None


def stage_export(args):
    for e in index_entries(args.runs, args):
        name, algo = e["run_name"], e["algorithm"]
        if is_done(name, "export"):
            log(f"export {name}: already done, skipping")
            continue
        train_info_path = marker(name, "train")
        if not os.path.exists(train_info_path):
            log(f"export {name}: no training result, skipping")
            continue
        step_dir = json.load(open(train_info_path, encoding="utf-8"))["step_dir"]
        out_dir = os.path.join(CHECKPOINTS, e["run_name"])
        # Export with the config the run was TRAINED with, read from the run
        # YAML. Deriving it from the algorithm name silently reverted the block
        # size (dflash2/dspark were trained at 16 but exported at 8/7).
        draft_cfg = _draft_config_from_run(e)
        if not draft_cfg:
            # Warm-start runs name a checkpoint instead of a config; the
            # checkpoint carries its own config.json, which is what the run
            # actually trained against.
            src = (e.get("init_from") or "").strip()
            cand = src and os.path.join(src, "config.json")
            if cand and os.path.exists(cand):
                draft_cfg = cand
        if not draft_cfg:
            log(f"export {name}: cannot resolve draft config, skipping")
            continue
        cmd = [VENV_PY, "-m", "specforge.cli", "export", "--to", "hf",
               "--checkpoint", step_dir,
               "--draft-config", draft_cfg,
               "--output-dir", out_dir]
        # EAGLE-family drafts own an embedding, but training keeps it frozen and
        # out of the checkpoint. Without this the export aborts with
        #   "checkpoint has no embed_tokens.weight ... pass embedding_source"
        if algo in ("eagle3", "peagle"):
            cmd += ["--embedding-source", f"{ROOT}/model/Qwen3-8B"]
        log_path = os.path.join(LOGS, f"{name}.export.log")
        with open(log_path, "w", encoding="utf-8") as fh:
            rc = subprocess.run(cmd, cwd=REPO, stdout=fh,
                                stderr=subprocess.STDOUT).returncode
        if rc != 0:
            log(f"export {name}: FAILED rc={rc} (see {log_path})")
            continue
        mark(name, "export", {"draft_dir": out_dir})
        log(f"export {name}: done -> {out_dir}")


def stage_bench(args):
    port = args.port
    for e in index_entries(args.runs, args):
        name, algo = e["run_name"], e["algorithm"]
        if is_done(name, "bench"):
            log(f"bench {name}: already done, skipping")
            continue
        export_info = marker(name, "export")
        draft = ""
        if os.path.exists(export_info):
            draft = json.load(open(export_info, encoding="utf-8"))["draft_dir"]
        out_json = os.path.join(LOGS, f"{name}.bench.json")
        cmd = [sys.executable, f"{PIPELINE_DIR}/3_bench/bench_one.py",
               "--draft", draft or "/nonexistent",
               "--algorithm", algo, "--port", str(port),
               "--limit", str(args.bench_limit), "--sets", args.sets,
               "--prompt-dir", os.path.join(PIPE, "prompts"),
               "--out", out_json,
               "--log", os.path.join(LOGS, f"{name}.server.log")]
        # The suite records what SGLang needs per run; the harness must not
        # re-derive it or the two can drift apart.
        if e.get("sglang_algo") is not None:
            cmd += ["--sglang-algo", e["sglang_algo"]]
        if e.get("draft_tokens") is not None:
            cmd += ["--draft-tokens", str(e["draft_tokens"])]
        log(f"bench {name} ({algo}) ...")
        subprocess.run(cmd, check=False)
        if os.path.exists(out_json):
            res = json.load(open(out_json, encoding="utf-8"))
            mark(name, "bench", res)
            taus = {s: v.get("tau_mean") for s, v in (res.get("sets") or {}).items()}
            log(f"bench {name}: {res.get('status')} tau={taus}")


def stage_report(args):
    entries = index_entries(None, args)
    rows, skipped, all_sets = [], [], []
    for e in entries:
        name = e["run_name"]
        rec = {"run": name, "algorithm": e["algorithm"], "init": e["init"]}
        for stage, key in (("train", "train"), ("export", "export"), ("bench", "bench")):
            p = marker(name, stage)
            if not os.path.exists(p):
                continue
            # A marker can be zero-length if the process died mid-write; treat
            # that as "stage absent" rather than aborting the whole report.
            try:
                rec[key] = json.load(open(p, encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
        b = rec.get("bench") or {}
        if b.get("status") == "OK" and b.get("sets"):
            rec["tau"] = {s: v.get("tau_mean") for s, v in b["sets"].items()
                          if v.get("status") == "OK"}
            rec["accept"] = {s: v.get("accept_rate_mean")
                             for s, v in b["sets"].items() if v.get("status") == "OK"}
            for s in rec["tau"]:
                if s not in all_sets:
                    all_sets.append(s)
            rows.append(rec)
        else:
            skipped.append(rec)

    # Column order: the in-domain set first, then the rest alphabetically.
    all_sets.sort(key=lambda s: (s != "data1", s))

    def mean_tau(r):
        v = [x for x in r["tau"].values() if x is not None]
        return sum(v) / len(v) if v else 0.0

    rows.sort(key=mean_tau, reverse=True)

    os.makedirs(REPORTS, exist_ok=True)
    with open(os.path.join(REPORTS, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump({"benchmarked": rows, "not_benchmarked": skipped,
                   "sets": all_sets}, fh, indent=2, ensure_ascii=False, default=str)

    header = "| rank | run | algorithm | init | mean | " + \
             " | ".join(all_sets) + " |"
    sep = "| --- | --- | --- | --- | --- | " + " | ".join("---" for _ in all_sets) + " |"
    lines = [
        "# Draft benchmark: Qwen3-8B",
        "",
        "Accepted length (tau) per **speculative step**, measured against a live",
        "SGLang server. Higher is better. `data1` is the in-domain set the drafts",
        "were trained on; everything else is unseen, so those columns measure how",
        "well a draft transfers to out-of-distribution text.",
        "",
        "Caveat: raw tau is not directly comparable *across algorithms* when the",
        "proposal budget differs. As of this report: dflash / domino / dspine /",
        "eagle3 draft 16 tokens per step (eagle3 as a depth-4 topk-2 tree);",
        "dflash2 uses 16 for scratch but 8 for warmstart; dspark uses 7",
        "(gamma 6 + 1). Those last two are upstream limits -- no block-16",
        "pretrained checkpoint exists for dflash2, and the published dspark",
        "block-16 weights could not be adapted to SpecForge's contract. Use the",
        "accept rate column for cross-algorithm reading.",
        "",
        header, sep,
    ]
    for i, r in enumerate(rows, 1):
        cells = []
        for s in all_sets:
            v = r["tau"].get(s)
            cells.append(f"{v:.2f}" if v is not None else "—")
        lines.append(
            f"| {i} | `{r['run']}` | {r['algorithm']} | {r['init']} | "
            f"**{mean_tau(r):.2f}** | " + " | ".join(cells) + " |")

    lines += ["", "## Accept rate", "", header, sep]
    for i, r in enumerate(rows, 1):
        cells = []
        for s in all_sets:
            v = r["accept"].get(s)
            cells.append(f"{v:.1%}" if v is not None else "—")
        lines.append(
            f"| {i} | `{r['run']}` | {r['algorithm']} | {r['init']} | — | "
            + " | ".join(cells) + " |")

    if skipped:
        lines += ["", "## Not benchmarked", "",
                  "| run | algorithm | init | reason |", "| --- | --- | --- | --- |"]
        for r in skipped:
            b = r.get("bench") or {}
            reason = b.get("reason") or b.get("status") or "no bench result"
            reason = str(reason).replace("|", "/").replace("\n", " ")[:120]
            lines.append(f"| `{r['run']}` | {r['algorithm']} | {r['init']} | {reason} |")

    lines += ["", "---", "",
              "Reproduce: `./run.sh pipeline`", ""]
    out_md = os.path.join(REPORTS, "summary.md")
    with open(out_md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    log(f"report -> {out_md}")
    print("\n".join(lines))


STAGES = {
    "configs": stage_configs,
    "train": stage_train,
    "export": stage_export,
    "bench": stage_bench,
    "report": stage_report,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default=str(_p.SUITE),
                    help="suite manifest listing every run and eval set")
    ap.add_argument("--stages", default="configs,train,export,bench,report")
    ap.add_argument("--runs", default=None,
                    help="comma separated run names; default is every run in the suite")
    ap.add_argument("--bench-limit", type=int, default=100,
                    help="rows per test set; 0 means full size")
    ap.add_argument("--sets",
                    default="data1,aime25,math500,gsm8k,humaneval,mbpp,lcb,mtbench",
                    help="comma separated evaluation sets")
    ap.add_argument("--port", type=int, default=30120)
    args = ap.parse_args()
    if args.runs:
        args.runs = {r.strip() for r in args.runs.split(",") if r.strip()}

    for stage in [s.strip() for s in args.stages.split(",") if s.strip()]:
        if stage not in STAGES:
            sys.exit(f"unknown stage {stage!r}; pick from {sorted(STAGES)}")
        log(f"=== stage: {stage} ===")
        STAGES[stage](args)
    log("pipeline finished")


if __name__ == "__main__":
    main()
