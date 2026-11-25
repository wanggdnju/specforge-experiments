"""Benchmark DSpine drafts on the same 8 evaluation sets as the other algorithms.

DSpine needs its own environment: it is an SGLang *plugin*
(`sglang-dspine 0.2.0`) pinned to sglang 0.5.22 and torch 2.14.1, which cannot
co-exist with the 0.5.20 venv used for every other algorithm. So this script
launches the server from `venv-dspine` and the measurement client from the same
interpreter; only the prompt files and result format are shared.

    python 3_bench/bench_dspine.py --draft <dir> --out <json>
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
import signal
import subprocess
import sys
import time
import urllib.request

PY = str(_p.PY_DSPINE)
TARGET = str(_p.TARGET_MODEL)
SP = str(_p.PY_DSPINE.parent.parent / "lib" / "python3.12" / "site-packages")
MEASURE = str(_p.PIPELINE / "3_bench" / "measure_tau.py")
PROMPT_DIR = "/tmp/sf_prompts_dspine"

SETS = ["data1", "aime25", "math500", "gsm8k", "humaneval", "mbpp", "lcb", "mtbench"]
LIMITS = {"data1": 100, "aime25": 30, "math500": 100, "gsm8k": 100,
          "humaneval": 100, "mbpp": 100, "lcb": 100, "mtbench": 80}


def serve(draft, port, log_path):
    env = os.environ.copy()
    # sglang-dspine registers DSPINE; it needs the same CUDA-13 shim the other
    # servers use, plus the plugin package visible on PYTHONPATH.
    # no_torchcodec first (its broken build raises OSError that sglang does not
    # catch), and it chains to the sglang_compat shim for the CUDA-13 fixes.
    env["PYTHONPATH"] = str(_p.shim_dir("no_torchcodec"))
    env["SGLANG_COMPAT_HIDE_SPEC_SAMPLING"] = "1"
    env["LD_LIBRARY_PATH"] = f"{SP}/nvidia/cu13/lib:{SP}/torch/lib"
    cmd = [
        PY, "-m", "sglang.launch_server",
        "--model-path", TARGET,
        "--speculative-algorithm", "DSPINE",
        "--speculative-draft-model-path", draft,
        "--speculative-num-draft-tokens", "16",
        "--tp-size", "1",
        "--dtype", "bfloat16",
        "--page-size", "1",
        "--mem-fraction-static", "0.65",
        "--context-length", "8192",
        "--max-running-requests", "64",
        "--port", str(port), "--host", "127.0.0.1",
    ]
    fh = open(log_path, "w")
    proc = subprocess.Popen(cmd, cwd="/tmp", env=env, stdout=fh,
                            stderr=subprocess.STDOUT, start_new_session=True)
    return proc, fh


def wait_health(port, proc, log_path, timeout_s=1800):
    url = f"http://127.0.0.1:{port}/health"
    start = time.time()
    while time.time() - start < timeout_s:
        if proc.poll() is not None:
            tail = open(log_path, errors="replace").read()[-3000:]
            raise SystemExit(f"dspine server died:\n{tail}")
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                if r.status == 200:
                    return time.time() - start
        except Exception:
            pass
        time.sleep(5)
    raise SystemExit("dspine server never became healthy")


def stop(proc):
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        pass
    try:
        proc.wait(timeout=60)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass
    for _ in range(40):
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True).stdout.strip()
        if out and int(out.splitlines()[0]) < 2000:
            return
        time.sleep(5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", required=True)
    ap.add_argument("--port", type=int, default=30150)
    ap.add_argument("--out", required=True)
    ap.add_argument("--log", default="/tmp/bench_dspine_server.log")
    args = ap.parse_args()

    os.makedirs(PROMPT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    def write(payload):
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(json.dumps(payload, indent=2))

    if not os.path.isdir(args.draft):
        write({"algorithm": "dspine", "status": "SKIP",
               "reason": f"draft missing: {args.draft}"})
        return

    proc = fh = None
    try:
        proc, fh = serve(os.path.abspath(args.draft), args.port, args.log)
        took = wait_health(args.port, proc, args.log)
        print(f"[dspine] healthy after {took:.0f}s", file=sys.stderr)

        results = {}
        for name in SETS:
            pf = os.path.join(PROMPT_DIR, f"{name}.jsonl")
            gen = [sys.executable, os.path.join(os.path.dirname(__file__), "load_prompts.py"),
                   "--set", name, "--out", pf, "--limit", str(LIMITS[name])]
            g = subprocess.run(gen, capture_output=True, text=True)
            if g.returncode != 0:
                results[name] = {"status": "FAIL", "reason": g.stderr[-500:]}
                continue
            r = subprocess.run(
                [sys.executable, MEASURE, "--port", str(args.port),
                 "--prompts", pf, "--limit", str(LIMITS[name]),
                 "--label", f"dspine:{name}"],
                capture_output=True, text=True, timeout=3600)
            if r.returncode != 0:
                results[name] = {"status": "FAIL", "reason": (r.stderr or r.stdout)[-500:]}
                continue
            m = json.loads(r.stdout.strip().splitlines()[-1])
            m["status"] = "OK"
            results[name] = m
            print(f"[dspine] {name}: tau={m.get('tau_mean')}", file=sys.stderr)

        write({"algorithm": "dspine", "status": "OK",
               "draft": args.draft, "sets": results})
    except SystemExit as exc:
        write({"algorithm": "dspine", "status": "FAIL", "reason": str(exc)[-2000:]})
    finally:
        if proc is not None:
            stop(proc)
        if fh is not None:
            fh.close()


if __name__ == "__main__":
    main()
