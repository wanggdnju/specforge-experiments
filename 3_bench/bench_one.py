"""Serve one exported draft with SGLang and measure accepted length on the test set.

Algorithm -> SGLang flags, because they are not uniform:

| SpecForge strategy | --speculative-algorithm | draft tokens flag |
| --- | --- | --- |
| dflash | DFLASH | --speculative-num-draft-tokens 16 |
| domino | DFLASH | --speculative-num-draft-tokens 16 (routed by projector_type) |
| dspark | DSPARK | --speculative-dspark-block-size 7 |
| eagle3 | EAGLE3 | --speculative-num-steps / --speculative-eagle-topk |
| peagle | *unsupported* | SGLang 0.5.20 has no PEAGLE enum |

`peagle` is reported as SKIP rather than failing the sweep. Domino works because
SGLang dispatches on the draft config's ``projector_type`` ("domino") inside the
DFLASH worker; the algorithm flag is still DFLASH.

Usage:
    python bench_one.py --draft <dir> --algorithm dflash --out <json>
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
import urllib.error
import urllib.request

SP = str(_p.sglang_site_packages())
TARGET = str(_p.TARGET_MODEL)
WORLD = str(_p.DATASETS / "v4-data1" / "test.jsonl")
COMPAT = str(_p.shim_dir("sglang_compat"))

# How each strategy is presented to SGLang.
SERVING = {
    "dflash":  dict(algo="DFLASH", extra=["--speculative-num-draft-tokens", "16"]),
    # DFlash2 runs on the DFLASH worker; its block size is 8, not 16.
    "dflash2": dict(algo="DFLASH", extra=["--speculative-num-draft-tokens", "8"]),
    "domino":  dict(algo="DFLASH", extra=["--speculative-num-draft-tokens", "16"]),
    "dspark":  dict(algo="DSPARK", extra=[]),   # block size comes from the draft config
    # DSpine trains through the dspark path in SpecForge, so serve it the same way.
    "dspine":  dict(algo="DSPARK", extra=["--speculative-dspark-block-size", "16"]),
    "eagle3":  dict(algo="EAGLE3", extra=["--speculative-num-steps", "4",
                                          "--speculative-eagle-topk", "2",
                                          "--speculative-num-draft-tokens", "16"]),
    # PEAGLE rides the EAGLE3 path; its draft config is rewritten to the EAGLE3
    # architecture shape in prepare_peagle_config().
    "peagle":  dict(algo="EAGLE3", extra=["--speculative-num-steps", "4",
                                          "--speculative-eagle-topk", "2",
                                          "--speculative-num-draft-tokens", "16"]),
}
# PEAGLE is NOT unsupported: PEagleDraftModel subclasses Eagle3DraftModel, and
# SGLang's EAGLE3 loader can serve it once the draft's config is rewritten
# (see prepare_peagle_config). It rides the EAGLE3 algorithm name.
UNSUPPORTED = {
    # dspine is served by the separate sglang-dspine plugin, which requires its
    # own venv (sglang 0.5.22); it is exercised by bench_dspine.py, not here.
    "dspine": "served by the sglang-dspine plugin in venv-dspine (see bench_dspine.py)",
}


def prepare_peagle_config(draft_dir):
    """Rewrite a PEagle draft into the shape SGLang's EAGLE3 loader expects.

    SGLang registers only `LlamaForCausalLMEagle3` and reads the FC input dim
    from `target_hidden_size`; a PEagle checkpoint ships `PEagleDraftModel`
    with `target_hidden_size: null`. Both are pure metadata differences -- the
    tensors are identical (`fc.weight` is [hidden, 3*target_hidden], matching
    num_aux_hidden_states=3) -- so a sidecar directory with a patched config is
    enough. The weights are symlinked, never copied or modified.
    """
    import json
    import os
    import shutil

    src = os.path.abspath(draft_dir)
    with open(os.path.join(src, "config.json"), encoding="utf-8") as fh:
        c = json.load(fh)
    if c.get("architectures") == ["LlamaForCausalLMEagle3"] and c.get("target_hidden_size"):
        return src  # already prepared

    dst = src.rstrip("/") + "-eagle3"
    os.makedirs(dst, exist_ok=True)
    c["architectures"] = ["LlamaForCausalLMEagle3"]
    c["model_type"] = "llama"
    c.pop("auto_map", None)
    c["target_hidden_size"] = c.get("target_hidden_size") or c.get("hidden_size") or 4096
    aux = c.get("eagle_aux_hidden_state_layer_ids") or [2, 18, 33]
    c["eagle_aux_hidden_state_layer_ids"] = aux
    c["eagle_config"] = {"eagle_aux_hidden_state_layer_ids": aux}
    c["num_aux_hidden_states"] = len(aux)
    n_layers = c.get("num_hidden_layers") or c.get("num_depths") or 7
    c["num_hidden_layers"] = n_layers
    c.setdefault("max_position_embeddings", 40960)
    for k, v in (("hidden_size", 4096), ("intermediate_size", 12288),
                 ("num_attention_heads", 32), ("num_key_value_heads", 8),
                 ("head_dim", 128), ("rms_norm_eps", 1e-6),
                 ("vocab_size", 151936), ("rope_theta", 1000000),
                 ("tie_word_embeddings", False), ("draft_vocab_size", 151936)):
        c.setdefault(k, v)
    with open(os.path.join(dst, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(c, fh, indent=2, ensure_ascii=False)

    for name in os.listdir(src):
        if name.endswith((".safetensors", ".bin", ".py")):
            target = os.path.join(dst, name)
            if not os.path.exists(target):
                os.symlink(os.path.join(src, name), target)
    return dst


def serve(draft, algorithm, port, log_path):
    cfg = SERVING[algorithm]
    env = os.environ.copy()
    env["PYTHONPATH"] = COMPAT
    env["SGLANG_COMPAT_HIDE_SPEC_SAMPLING"] = "1"
    env["LD_LIBRARY_PATH"] = f"{SP}/nvidia/cu13/lib:{SP}/torch/lib"
    # Domino's GRU head cannot initialise cuDNN on this box; the shim turns
    # cuDNN off inside the server so torch uses its native RNN kernel.
    if algorithm == "domino":
        env["SF_DISABLE_CUDNN"] = "1"
        env["PYTHONPATH"] = (f"{_p.shim_dir('cudnn_off')}:"
                             + env.get("PYTHONPATH", ""))
    # EAGLE's tree-verify ops load fine but are silent no-ops on CUDA 12.9, and
    # the fallback path faults with "Triton Error [CUDA]: an illegal memory
    # access". The compat shim ships Triton replacements for exactly those two
    # ops; this switch installs them.
    if algorithm in ("eagle3", "peagle"):
        env["SGLANG_COMPAT_TRITON_TREE"] = "1"
    cmd = [
        str(_p.PY_SGLANG), "-m", "sglang.launch_server",
        "--model-path", TARGET,
        "--speculative-algorithm", cfg["algo"],
        "--speculative-draft-model-path", draft,
        *cfg["extra"],
        "--mem-fraction-static", "0.6",
        # domino's GRU correction makes decode-graph capture fail on this card:
        #   "Capture cuda graph failed: cuDNN error: CUDNN_STATUS_NOT_INITIALIZED"
        # Lowering mem-fraction did not help. Decode graphs are a throughput
        # optimisation only -- acceptance length is identical without them -- so
        # disable capture for domino rather than fight the allocator.
        *(["--disable-cuda-graph"] if algorithm == "domino" else []),
        "--port", str(port), "--host", "127.0.0.1",
    ]
    fh = open(log_path, "w")
    # start_new_session so a Ctrl-C on the harness does not orphan the server
    # into the next run holding the GPU.
    proc = subprocess.Popen(cmd, cwd="/tmp", env=env, stdout=fh,
                            stderr=subprocess.STDOUT, start_new_session=True)
    return proc, fh


def wait_health(port, proc, log_path, timeout_s=1800):
    url = f"http://127.0.0.1:{port}/health"
    start = time.time()
    while time.time() - start < timeout_s:
        if proc.poll() is not None:
            tail = open(log_path, errors="replace").read()[-2500:]
            raise SystemExit(f"server died during startup:\n{tail}")
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                if r.status == 200:
                    return time.time() - start
        except Exception:
            pass
        time.sleep(5)
    raise SystemExit(f"server never became healthy after {timeout_s}s")


def stop(proc, port):
    """Terminate the server and wait for the GPU to actually free up."""
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
    # sglang spawns children; poll the card rather than trusting the pid.
    for _ in range(60):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=20).stdout.strip()
            if out and int(out.splitlines()[0]) < 2000:
                return
        except Exception:
            pass
        time.sleep(5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", required=True)
    ap.add_argument("--algorithm", required=True, choices=sorted(SERVING) + sorted(UNSUPPORTED))
    ap.add_argument("--port", type=int, default=30120)
    ap.add_argument("--limit", type=int, default=100,
                    help="per-set cap; 0 means use each set's full size")
    ap.add_argument("--sets", default="data1,aime25,math500,gsm8k,humaneval,mbpp,lcb,mtbench",
                    help="comma separated evaluation sets")
    ap.add_argument("--sglang-algo", default=None,
                    help="override the SGLang algorithm (from the suite manifest)")
    ap.add_argument("--draft-tokens", type=int, default=None,
                    help="override the proposal block size (from the suite manifest)")
    ap.add_argument("--prompt-dir", default="/tmp/sf_prompts")
    ap.add_argument("--out", required=True, help="path for the result JSON")
    ap.add_argument("--log", default="/tmp/bench_one_server.log")
    args = ap.parse_args()

    def write(payload):
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(json.dumps(payload, indent=2))

    if args.algorithm in UNSUPPORTED:
        write({"algorithm": args.algorithm, "status": "SKIP",
               "reason": UNSUPPORTED[args.algorithm]})
        return

    # The server is launched with cwd=/tmp (see serve()), so a relative draft
    # path would be resolved against the wrong directory and transformers would
    # read it as a HuggingFace repo id ("Repo id must be in the form ...").
    args.draft = os.path.abspath(args.draft)

    if not os.path.isdir(args.draft):
        write({"algorithm": args.algorithm, "status": "SKIP",
               "reason": f"draft directory missing: {args.draft}"})
        return

    # PEAGLE drafts are served through SGLang's EAGLE3 loader after a metadata
    # rewrite of the config (same tensors, different arch string).
    if args.algorithm == "peagle":
        args.draft = prepare_peagle_config(args.draft)

    # The suite manifest can pin the SGLang algorithm and block size for a run;
    # honour those over the table above so the manifest stays authoritative.
    if args.sglang_algo is not None:
        # Override the algorithm name and block size only. The rest of the
        # family's required flags (eagle3 needs --speculative-num-steps and
        # --speculative-eagle-topk; omitting them trips an AssertionError in
        # SGLang's arg resolution) must survive.
        base = SERVING.get(args.algorithm, {"extra": []})
        # Drop the old flag *and its value*. Filtering only the "--flag" token
        # left a bare "16" in the list, which sglang rejected with
        #   "unrecognized arguments: 16".
        extra, skip = [], False
        for tok in base["extra"]:
            if skip:
                skip = False
                continue
            if tok == "--speculative-num-draft-tokens":
                skip = True
                continue
            extra.append(tok)
        if args.draft_tokens:
            extra += ["--speculative-num-draft-tokens", str(args.draft_tokens)]
        SERVING[args.algorithm] = {"algo": args.sglang_algo.upper(), "extra": extra}

    proc = fh = None
    try:
        proc, fh = serve(args.draft, args.algorithm, args.port, args.log)
        took = wait_health(args.port, proc, args.log)
        print(f"[bench] healthy after {took:.0f}s", file=sys.stderr)

        # One server, every test set: loading the target+draft costs ~2-4 min,
        # so reusing the server across sets is what makes the 8-set sweep
        # affordable. Each set is rendered to chat prompts first (load_prompts).
        results = {}
        for set_name in [s.strip() for s in args.sets.split(",") if s.strip()]:
            prompt_file = os.path.join(args.prompt_dir, f"{set_name}.jsonl")
            gen = [sys.executable, os.path.join(os.path.dirname(__file__), "load_prompts.py"),
                   "--set", set_name, "--out", prompt_file]
            if args.limit > 0:
                gen += ["--limit", str(args.limit)]
            g = subprocess.run(gen, capture_output=True, text=True, timeout=1800)
            if g.returncode != 0:
                results[set_name] = {"status": "FAIL",
                                     "reason": (g.stderr or g.stdout)[-600:]}
                continue

            cmd = [str(_p.PY_SGLANG),
                   os.path.join(os.path.dirname(__file__), "measure_tau.py"),
                   "--port", str(args.port), "--prompts", prompt_file,
                   "--limit", str(len(open(prompt_file, encoding="utf-8").readlines())),
                   "--label", f"{args.algorithm}:{set_name}"]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            if r.returncode != 0:
                results[set_name] = {"status": "FAIL",
                                     "reason": (r.stderr or r.stdout)[-600:]}
                continue
            m = json.loads(r.stdout.strip().splitlines()[-1])
            m["status"] = "OK"
            results[set_name] = m
            print(f"[bench] {set_name}: tau={m.get('tau_mean')} n={m.get('n')}",
                  file=sys.stderr)

        write({"algorithm": args.algorithm, "status": "OK",
               "draft": args.draft, "sets": results})
    except SystemExit as exc:
        write({"algorithm": args.algorithm, "status": "FAIL", "reason": str(exc)[-1500:]})
    finally:
        if proc is not None:
            stop(proc, args.port)
        if fh is not None:
            fh.close()


if __name__ == "__main__":
    main()
