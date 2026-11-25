#!/usr/bin/env bash
# Sequential benchmark runner with hard per-model timeouts.
#
# Why a shell driver instead of one python process: a broken server can hang
# forever waiting on /health, and a hung SGLang keeps the GPU allocated. Each
# model therefore runs under `timeout`, and any surviving sglang process is
# killed between models so the next one starts from a clean card.
#
# Every step is logged and echoed, so progress is visible without polling a
# background job.
#
# Usage:
#   ./run_bench_queue.sh [run1 run2 ...]      # default: every ready checkpoint

# Resolve every location from experiments/paths.py so nothing here hardcodes
# an absolute path. Override any value with an environment variable of the same
# name (see the table in paths.py).
PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sf_path() { python3 "$PIPELINE_DIR/paths.py" --print "$1"; }
set -u

REPO="$(sf_path REPO)"
PIPE=$REPO/artifacts/pipeline
LOGS=$PIPE/logs
CKPT=$REPO/artifacts/checkpoints
PY="$(sf_path PY_SGLANG)"
BENCH="$(sf_path TOOLKIT)/3_bench/bench_one.py"
SUITE="$(sf_path SUITE)"

# model timeout: 8 sets x ~100 rows, plus ~4 min server load
PER_MODEL_TIMEOUT=${PER_MODEL_TIMEOUT:-2400}

mkdir -p "$LOGS"

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$LOGS/queue.log"; }

free_gpu() {
    local used
    for _ in $(seq 1 40); do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
        [ "${used:-0}" -lt 2000 ] && return 0
        sleep 5
    done
    log "WARNING: GPU still holds ${used} MiB"
}

# Resolve algorithm / sglang params for a run from the suite manifest.
suite_field() {
    python3 - "$SUITE" "$1" "$2" <<'PY'
import sys, yaml
suite_path, run_name, field = sys.argv[1], sys.argv[2], sys.argv[3]
for r in yaml.safe_load(open(suite_path, encoding="utf-8"))["runs"]:
    if r["name"] == run_name:
        v = r.get(field)
        print("" if v is None else v)
        break
PY
}

RUNS=("$@")
if [ ${#RUNS[@]} -eq 0 ]; then
    # Default: every run the suite knows about, in manifest order.
    mapfile -t RUNS < <(python3 - "$SUITE" <<'PY'
import sys, yaml
for r in yaml.safe_load(open(sys.argv[1], encoding="utf-8"))["runs"]:
    print(r["name"])
PY
)
fi

log "queue start: ${#RUNS[@]} models"

for name in "${RUNS[@]}"; do
    # `draft` in the manifest means "bench this directory as-is" -- used by the
    # pretrained baselines, which are never trained or exported by this harness.
    # Otherwise the draft is the HF export of this run's training output.
    draft=$(suite_field "$name" draft)
    [ -z "$draft" ] && draft="$CKPT/$name"
    algo=$(suite_field "$name" algorithm)
    sgl_algo=$(suite_field "$name" sglang_algo)
    tokens=$(suite_field "$name" draft_tokens)

    if [ ! -d "$draft" ]; then
        log "SKIP $name: no checkpoint at $draft"
        continue
    fi
    if [ -f "$LOGS/$name.bench.done" ]; then
        if python3 -c "import json,sys;d=json.load(open('$LOGS/$name.bench.json'));sys.exit(0 if d.get('status')=='OK' and any(v.get('status')=='OK' for v in (d.get('sets') or {}).values()) else 1)" 2>/dev/null; then
            log "SKIP $name: already benchmarked"
            continue
        fi
        rm -f "$LOGS/$name.bench.done"
    fi

    log "START $name (algo=$algo sglang=$sgl_algo tokens=$tokens)"

    # Stale servers from a previous model would grab the port.
    pkill -9 -f "sglang.launch_server" 2>/dev/null
    sleep 5
    free_gpu

    # Build the optional flags as an array. `${var:+--flag "$var"}` unquoted
    # word-splits, which turned `--draft-tokens 16` into a stray positional
    # `16` and made sglang exit with "unrecognized arguments: 16".
    EXTRA=()
    [ -n "$sgl_algo" ] && EXTRA+=(--sglang-algo "$sgl_algo")
    [ -n "$tokens" ] && EXTRA+=(--draft-tokens "$tokens")

    timeout -k 30 "$PER_MODEL_TIMEOUT" python3 "$BENCH" \
        --draft "$draft" \
        --algorithm "$algo" \
        "${EXTRA[@]}" \
        --port 30130 \
        --out "$LOGS/$name.bench.json" \
        --log "$LOGS/$name.server.log" \
        2>&1 | tee -a "$LOGS/queue.log"

    rc=$?
    if [ $rc -eq 124 ]; then
        log "TIMEOUT $name after ${PER_MODEL_TIMEOUT}s"
    fi

    pkill -9 -f "sglang.launch_server" 2>/dev/null
    sleep 10
    free_gpu

    # NOTE: run_pipeline's fallback report also writes to queue.log; this check
    # must not swallow its output, so the python -c writes to stderr only.
    if python3 -c "
import json,sys
try: d=json.load(open('$LOGS/$name.bench.json'))
except Exception: sys.exit(1)
sets=(d.get('sets') or {})
ok=[s for s,v in sets.items() if v.get('status')=='OK']
print('  -> sets ok:',len(ok),'tau:',{s:round(v['tau_mean'],3) for s,v in sets.items() if v.get('status')=='OK'})
sys.exit(0 if ok else 1)"; then
        touch "$LOGS/$name.bench.done"
        log "DONE $name"
    else
        log "FAILED $name (see $LOGS/$name.server.log)"
    fi
done

log "queue finished"
python3 "$(sf_path TOOLKIT)/2_train/run_pipeline.py" --suite "$SUITE" --stages report 2>&1 | tail -40 | tee -a "$LOGS/queue.log"
