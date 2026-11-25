#!/usr/bin/env bash
# Run a SpecForge *online external* training job on this box.
#
# Why this exists
# ---------------
# SpecForge's online producer talks HTTP to a "spec capture server" and expects
# captured tensors to already be in Mooncake. The official server is SGLang at
# commit c757c4b with the specforge-sglang-capture plugin. That SGLang cannot be
# built here (needs Rust 1.92, and its documented toolchain is CUDA 13 while this
# driver is 12.9).
#
# capture_server.py is a stand-in: a plain PyTorch process that serves the same
# two endpoints and writes the same Mooncake keys. It is NOT the official capture
# path -- it does not reproduce SGLang's radix cache isolation or chunked-prefill
# semantics. Use it to exercise the training pipeline; do not treat its numbers as
# bit-comparable with a real SGLang capture run.
#
# Usage
#   ./run_online.sh <config.yaml> [extra `specforge train` args...]
#
# The YAML must already point at this host's Mooncake:
#   deployment.disaggregated.backend: mooncake
#   mooncake_metadata_server: http://127.0.0.1:35880/metadata
#   mooncake_master_server_addr: 127.0.0.1:35551
#   mooncake_protocol: tcp
#   server_urls: [http://127.0.0.1:30000]
# and it must set model.aux_hidden_state_layer_ids (or inherit EAGLE3's defaults,
# which this script resolves the same way the trainer does).

set -euo pipefail

CONFIG="${1:?usage: run_online.sh <config.yaml> [train args...]}"
shift || true
CONFIG="$(readlink -f "$CONFIG")"

# ---------------------------------------------------------------- paths / env
# Locations come from experiments/paths.py; override with SF_* environment
# variables rather than editing this file.
PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sf_path() { python3 "$PIPELINE_DIR/paths.py" --print "$1"; }
VENV_PY="${SF_VENV_PY:-$(sf_path PY_TRAIN)}"
SPECFORGE_DIR="${SF_REPO:-$(sf_path REPO)}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RPC_PORT="${SF_MC_RPC_PORT:-35551}"
META_PORT="${SF_MC_META_PORT:-35880}"
CAP_PORT="${SF_CAP_PORT:-30000}"
SEGMENT_BYTES="${SF_CAP_SEGMENT_BYTES:-17179869184}"   # 16 GiB
LOG_DIR="${SF_LOG_DIR:-$(pwd)/sf_online_logs}"
mkdir -p "$LOG_DIR"

MC_LOG="$LOG_DIR/mooncake_master.log"
CAP_LOG="$LOG_DIR/capture_server.log"

cleanup() {
    local rc=$?
    [[ -n "${CAP_PID:-}" ]] && kill "$CAP_PID" 2>/dev/null || true
    [[ -n "${MC_PID:-}"  ]] && kill "$MC_PID"  2>/dev/null || true
    # Give Mooncake a moment to unregister its segments.
    wait 2>/dev/null || true
    exit $rc
}
trap cleanup EXIT INT TERM

log() { printf '[run_online] %s\n' "$*" >&2; }

# ------------------------------------------------------------------ preflight
[[ -x "$VENV_PY" ]] || { log "python not found: $VENV_PY"; exit 1; }
command -v mooncake_master >/dev/null || { log "mooncake_master not on PATH"; exit 1; }
"$VENV_PY" -c 'import mooncake' 2>/dev/null || { log "'mooncake' not importable in $VENV_PY"; exit 1; }

# Resolve the capture contract from the run config so the server and the trainer
# can never disagree about which layers to capture or which strategy this is.
read -r MODEL_PATH METHOD LAYERS STORE_ID < <(
    "$VENV_PY" - "$CONFIG" <<'PY'
import sys
from specforge.config import load_config
from specforge.application import resolve_run
from specforge.algorithms.common.providers import resolve_server_capture_layout

cfg = load_config(sys.argv[1], [])
resolved = resolve_run(cfg)
# capture_method lives on the streaming provider, not on the layout.
provider = resolved.algorithm.providers.server_streaming_for("text")
layout = resolve_server_capture_layout(resolved.algorithm, resolved.config, modality="text")
assert layout is not None

# Resolve the layer set through the algorithm's own capture contract instead of
# reading model.aux_hidden_state_layer_ids directly: that field is EAGLE3-shaped
# (exactly three layers), while DFlash defines its five layers in the draft
# config's dflash_config.target_layer_ids. The contract is what the trainer will
# request, so the server must capture precisely that.
from specforge.training.capture_contract import resolve_server_capture_contract

contract = resolve_server_capture_contract(resolved.config, algorithm=resolved.algorithm)
layers = list(contract.aux_layer_ids)
if not layers:
    raise SystemExit(f"algorithm {resolved.algorithm.name!r} requests no aux layers")

store_id = resolved.config.deployment.disaggregated.store_id or resolved.config.run_id
print(
    resolved.config.model.target_model_path,
    provider.capture_method,
    ",".join(str(i) for i in layers),
    store_id,
)
PY
)

log "target       : $MODEL_PATH"
log "capture      : method=$METHOD aux_layers=$LAYERS store_id=$STORE_ID"
log "logs         : $LOG_DIR"

# ------------------------------------------------------------ mooncake master
# `ss`/`netstat`/`lsof` are not installed on this host, so probe the port with
# Python instead of a shell utility.
port_open() {
    "$VENV_PY" - "$1" <<'PY' >/dev/null 2>&1
import socket, sys
s = socket.socket()
s.settimeout(1.0)
sys.exit(0 if s.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0 else 1)
PY
}

if port_open "$RPC_PORT"; then
    log "a mooncake master already listens on ${RPC_PORT}; reusing it"
    MC_PID=""
else
    log "starting mooncake_master (rpc=${RPC_PORT} metadata=${META_PORT})"
    mooncake_master \
        --rpc-port "$RPC_PORT" \
        --enable_http_metadata_server=true \
        --http_metadata_server_port "$META_PORT" \
        > "$MC_LOG" 2>&1 &
    MC_PID=$!
    for _ in $(seq 1 60); do
        if port_open "$RPC_PORT"; then break; fi
        kill -0 "$MC_PID" 2>/dev/null || { log "mooncake_master died; see $MC_LOG"; exit 1; }
        sleep 1
    done
    log "mooncake_master ready (pid $MC_PID)"
fi

# ------------------------------------------------------------- capture server
# The capture server holds a Mooncake *client* segment: the captured bytes live in
# this process's memory. It must stay alive for the whole run.
if port_open "$CAP_PORT"; then
    log "port ${CAP_PORT} is already in use -- refusing to start a second capture server"
    log "(stop whatever is serving it, or set SF_CAP_PORT and update server_urls in $CONFIG)"
    exit 1
fi

log "starting capture_server on 127.0.0.1:${CAP_PORT}"
"$VENV_PY" "$(sf_path TOOLKIT)/serving/capture_server.py" \
    --model-path "$MODEL_PATH" \
    --port "$CAP_PORT" \
    --aux-layers "$LAYERS" \
    --method "$METHOD" \
    --store-id "$STORE_ID" \
    --mooncake-metadata-server "http://127.0.0.1:${META_PORT}/metadata" \
    --mooncake-master-server-addr "127.0.0.1:${RPC_PORT}" \
    --segment-bytes "$SEGMENT_BYTES" \
    --verbose \
    > "$CAP_LOG" 2>&1 &
CAP_PID=$!

for _ in $(seq 1 300); do
    if curl -sf --max-time 2 "http://127.0.0.1:${CAP_PORT}/server_info" >/dev/null 2>&1; then break; fi
    kill -0 "$CAP_PID" 2>/dev/null || { log "capture_server died; see $CAP_LOG"; tail -20 "$CAP_LOG" >&2; exit 1; }
    sleep 2
done
curl -sf --max-time 5 "http://127.0.0.1:${CAP_PORT}/server_info" >/dev/null || {
    log "capture_server never became ready; see $CAP_LOG"; exit 1; }
log "capture_server ready (pid $CAP_PID)"

# -------------------------------------------------------------------- training
log "launching: specforge train --config $CONFIG $*"
cd "$SPECFORGE_DIR"
set +e
"$VENV_PY" -m specforge.cli train --config "$CONFIG" "$@"
RC=$?
set -e
log "training exited with $RC"
exit $RC
