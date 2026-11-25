#!/usr/bin/env bash
# Entry point for the whole pipeline. Each subcommand is a thin wrapper so the
# exact commands are discoverable without reading four README sections.
#
#   ./experiments/run.sh build-data [--per-source N]
#   ./experiments/run.sh train <run|all>
#   ./experiments/run.sh export <run|all>
#   ./experiments/run.sh bench <run|all>
#   ./experiments/run.sh report
#   ./experiments/run.sh pipeline [stages]
#
# Run from anywhere:
#   ./run.sh report
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sf_path() { python3 "$HERE/paths.py" --print "$1"; }
REPO="$(sf_path REPO)"      # SpecForge checkout
ROOT="$(sf_path ROOT)"      # workspace root: model/, data/, venvs/
SUITE="$(sf_path SUITE)"    # this toolkit's suite manifest

# Training needs the venv that has specforge installed; serving needs the one
# with SGLang. Two different interpreters on purpose -- see PITFALLS §20.
PY_TRAIN="$(sf_path PY_TRAIN)"
PY_BENCH="$(sf_path PY_SGLANG)"
PY_DSPINE="$(sf_path PY_DSPINE)"

usage() { sed -n '2,14p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 1; }

cmd="${1:-}"; shift || true

case "$cmd" in
  build-data)
    exec "$PY_TRAIN" -I "$HERE/1_data/build_all8.py" --stage all "$@"
    ;;

  train)
    target="${1:-all}"; shift || true
    if [ "$target" = "all" ]; then
      exec "$PY_TRAIN" "$HERE/2_train/run_pipeline.py" --suite "$SUITE" \
           --stages train "$@"
    fi
    exec "$PY_TRAIN" "$HERE/2_train/run_pipeline.py" --suite "$SUITE" \
         --stages train --runs "$target" "$@"
    ;;

  export)
    target="${1:-all}"; shift || true
    if [ "$target" = "all" ]; then
      exec "$PY_TRAIN" "$HERE/2_train/run_pipeline.py" --suite "$SUITE" \
           --stages export "$@"
    fi
    exec "$PY_TRAIN" "$HERE/2_train/run_pipeline.py" --suite "$SUITE" \
         --stages export --runs "$target" "$@"
    ;;

  bench)
    target="${1:-all}"; shift || true
    if [ "$target" = "all" ]; then
      exec "$HERE/3_bench/run_bench_queue.sh"
    fi
    exec "$HERE/3_bench/run_bench_queue.sh" "$target"
    ;;

  bench-dspine)
    # DSpine runs from its own venv (sglang 0.5.22 + sglang-dspine plugin).
    draft="${1:?usage: run.sh bench-dspine <draft-dir> [out.json]}"
    out="${2:-$REPO/artifacts/pipeline/logs/$(basename "$draft").bench.json}"
    exec "$PY_DSPINE" "$HERE/3_bench/bench_dspine.py" --draft "$draft" --out "$out"
    ;;

  report)
    exec "$PY_TRAIN" "$HERE/2_train/run_pipeline.py" --suite "$SUITE" --stages report
    ;;

  pipeline)
    stages="${1:-configs,train,export,bench,report}"; shift || true
    exec "$PY_TRAIN" "$HERE/2_train/run_pipeline.py" --suite "$SUITE" \
         --stages "$stages" "$@"
    ;;

  *)
    usage
    ;;
esac
