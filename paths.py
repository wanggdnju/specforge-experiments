"""Single place where every filesystem location is resolved.

Nothing else in this toolkit hardcodes an absolute path. Change one of the
values below (or set the matching environment variable) and the whole
pipeline moves with it.

Resolution order for each setting:

1. environment variable (see the table below)
2. `local.env` next to this file if present  -- your machine-specific
   overrides, git-ignored, so the repo stays generic
3. the default derived from the repo location

This toolkit drives **SpecForge** but does not vendor it. Two roots are
therefore in play:

* `TOOLKIT` -- this repository (the scripts, configs, docs).
* `SPECFORGE` -- a checkout of the SpecForge framework. Its working directory
  is where `specforge train` runs, so artifact and config paths in the run
  YAMLs are resolved relative to it.

`REPO` is kept as an alias of `SPECFORGE` because the run configs already
write paths relative to the framework checkout.

Environment variables
---------------------
| variable | meaning | default |
| --- | --- | --- |
| SF_TOOLKIT | this repository | the directory holding this file |
| SF_SPECFORGE | SpecForge checkout (framework cwd) | `$SF_TOOLKIT/../SpecForge` |
| SF_ROOT | workspace root holding models/, venvs/, data/ | `$SF_TOOLKIT/..` |
| SF_TARGET_MODEL | target model directory | `$SF_ROOT/model/Qwen3-8B` |
| SF_TOKENIZER | tokenizer (defaults to the target) | $SF_TARGET_MODEL |
| SF_RAW_DIR | raw request dumps (data1..data8) | `$SF_ROOT/data/specforge/raw` |
| SF_DATASET_DIR | processed datasets | `$SF_ROOT/data/specforge` |
| SF_CACHE_DIR | specforge caches | $SF_DATASET_DIR/cache |
| SF_VOCAB_MAPPING | EAGLE3 vocab mapping file | `$SF_DATASET_DIR/vocab_mapping/qwen3-8b-data1.pt` |
| SF_PY_TRAIN | python with specforge installed | `$SF_ROOT/venv-specforge/bin/python` |
| SF_PY_SGLANG | python with SGLang installed | `$SF_ROOT/venv-sglang/bin/python` |
| SF_PY_DSPINE | python with sglang-dspine | `$SF_ROOT/venv-dspine/bin/python` |
| SF_SGLANG_SITE | SGLang site-packages (for CUDA-13 libs) | $SF_PY_SGLANG's site-packages |
| SF_SUITE | suite manifest | `$SF_TOOLKIT/configs/suite.yaml` |

Everything below is derived from those two roots, both found from `__file__`,
so the toolkit can be cloned anywhere.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------- repos itself
TOOLKIT = Path(__file__).resolve().parent            # .../specforge-experiments

# Optional machine-specific overrides, git-ignored.
_LOCAL_ENV = TOOLKIT / "local.env"


def _load_local_env() -> None:
    if not _LOCAL_ENV.exists():
        return
    for line in _LOCAL_ENV.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


_load_local_env()


def _env(name: str, default: str | Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser() if raw else Path(default)


# --------------------------------------------------------------- framework root
# SpecForge runs from its own checkout; every relative path in a run config is
# resolved against it. Default layout puts both repos side by side:
#   <root>/SpecForge               <- $SF_SPECFORGE
#   <root>/specforge-experiments   <- $SF_TOOLKIT
SPECFORGE = _env("SF_SPECFORGE", TOOLKIT.parent / "SpecForge")
REPO = SPECFORGE                                     # alias used by run configs
PIPELINE = TOOLKIT                                   # this toolkit's root


# ------------------------------------------------------------------- workspace
ROOT = _env("SF_ROOT", TOOLKIT.parent)

TARGET_MODEL = _env("SF_TARGET_MODEL", ROOT / "model" / "Qwen3-8B")
TOKENIZER = _env("SF_TOKENIZER", TARGET_MODEL)

RAW_DIR = _env("SF_RAW_DIR", ROOT / "data" / "specforge" / "raw")
DATASET_DIR = _env("SF_DATASET_DIR", ROOT / "data" / "specforge")
CACHE_DIR = _env("SF_CACHE_DIR", DATASET_DIR / "cache")
VOCAB_MAPPING = _env("SF_VOCAB_MAPPING",
                     DATASET_DIR / "vocab_mapping" / "qwen3-8b-data1.pt")

PY_TRAIN = _env("SF_PY_TRAIN", ROOT / "venv-specforge" / "bin" / "python")
PY_SGLANG = _env("SF_PY_SGLANG", ROOT / "venv-sglang" / "bin" / "python")
PY_DSPINE = _env("SF_PY_DSPINE", ROOT / "venv-dspine" / "bin" / "python")

# ---------------------------------------------------- framework-checkout paths
# Artifacts live inside the framework checkout because `specforge train` writes
# relative to its cwd.
ARTIFACTS = REPO / "artifacts"
RUNS = ARTIFACTS / "runs"
CHECKPOINTS = ARTIFACTS / "checkpoints"
DATASETS = ARTIFACTS / "datasets"
PIPELINE_WORK = ARTIFACTS / "pipeline"
CONFIGS_OUT = PIPELINE_WORK / "configs"
LOGS = PIPELINE_WORK / "logs"
REPORTS = PIPELINE_WORK / "reports"
PROMPTS = PIPELINE_WORK / "prompts"

# ------------------------------------------------------------- toolkit paths
SUITE = _env("SF_SUITE", TOOLKIT / "configs" / "suite.yaml")
RUN_CONFIGS = TOOLKIT / "configs" / "runs"
DRAFT_CONFIGS = TOOLKIT / "configs"


def sglang_site_packages() -> Path:
    """Where SGLang's CUDA-13 libraries live (used on LD_LIBRARY_PATH)."""
    override = os.environ.get("SF_SGLANG_SITE")
    if override:
        return Path(override)
    # <venv>/bin/python -> <venv>/lib/pythonX.Y/site-packages
    py = PY_SGLANG
    venv = py.parent.parent
    lib = venv / "lib"
    if lib.is_dir():
        for child in sorted(lib.glob("python3.*")):
            candidate = child / "site-packages"
            if candidate.is_dir():
                return candidate
    return venv / "lib" / "python3.12" / "site-packages"


def shim_dir(name: str) -> Path:
    """Path to one of our own shims (shims/<name>)."""
    return TOOLKIT / "shims" / name


def describe() -> str:
    """Human-readable dump, for debugging a misconfigured machine."""
    rows = [
        ("TOOLKIT", TOOLKIT), ("SPECFORGE", SPECFORGE), ("ROOT", ROOT),
        ("TARGET_MODEL", TARGET_MODEL),
        ("TOKENIZER", TOKENIZER), ("RAW_DIR", RAW_DIR),
        ("DATASET_DIR", DATASET_DIR), ("VOCAB_MAPPING", VOCAB_MAPPING),
        ("PY_TRAIN", PY_TRAIN), ("PY_SGLANG", PY_SGLANG),
        ("PY_DSPINE", PY_DSPINE), ("SUITE", SUITE),
        ("ARTIFACTS", ARTIFACTS),
    ]
    lines = [f"{k:16s} {v}  {'ok' if Path(v).exists() else 'MISSING'}"
             for k, v in rows]
    lines.append(f"{'SGLANG_SITE':16s} {sglang_site_packages()}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys as _sys
    if len(_sys.argv) == 3 and _sys.argv[1] == "--print":
        print(globals()[_sys.argv[2]])
    else:
        print(describe())
