"""Make sglang fall back to its greedy verify path on this box.

Why this exists
---------------
sglang 0.5.20's DFLASH worker calls

    tree_speculative_sampling_target_only(...)

from `sgl_kernel.speculative` for *non-greedy* verification. That op is built
against CUDA 13 (`libcudart.so.13` / `libnvrtc.so.13`) and this driver is
550.54.15, i.e. CUDA 12.9. Importing sgl_kernel works once the venv's
`nvidia/cu13/lib` is on LD_LIBRARY_PATH, and ordinary kernel ops do run -- but
this particular op aborts at launch with

    TreeSpeculativeSamplingTargetOnly failed with error code
    CUDA driver version is insufficient for CUDA runtime version

Every one of our benchmarks is temperature=0, i.e. greedy, which sglang routes
through a separate pure-PyTorch path (`compute_dflash_correct_drafts_and_bonus`)
that never touches that kernel. The *only* reason the server dies at startup is
`prewarm_sampling()`, which unconditionally rehearses the non-greedy path.

sglang already handles the absent-kernel case: dflash_utils sets
`_DFLASH_SAMPLING_VERIFY_AVAILABLE = False` when the import fails, and then
  - prewarm_sampling returns instead of rehearsing, and
  - _validate_phase1_sampling_support logs
      "falling back to greedy argmax verification ...
       use temperature=0 and top_k=1 for lossless greedy decoding."
So the documented fallback is exactly our regime. This module does not add a
new code path -- it makes the existing one select itself.

What it does NOT do
-------------------
It hides one name from `sgl_kernel.speculative` rather than failing the whole
package, so flash-attention, sampling and every other kernel still load. It
does not modify the venv. Nothing here is used by a non-greedy request: if one
ever arrives, sglang takes its own `is_dflash_sampling_verify_available()`
branch and would raise, which is the correct outcome -- better a loud error than
a silently wrong sampling distribution.

Loaded via PYTHONPATH=<this dir>, since `sitecustomize` is imported automatically
at interpreter start.
"""
import importlib
import os

# ---------------------------------------------------------------------------
# ceph-metadata workaround -- ported from
# code/sd_test/scripts/bench/sglang_compat/sitecustomize.py
#
# This module is the one run_eagle3_tree.sh actually loads (it is the only dir
# that also has triton_tree.py) and it did NOT carry this, so a window-60 server
# launch wedged in uninterruptible D state on ceph_mdsc_do_request and never
# became healthy. The two behaviours are both needed on this box, so they have
# to live in one shim rather than two directories.
#
# `packages_distributions()` stats every RECORD entry over ceph and can block
# forever (SIGALRM does not fire in D state); sglang calls it at import time.
# Replaced by a top_level.txt mapping, cached next to this file and keyed on the
# site-packages mtimes. Gated, and independent of the other switches here.
# ---------------------------------------------------------------------------
if os.environ.get("SGLANG_COMPAT_FAST_METADATA") == "1":
    import importlib.metadata as _md

    def _packages_distributions(_distributions=_md.distributions):
        out = {}
        for _dist in _distributions():
            try:
                _name = _dist.metadata["Name"]
            except Exception:
                continue
            if not _name:
                continue
            try:
                _tl = _dist.read_text("top_level.txt")
            except Exception:
                _tl = None
            _tops = [x.strip() for x in (_tl or "").splitlines() if x.strip()]
            if not _tops:
                # `files` merely parses RECORD paths; it is skip_missing_files()
                # that stats them and wedges.
                try:
                    _tops = [f.parts[0] if len(f.parts) > 1 else f.stem
                             for f in (_dist.files or [])]
                except Exception:
                    continue
            for _t in _tops:
                if _t and "." not in _t:
                    out.setdefault(_t, [])
                    if _name not in out[_t]:
                        out[_t].append(_name)
        return out

    import json as _json
    import sys as _sys

    _cache_path = os.environ.get("SGLANG_COMPAT_METADATA_CACHE") or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), ".pkgdist_cache.json")

    def _cache_key():
        seen = []
        for _p in _sys.path:
            if _p.endswith("site-packages") and not any(
                    s[0] == _p for s in seen):
                try:
                    seen.append([_p, os.stat(_p).st_mtime])
                except OSError:
                    seen.append([_p, 0])
        return seen

    def _cached_packages_distributions(_build=_packages_distributions,
                                       _path=_cache_path, _refresh=(
                                           os.environ.get(
                                               "SGLANG_COMPAT_METADATA_CACHE_REFRESH") == "1")):
        _key = _cache_key()
        if not _refresh:
            try:
                with open(_path) as _fh:
                    _blob = _json.load(_fh)
                if _blob.get("key") == _key:
                    return _blob["map"]
            except (OSError, ValueError, KeyError):
                pass
        _mapping = _build()
        try:
            _tmp = _path + ".tmp"
            with open(_tmp, "w") as _fh:
                _json.dump({"key": _key, "map": _mapping}, _fh)
            os.replace(_tmp, _path)
        except OSError:
            pass
        return _mapping

    _md.packages_distributions = _cached_packages_distributions
    del _md

_HIDDEN = ("tree_speculative_sampling_target_only",)

if os.environ.get("SGLANG_COMPAT_HIDE_SPEC_SAMPLING") == "1":
    try:
        import sgl_kernel.speculative as _spec
    except Exception:
        _spec = None
    if _spec is not None:
        for _name in _HIDDEN:
            # delattr on a module is enough: `from sgl_kernel import x` and
            # `getattr(sgl_kernel, x, None)` both fail afterwards, which is what
            # the `except Exception` in dflash_utils keys on.
            if hasattr(_spec, _name):
                delattr(_spec, _name)
        # dflash_utils does `from sgl_kernel import tree_speculative_...`, i.e.
        # the name is re-exported from the package __init__, not only the
        # submodule. Remove it there too.
        try:
            import sgl_kernel as _sk
            for _name in _HIDDEN:
                if hasattr(_sk, _name):
                    delattr(_sk, _name)
        except Exception:
            pass
        del importlib


# Second, independent CUDA-13 problem: the EAGLE tree ops load fine and raise
# nothing, but write nothing either (see triton_tree.py for the measurements).
# That is a silent no-op, so it needs a different fix -- route those two ops to
# sglang's own Triton kernels, which JIT for the real device.
if os.environ.get("SGLANG_COMPAT_TRITON_TREE") == "1":
    try:
        import triton_tree
        triton_tree._install()
    except Exception as _e:
        import sys
        print(f"[sglang_compat] triton_tree import failed: {_e}",
              file=sys.stderr)
