"""Neutralise the broken torchcodec build without breaking transformers.

torchcodec 0.17 dlopens libtorchcodec_image.so at import time and raises
OSError. That escapes sglang's `except (ImportError, RuntimeError)` guard and
kills startup; it also propagates through transformers' audio_utils, which
imports torchcodec eagerly.

Blocking the whole package is wrong -- transformers' `is_torchcodec_available()`
uses importlib.util.find_spec, and a blocker makes importing its audio path
fail, taking AutoProcessor with it. Instead, stub the modules so the *import*
succeeds and the availability probe simply reports the decoders as absent.
"""
import sys
import types

if "torchcodec" not in sys.modules:
    import importlib.machinery

    decoders = types.ModuleType("torchcodec.decoders")
    decoders.AudioDecoder = None
    decoders.VideoDecoder = None
    decoders.__spec__ = importlib.machinery.ModuleSpec("torchcodec.decoders", loader=None)

    pkg = types.ModuleType("torchcodec")
    pkg.decoders = decoders
    pkg.__spec__ = importlib.machinery.ModuleSpec("torchcodec", loader=None)
    pkg.__version__ = "0.0.0+stub"

    sys.modules["torchcodec"] = pkg
    sys.modules["torchcodec.decoders"] = decoders

# Chain to the sglang compat shim (CUDA-13 workarounds for DSPINE/DFLASH).
import os

_shim = str(__import__("pathlib").Path(__file__).resolve().parent.parent / "sglang_compat" / "sitecustomize.py")
if os.path.exists(_shim):
    try:
        with open(_shim, encoding="utf-8") as fh:
            exec(compile(fh.read(), _shim, "exec"),
                 {"__name__": "sglang_compat_shim", "__file__": _shim})
    except Exception as exc:
        print(f"[no_torchcodec] compat shim failed: {exc}", file=sys.stderr)
