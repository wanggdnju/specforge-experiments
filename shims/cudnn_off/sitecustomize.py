"""Pre-torch setup for the Domino draft server.

Two problems must be handled before the sglang server imports torch:

1. Domino's GRU correction head calls `torch.nn.GRU`, and on this box cuDNN 9.2
   reports itself available but fails to initialise
   ("cuDNN error: CUDNN_STATUS_NOT_INITIALIZED"). Disabling cuDNN makes torch
   use its native RNN kernel, which gives identical results.

2. The sglang compat shim (../sglang_compat/sitecustomize.py) must still run:
   Python imports only ONE `sitecustomize` module, so putting this directory
   ahead of it on PYTHONPATH would otherwise shadow the shim entirely. We chain
   to it explicitly.

Enabled by SF_DISABLE_CUDNN=1 plus this directory on PYTHONPATH, for the domino
server only.
"""
import os
import sys

if os.environ.get("SF_DISABLE_CUDNN") == "1":
    try:
        import torch
        torch.backends.cudnn.enabled = False
    except Exception:
        pass

# Chain to the real compat shim rather than shadowing it.
_shim = str(__import__("pathlib").Path(__file__).resolve().parent.parent / "sglang_compat" / "sitecustomize.py")
if os.path.exists(_shim):
    try:
        with open(_shim, encoding="utf-8") as fh:
            code = compile(fh.read(), _shim, "exec")
        exec(code, {"__name__": "sglang_compat_sitecustomize",
                    "__file__": _shim})
    except Exception as exc:  # pragma: no cover - diagnostics only
        print(f"[cudnn_off] compat shim failed: {exc}", file=sys.stderr)
