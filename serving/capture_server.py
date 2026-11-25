"""A minimal stand-in for the SGLang spec-capture server.

SpecForge's online producer only needs two things from the thing at
``server_urls[0]``:

1. ``GET /server_info`` advertising a plugin capture backend and the capture
   contract (method + aux layer ids).
2. ``POST /generate`` accepting a list of ``{"input_ids": [...]}`` plus a
   ``spec_capture`` JSON blob in ``sampling_params[i].custom_params``, running a
   full prefill of the target, and returning per-request ``meta_info.spec_capture``
   with key/shape/dtype metadata.

Tensors never travel over HTTP: they are written straight into the Mooncake
store under SpecForge's key layout ``{store_id}/{sample_id}/g{gen}/{name}``.

This process therefore replaces "compile SGLang at commit c757c4b" with "run
transformers in a plain Python process". It is NOT the official capture path:
the official plugin also guarantees cache isolation, chunked-prefill-size -1,
and post-norm hidden-state semantics discovered inside the engine. This
implementation reproduces those semantics explicitly and verifies them.

The process must stay alive while the trainer runs: the Mooncake segment it
contributes is what actually holds the captured bytes.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

STATE = {}


# --------------------------------------------------------------------------
# Mooncake sink
# --------------------------------------------------------------------------
class MooncakeSink:
    """Raw-byte publisher matching MooncakeFeatureStore's layout."""

    def __init__(self, metadata_server, master_addr, segment_bytes, protocol="tcp"):
        from mooncake.store import MooncakeDistributedStore

        self.store = MooncakeDistributedStore()
        rc = self.store.setup(
            global_segment_size=segment_bytes,
            local_buffer_size=64 << 20,
            protocol=protocol,
            rdma_devices="",
            local_hostname="127.0.0.1",
            metadata_server=metadata_server,
            master_server_addr=master_addr,
        )
        if rc != 0:
            raise RuntimeError(f"mooncake setup failed (status {rc})")

    def put(self, key, tensor):
        assert tensor.is_contiguous(), key
        nb = tensor.numel() * tensor.element_size()
        self.store.register_buffer(tensor.data_ptr(), nb)
        try:
            rc = self.store.put_from(key, tensor.data_ptr(), nb)
        finally:
            self.store.unregister_buffer(tensor.data_ptr())
        if rc is not None and int(rc) < 0:
            raise RuntimeError(f"mooncake put_from failed (status {rc}) for {key}")

    def remove(self, key):
        try:
            self.store.remove(key)
        except Exception:
            pass


# --------------------------------------------------------------------------
# Target forward with hidden-state capture
# --------------------------------------------------------------------------
DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}
TORCH_TO_NAME = {v: k for k, v in DTYPE_MAP.items()}


def load_target(model_path, dtype):
    config = AutoConfig.from_pretrained(model_path)
    # SGLang's aux capture returns post-norm states out of each decoder layer.
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=DTYPE_MAP[dtype], device_map="cuda:0"
    ).eval()
    return config, model


@torch.inference_mode()
def capture_one(input_ids, aux_layers, want_last, dtype):
    """Run one full prefill and return the requested hidden states.

    Returns ``{name: tensor[1, L, width]}`` where ``aux`` is the concatenation
    of the aux layers in the requested order and ``last_hidden`` is the
    post-final-norm state.
    """
    model = STATE["model"]
    device = next(model.parameters()).device
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    out = model.model(
        input_ids=ids,
        use_cache=False,
        output_hidden_states=True,
    )
    # hidden_states[0] is the embedding output; hidden_states[i+1] is the output
    # of decoder layer i, i.e. post-norm for Qwen3's pre-norm blocks.
    hs = out.hidden_states
    aux = torch.cat([hs[i + 1] for i in aux_layers], dim=-1)
    result = {"aux": aux.to(DTYPE_MAP[dtype]).contiguous()}
    if want_last:
        result["last_hidden"] = out.last_hidden_state.to(DTYPE_MAP[dtype]).contiguous()
    return result


def dtype_name(t):
    return TORCH_TO_NAME.get(t.dtype, str(t.dtype).replace("torch.", ""))


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------
def make_handler(cfg):
    aux_layers = tuple(cfg["aux_layers"])

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            if cfg["verbose"]:
                print(f"[capture-server] {fmt % args}", flush=True)

        def _send(self, payload, status=200):
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.rstrip("/") != "/server_info":
                self._send({"error": "not found"}, 404)
                return
            self._send(
                {
                    "model_path": cfg["model_path"],
                    "internal_states": [
                        {
                            "spec_capture_backend": "plugin",
                            "spec_capture_method": cfg["method"],
                            "spec_capture_aux_layer_ids": list(aux_layers),
                        }
                    ],
                }
            )

        def do_POST(self):
            if self.path.rstrip("/") != "/generate":
                self._send({"error": "not found"}, 404)
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length))
                rows = self._generate(body)
            except Exception as exc:
                traceback.print_exc()
                self._send({"error": f"{type(exc).__name__}: {exc}"}, 500)
                return
            self._send(rows)

        def _generate(self, body):
            input_ids = body["input_ids"]
            if input_ids and not isinstance(input_ids[0], list):
                input_ids = [input_ids]
            sampling = body.get("sampling_params") or [{}] * len(input_ids)
            if not isinstance(sampling, list):
                sampling = [sampling] * len(input_ids)
            store_id = cfg["store_id"]
            rows = []
            for i, ids in enumerate(input_ids):
                spec = json.loads(
                    (sampling[i].get("custom_params") or {}).get("spec_capture", "{}")
                )
                result = self._capture(ids, spec, store_id)
                rows.append({"meta_info": {"spec_capture": result}})
            return rows

        def _capture(self, ids, spec, store_id):
            if not spec:
                return {"error": "request carried no spec_capture payload"}
            sample_id = spec["sample_id"]
            gen = int(spec.get("gen", 1))
            replace = bool(spec.get("replace"))
            # The trainer owns the store namespace: echo whatever its producer
            # asked for. A real SGLang sink resolves the same value from the
            # spec payload, so this is what the adapter expects back.
            store_id = str(spec.get("store_id") or store_id)
            want_last = "last_hidden" in (spec.get("features") or {})
            tensors = capture_one(ids, aux_layers, want_last, cfg["dtype"])
            names = {
                "aux": spec["features"].get("aux"),
                "last_hidden": spec["features"].get("last_hidden"),
            }
            features = {}
            if replace:
                for old in (names["aux"], names["last_hidden"]):
                    if old:
                        cfg["sink"].remove(f"{store_id}/{sample_id}/g{gen}/{old}")
            for slot, out_name in names.items():
                if not out_name:
                    continue
                t = tensors[slot]
                cfg["sink"].put(f"{store_id}/{sample_id}/g{gen}/{out_name}", t)
                features[out_name] = {
                    "shape": list(t.shape),
                    "dtype": dtype_name(t),
                }
            for item in spec.get("passthrough") or []:
                name = item["name"]
                t = torch.tensor(item["data"], dtype=_torch_dtype(item["dtype"]))
                t = t.reshape(item["shape"]).contiguous()
                cfg["sink"].put(f"{store_id}/{sample_id}/g{gen}/{name}", t)
                features[name] = {"shape": list(t.shape), "dtype": dtype_name(t)}
            return {
                "sample_id": sample_id,
                "store_id": store_id,
                "gen": gen,
                "aux_layer_ids": list(aux_layers),
                "features": features,
            }

    return Handler


def _torch_dtype(name):
    return {
        "int64": torch.int64,
        "int32": torch.int32,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "bool": torch.bool,
    }[name]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--aux-layers", required=True, help="comma separated layer ids")
    ap.add_argument("--method", default="eagle3")
    ap.add_argument("--store-id", required=True)
    ap.add_argument("--mooncake-metadata-server", required=True)
    ap.add_argument("--mooncake-master-server-addr", required=True)
    ap.add_argument("--segment-bytes", type=int, default=16 << 30)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    cfg = {
        "model_path": args.model_path,
        "aux_layers": [int(x) for x in args.aux_layers.split(",") if x.strip() != ""],
        "method": args.method,
        "store_id": args.store_id,
        "dtype": args.dtype,
        "verbose": args.verbose,
        "sink": MooncakeSink(
            args.mooncake_metadata_server,
            args.mooncake_master_server_addr,
            args.segment_bytes,
        ),
    }
    print(f"[capture-server] mooncake ready, store_id={args.store_id}", flush=True)
    _, model = load_target(args.model_path, args.dtype)
    STATE["model"] = model
    print(f"[capture-server] target loaded: {args.model_path}", flush=True)

    server = ThreadingHTTPServer((args.host, args.port), make_handler(cfg))
    print(f"[capture-server] listening on {args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
