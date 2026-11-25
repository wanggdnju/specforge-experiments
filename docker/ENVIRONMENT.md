# Environment

> 中文说明见 `ENVIRONMENT_zh.md`

What the experiments actually ran on, and what to change if you rebuild it.
`Dockerfile` in this directory reproduces the setup from the same base image.

**Training is SpecForge on plain PyTorch.** Neither vLLM nor SGLang is involved
in the training step. The base image is a vLLM image only because it already
ships a working torch + CUDA + flashinfer stack:

| stage | backend |
| --- | --- |
| training | SpecForge (SGLang team), pure PyTorch |
| feature capture during training | `transformers` running the target model (stand-in for the official SGLang capture plugin) |
| data regeneration | vLLM — chosen for throughput, unrelated to training |
| serving / benchmark | SGLang |

---

## Base image

```
vllm/vllm-openai:v5df63a7203cdb9328d824a27b682e77c7b9928be-cu129
```

| | |
| --- | --- |
| upstream | official vLLM image, commit `v5df63a72` |
| vLLM | 0.32.0+cu129 |
| CUDA | 12.9 (runtime in image) |
| OS | Ubuntu 24.04.2 LTS |
| Python | 3.12.3 |

The host driver was 550.54.15 / CUDA 12.9. **Match the driver to the image's
CUDA major version** — the hard part of this setup is that several components
want CUDA 13 while the driver only offers 12.9 (see below).

---

## Python environments

Four interpreters, because the pins are mutually incompatible:

| env | who uses it | key pins |
| --- | --- | --- |
| system python | data regeneration (vLLM); also the source of torch for `venv-specforge` | vllm 0.32.0+cu129, torch 2.13.0+cu129, transformers 5.19.0, flashinfer 0.7.0.post1, triton 3.7.1, numpy 2.2.6 |
| `venv-specforge` | training — SpecForge, no vLLM/SGLang import | `--system-site-packages` (inherits torch from the base image), plus specforge, datasets 5.2.0, liger-kernel 0.8.5, tensorboard 2.21.0, mooncake-transfer-engine 0.3.13.post1 |
| `venv-sglang` | serving | sglang 0.5.20, sglang-kernel 0.4.7, flash-attn-4 4.0.0b39, flashinfer 0.6.18, torch 2.13.0+cu129 |
| `venv-dspine` | DSpine serving | sglang 0.5.22, sglang-dspine 0.2.0, sglang-kernel 0.4.9, torch **2.14.1**+cu126, flashinfer **0.7.1** |

`venv-specforge` uses system site packages on purpose: a fresh `pip install
torch` would pull a different CUDA build and break against the base image's
libraries.

---

## Things that will bite you

These are the environment-specific ones; the modelling pitfalls are in
`../PITFALLS.md`.

### CUDA 13 kernels on a CUDA 12.9 driver

`sgl_kernel` (pulled in by SGLang) dlopens `libnvrtc.so.13` at import. The
specforge venv has no CUDA-13 tree; `venv-sglang` does. Put it on the loader
path **before the process starts**:

```bash
SP=<venv>/lib/python3.12/site-packages
export LD_LIBRARY_PATH="$SP/nvidia/cu13/lib:$SP/torch/lib:$LD_LIBRARY_PATH"
```

Setting it after launch is too late — the loader reads it once at exec.
Several ops then still abort with
`CUDA driver version is insufficient`, which is what `../shims/sglang_compat`
works around.

### `torchcodec` fails at import with `OSError`

`torchcodec` 0.17 dlopens `libtorchcodec_image.so` and raises `OSError` on this
image. SGLang guards that optional import with
`except (ImportError, RuntimeError)`, so the `OSError` escapes and kills the
process. `../shims/no_torchcodec` stubs the module **with a real
`ModuleSpec`** (a bare `types.ModuleType` breaks `importlib.util.find_spec`,
which `transformers` calls).

### Domino's GRU head cannot initialise cuDNN

`torch.nn.GRU` in Domino's corrector raises
`cuDNN error: CUDNN_STATUS_NOT_INITIALIZED`. `../shims/cudnn_off` sets
`torch.backends.cudnn.enabled = False` before torch loads, which falls back to
torch's native RNN kernel — identical results, slower setup.

### Python imports only one `sitecustomize`

Of the three shims above, only the first one on `PYTHONPATH` would load. Each
one therefore chains to the next explicitly. If you add a fourth, chain it too.

### No `git` in some hosts; GitHub is partly blocked

`codeload.github.com/<repo>/tar.gz/refs/heads/main` works where
`raw.githubusercontent.com` and the API do not. `pip download` works. The HF
CLI through `HF_ENDPOINT=https://hf-mirror.com` is ~10× faster than `curl` from
the same mirror.

### A killed training run leaks GPU memory

`mooncake_master`, `capture_server` and `EngineCore` survive a `kill` on the
trainer. They keep the card allocated, and the *next* vLLM start fails with
"free memory less than desired". Confirm `nvidia-smi` reads 0 MiB before
relaunching.

---

## Storage and models

| item | size | where the scripts expect it |
| --- | --- | --- |
| target model (Qwen3-8B) | 16 GB | `$SF_TARGET_MODEL` |
| pretrained drafts (×4) | 2–2.5 GB each | `$SF_ROOT/model/` |
| raw request dumps | ~4.4 GB | `$SF_RAW_DIR` |
| processed dataset | ~90 MB (single task) / ~600 MB (eight tasks, flattened) | `$SF_DATASET_DIR` |
| training output per run | ~14 GB (draft + FSDP state) | `artifacts/runs/` |
| exported checkpoints | ~2 GB each | `artifacts/checkpoints/` |

A full sweep of 4 algorithms × {scratch, fine-tuned, all-task} produced about
205 GB under `artifacts/`. Budget disk accordingly — that is why it is
git-ignored.

All paths are configurable through `experiments/paths.py` or a
`experiments/local.env`; see the main README.

---

## Verifying a rebuild

```bash
python experiments/paths.py            # every location resolves?
python -c "import vllm, torch; print(vllm.__version__, torch.__version__)"
./experiments/run.sh report            # end-to-end plumbing, no GPU needed
```

For a real check, run one short training job and confirm the step-10 accuracy
is close to what this document reports for the same algorithm — a wrong
environment usually shows up as a wildly different starting loss, not as an
error.
