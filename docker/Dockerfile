# Reproducible environment for the speculative-decoding experiments.
#
# WHO DOES WHAT -- this matters, because the base image is a vLLM image but
# vLLM does NOT train anything here:
#
#   training            SpecForge (the SGLang team's framework), pure PyTorch
#   feature capture     transformers running the target model directly
#                       (a stand-in for SpecForge's official SGLang capture
#                       plugin, which cannot be built against a CUDA 12.9 driver)
#   data regeneration   vLLM  -- our own choice for speed, unrelated to training
#   serving / benchmark SGLang
#
# The base is vLLM's official CUDA 12.9 image at commit v5df63a72. It is used
# because it already has a working torch+CUDA+flashinfer stack; it is a
# convenient container, not the training backend.
#
# Four Python environments exist because the pins are mutually incompatible:
#
#   system python   vLLM 0.32.0 + torch 2.13 + mooncake   (data regeneration)
#   venv-specforge  SpecForge + torch 2.13 + mooncake     (training)
#   venv-sglang     SGLang 0.5.20 + torch 2.13            (serving)
#   venv-dspine     SGLang 0.5.22 + torch 2.14 + plugin   (DSpine serving)
#
# Build:
#   docker build -f experiments/docker/Dockerfile -t specforge-experiments .
#
# Run (adjust the workspace mount to wherever your models and data live):
#   docker run --gpus all -it \
#     -v /data/workspace:/workspace \
#     specforge-experiments

FROM vllm/vllm-openai:v5df63a7203cdb9328d824a27b682e77c7b9928be-cu129

# The base image ships CUDA 12.9 and Python 3.12; keep those. Do NOT upgrade
# CUDA: the driver on the target hosts is 12.9, and SGLang's CUDA-13 kernels
# only work through the compat shims in experiments/shims/.
ARG DEBIAN_FRONTEND=noninteractive
ARG WORKSPACE=/workspace

# ---------------------------------------------------------------- system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        ninja-build \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------- base image stack
# Already present: vllm 0.32.0+cu129, torch 2.13.0+cu129, transformers 5.19.0,
# flashinfer 0.7.0.post1, triton 3.7.1, numpy 2.2.6. vLLM is only used by the
# data-regeneration stage; training runs on torch alone via SpecForge.
RUN pip install --no-cache-dir \
        "ninja==1.13.2" \
        "accelerate==1.15.0"

# ---------------------------------------------------------------- mooncake
# Transfer engine used by SpecForge's disaggregated training. The pip package
# installs the `mooncake_master` / `mooncake_client` CLIs too.
RUN pip install --no-cache-dir "mooncake-transfer-engine==0.3.13.post1"

# ------------------------------------------------------- venv: specforge
# Training. SpecForge is pure PyTorch -- it imports neither vLLM nor SGLang.
# System-site-packages so it reuses the base image's torch/CUDA (a fresh torch
# install would pull a different CUDA build).
RUN python -m venv --system-site-packages ${WORKSPACE}/venv-specforge \
    && ${WORKSPACE}/venv-specforge/bin/pip install --no-cache-dir --upgrade pip \
    && ${WORKSPACE}/venv-specforge/bin/pip install --no-cache-dir \
        "datasets==5.2.0" \
        "liger-kernel==0.8.5" \
        "tensorboard==2.21.0"

# Install SpecForge itself (edit to a `pip install -e .` for development):
#   ADD https://github.com/sgl-project/SpecForge.git /tmp/SpecForge
#   RUN ${WORKSPACE}/venv-specforge/bin/pip install --no-cache-dir /tmp/SpecForge
# Left commented because the experiments repo is normally mounted, not baked.

# ---------------------------------------------------------- venv: sglang
# Serving for every algorithm except DSpine. SGLang 0.5.20 pins its own
# kernels; torch stays at the base image's 2.13 so the two share CUDA.
RUN python -m venv ${WORKSPACE}/venv-sglang \
    && ${WORKSPACE}/venv-sglang/bin/pip install --no-cache-dir --upgrade pip \
    && ${WORKSPACE}/venv-sglang/bin/pip install --no-cache-dir \
        "torch==2.13.0" \
        --extra-index-url https://download.pytorch.org/whl/cu129 \
    && ${WORKSPACE}/venv-sglang/bin/pip install --no-cache-dir \
        "sglang==0.5.20" \
        "ninja==1.13.2"

# ---------------------------------------------------------- venv: dspine
# DSpine ships as a separate SGLang plugin pinned to 0.5.22 / torch 2.14.1.
# It cannot share the 0.5.20 environment, hence its own venv.
RUN python -m venv ${WORKSPACE}/venv-dspine \
    && ${WORKSPACE}/venv-dspine/bin/pip install --no-cache-dir --upgrade pip \
    && ${WORKSPACE}/venv-dspine/bin/pip install --no-cache-dir \
        "torch==2.14.1" \
        --extra-index-url https://download.pytorch.org/whl/cu126 \
    && ${WORKSPACE}/venv-dspine/bin/pip install --no-cache-dir \
        "sglang==0.5.22" \
        "flashinfer-python==0.7.1" \
        "ninja"

# The plugin itself lives outside PyPI; clone and install it:
#   RUN git clone --depth 1 https://github.com/tencent/dspine /opt/dspine \
#     && ${WORKSPACE}/venv-dspine/bin/pip install --no-cache-dir \
#          --no-build-isolation -e /opt/dspine

# ------------------------------------------------------------------ sanity
# Fail the build early if the CUDA-13 / torch pairing is broken, rather than
# discovering it 40 minutes into a training run.
RUN python -c "import torch, vllm; \
assert torch.cuda.is_available(), 'no CUDA'; \
print('vllm', vllm.__version__, '| torch', torch.__version__)" \
    || echo "WARNING: GPU not visible at build time (normal on CI); check at run time"

WORKDIR ${WORKSPACE}
CMD ["/bin/bash"]
