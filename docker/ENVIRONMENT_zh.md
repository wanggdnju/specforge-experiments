# 环境说明

> English: `ENVIRONMENT.md`

实验实际运行的环境,以及重建时要注意什么。同目录的 `Dockerfile` 从同一个
基础镜像复现这套环境。

**训练用的是 SpecForge,跑在纯 PyTorch 上。** 训练环节既不用 vLLM 也不用 SGLang。
基础镜像之所以是 vLLM 镜像,只是因为它已经带了可用的 torch + CUDA + flashinfer 栈:

| 环节 | 后端 |
|---|---|
| 训练 | SpecForge(SGLang 团队),纯 PyTorch |
| 训练时的特征捕获 | `transformers` 直接跑目标模型(官方 SGLang capture 插件的替代) |
| 数据重建 | vLLM —— 为了吞吐量自己的选择,和训练无关 |
| 推理 / 评测 | SGLang |

---

## 基础镜像

```
vllm/vllm-openai:v5df63a7203cdb9328d824a27b682e77c7b9928be-cu129
```

| | |
|---|---|
| 上游 | 官方 vLLM 镜像,commit `v5df63a72` |
| vLLM | 0.32.0+cu129 |
| CUDA | 12.9(镜像内运行时) |
| OS | Ubuntu 24.04.2 LTS |
| Python | 3.12.3 |

宿主机驱动是 550.54.15 / CUDA 12.9。**务必让驱动与镜像的 CUDA 主版本匹配** ——
这套环境最难的地方就在于:多个组件要求 CUDA 13,而驱动只提供 12.9(见下文)。

---

## Python 环境

四套解释器,因为依赖版本互斥:

| 环境 | 谁在用 | 关键版本 |
|---|---|---|
| 系统 python | 数据重建(vLLM);`venv-specforge` 的 torch 也来自这里 | vllm 0.32.0+cu129、torch 2.13.0+cu129、transformers 5.19.0、flashinfer 0.7.0.post1、triton 3.7.1、numpy 2.2.6 |
| `venv-specforge` | 训练 —— SpecForge,不 import vLLM/SGLang | 用 `--system-site-packages`(继承基础镜像的 torch),另装 specforge、datasets 5.2.0、liger-kernel 0.8.5、tensorboard 2.21.0、mooncake-transfer-engine 0.3.13.post1 |
| `venv-sglang` | 推理 | sglang 0.5.20、sglang-kernel 0.4.7、flash-attn-4 4.0.0b39、flashinfer 0.6.18、torch 2.13.0+cu129 |
| `venv-dspine` | DSpine 推理 | sglang 0.5.22、sglang-dspine 0.2.0、sglang-kernel 0.4.9、torch **2.14.1**+cu126、flashinfer **0.7.1** |

`venv-specforge` 特意用 system site packages:如果重新 `pip install torch`,
会拉到一个不同的 CUDA 版本,和基础镜像的库对不上。

---

## 会坑到你的地方

这里只列环境相关的;建模类的问题在 `../PITFALLS_zh.md`。

### CUDA 13 内核跑在 CUDA 12.9 驱动上

`sgl_kernel`(SGLang 带进来的)在 import 时会 dlopen `libnvrtc.so.13`。
specforge venv 里没有 CUDA-13 的目录,`venv-sglang` 里有。必须在**进程启动前**
把它加到加载路径:

```bash
SP=<venv>/lib/python3.12/site-packages
export LD_LIBRARY_PATH="$SP/nvidia/cu13/lib:$SP/torch/lib:$LD_LIBRARY_PATH"
```

启动之后再设已经来不及 —— 加载器只在 exec 时读一次。
而且即使加对了,仍有若干算子在 launch 时报
`CUDA driver version is insufficient`,这正是 `../shims/sglang_compat`
要处理的问题。

### `torchcodec` 在 import 时抛 `OSError`

`torchcodec` 0.17 会 dlopen `libtorchcodec_image.so`,这个镜像上直接抛
`OSError`。SGLang 对可选依赖的保护是
`except (ImportError, RuntimeError)`,接不住 `OSError`,整个进程就挂了。
`../shims/no_torchcodec` 用一个**带真实 `ModuleSpec`** 的桩模块替换它
(裸 `types.ModuleType` 会让 `importlib.util.find_spec` 报错,
而 `transformers` 会调它)。

### Domino 的 GRU 头无法初始化 cuDNN

Domino 校正器里的 `torch.nn.GRU` 报
`cuDNN error: CUDNN_STATUS_NOT_INITIALIZED`。
`../shims/cudnn_off` 在 torch 加载前设
`torch.backends.cudnn.enabled = False`,回落到 torch 的原生 RNN 内核 ——
结果一致,只是初始化慢一点。

### Python 只会 import 一个 `sitecustomize`

上面三个 shim,只有 `PYTHONPATH` 上第一个能加载。因此每个都在代码里显式链到
下一个。如果你再加第四个,也要记得链。

### 部分宿主机没有 `git`,GitHub 也不完全通

`codeload.github.com/<repo>/tar.gz/refs/heads/main` 能用,
而 `raw.githubusercontent.com` 和 API 不行。`pip download` 正常。
HF CLI 走 `HF_ENDPOINT=https://hf-mirror.com` 比同一镜像的 `curl` 快约 10 倍。

### 杀掉训练进程会残留显存

`mooncake_master`、`capture_server`、`EngineCore` 会活下来,继续占着显卡,
导致**下一次** vLLM 启动报 "free memory less than desired"。
重新启动前确认 `nvidia-smi` 归零。

---

## 存储与模型

| 项 | 大小 | 脚本期望的位置 |
|---|---|---|
| 目标模型(Qwen3-8B) | 16 GB | `$SF_TARGET_MODEL` |
| 预训练草稿(×4) | 每个 2–2.5 GB | `$SF_ROOT/model/` |
| 原始请求日志 | 约 4.4 GB | `$SF_RAW_DIR` |
| 处理后数据集 | 单任务约 90 MB / 全任务 flatten 后约 600 MB | `$SF_DATASET_DIR` |
| 单个训练产出 | 约 14 GB(草稿 + FSDP 状态) | `artifacts/runs/` |
| 导出的 checkpoint | 每个约 2 GB | `artifacts/checkpoints/` |

完整跑完 4 个算法 × {从零, 微调, 全任务} 在 `artifacts/` 下产生约 205 GB。
磁盘要提前准备 —— 这也是它被 git 忽略的原因。

所有路径都可以通过 `experiments/paths.py` 或 `experiments/local.env` 配置,
详见主 README。

---

## 验证重建是否成功

```bash
python experiments/paths.py            # 所有位置都能解析吗
python -c "import vllm, torch; print(vllm.__version__, torch.__version__)"
./experiments/run.sh report            # 端到端串联起来,不需要 GPU
```

真正有效的检查:跑一个短训练,对比 step-10 的准确率是否和本文档记录的同一算法
接近。环境错了通常表现为初始 loss 差很远,而不是报错。
