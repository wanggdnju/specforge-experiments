# specforge-experiments 说明(中文)

> English: [README.md](README.md) · [PITFALLS.md](PITFALLS.md)

围绕 [SpecForge](https://github.com/sgl-project/SpecForge) 写的全部工具:构建数据集、
训练投机解码草稿模型、在真实流量上测接受长度。

**本仓库不打包 SpecForge。** 它驱动一份单独安装的 SpecForge 检出,用
`SF_SPECFORGE` 指过去(见下文「环境准备」)。这样工具链和框架内部解耦,可以自由跟随上游。

```
specforge-experiments/
├── paths.py               所有文件系统位置(唯一写路径的地方)
├── local.env.example      复制成 local.env 可覆盖 paths.py
├── run.sh                 入口:build-data | train | export | bench | report
│
├── 1_data/                原始数据 → train/test JSONL
│   ├── conversations.py       解析辅助函数(四种原始 schema)
│   ├── build_all8.py          主流程:flatten → regenerate → window
│   └── convert_speculators_dflash.py
│
├── 2_train/               配置 → 训练好的 checkpoint
│   ├── gen_configs.py         算法表 → run 配置
│   ├── run_pipeline.py        五阶段驱动
│   └── run_online.sh          capture server + `specforge train`
│
├── 3_bench/               checkpoint → 接受长度 τ
│   ├── bench_one.py           一个草稿,测所有评测集
│   ├── bench_dspine.py        DSpine(独立 venv)
│   ├── measure_tau.py         测量客户端
│   ├── load_prompts.py        通用集 + 8 个业务源
│   ├── run_bench_queue.sh     带超时的顺序队列
│   └── convert_dspine_serve.py
│
├── serving/
│   └── capture_server.py      SpecForge SGLang capture 插件的替代实现
│
└── shims/                 环境兼容层(通过 PYTHONPATH 加载)
    ├── sglang_compat/     CUDA-13 算子在 12.9 驱动上
    ├── no_torchcodec/     torchcodec 构建损坏
    └── cudnn_off/         Domino 的 GRU 头
```

文档:

| 文件 | 内容 |
| --- | --- |
| `README.md` / `README_zh.md` | 本文件 —— 工具说明 |
| `PITFALLS.md` / `PITFALLS_zh.md` | 踩过的 20 个坑及避免方法 |
| `RESULTS.md` / `RESULTS_zh.md` | 环境、数据准备、训练设置、实测数字 |
| `docker/` | 复现基础镜像的 Dockerfile,以及 `ENVIRONMENT.md` / `ENVIRONMENT_zh.md` |

三个带编号的目录就是三个阶段的顺序。按这个顺序读;
理解 `1_data` 不需要碰 `2_train` 里的任何东西。

训练配置在 `configs/runs/`,清单是 `configs/suite.yaml`,Qwen3-8B 的 draft config 在 `configs/`。
产出默认落在 `$SF_SPECFORGE/artifacts/`(已被 git 忽略)。

### 环境准备

本目录下**没有任何文件硬编码路径**。所有位置都从 `paths.py` 解析,顺序是:

1. 环境变量(`paths.py` 里有完整表格)
2. `local.env`(如存在;已被 git 忽略,放本机自己的配置)
3. 从仓库位置推导出的默认值

默认假设以下布局:

```
<root>/
├── SpecForge/               被驱动的框架($SF_SPECFORGE)
├── specforge-experiments/   本仓库($SF_TOOLKIT)
├── model/Qwen3-8B/          目标模型
├── data/specforge/          原始数据 + 处理后数据
├── venv-specforge/          装了 specforge 的 python
├── venv-sglang/             装了 SGLang 的 python
└── venv-dspine/             装了 sglang-dspine 插件的 python
```

布局不同时,把 `local.env.example` 复制成 `local.env`,设好 `SF_ROOT`
(以及其他需要挪的项),然后检查:

```bash
python paths.py     # 打印所有解析结果 + ok/MISSING
```

要用你自己的模型和数据,改两处:
`2_train/gen_configs.py` 里的 `WARM_DRAFTS` 表(每个算法一个 checkpoint),
以及 `suite.yaml` 的 `eval_sets`。

### 快速开始

`run.sh` 是总入口,每个子命令都只是下面各脚本的薄封装。在仓库根目录执行。

```bash
./run.sh build-data        # 原始数据 -> train/test JSONL
./run.sh train all         # 清单里的全部 run
./run.sh export all        # 训练产出 -> 可服务的 HF 格式
./run.sh bench all         # 起服务测 τ
./run.sh report            # 生成 markdown + JSON 汇总
./run.sh bench-dspine <dir>    # DSpine(独立 venv)
./run.sh pipeline          # 五个阶段依次执行
```

`run.sh` 会为每一步选择正确的解释器:训练用 `venv-specforge`(装了
specforge)、推理用 `venv-sglang`(装了 SGLang)、DSpine 用 `venv-dspine`。
这个拆分是故意的 —— 见 `PITFALLS_zh.md` 第 20 节。

---

## 0. 一句话概括

拿原始 LLM 请求日志(8 个业务源),把每条 assistant 消息拆成独立单轮样本,用
Qwen3-8B 重新生成答案(让草稿学习**目标模型当前**的分布),截断到 2048 token
窗口,再训练 DFlash 系草稿模型。训好的草稿导出为 HF 格式,与目标模型一起挂在
SGLang 后面,用 **接受长度 τ** 打分 —— 即每步投机解码被目标模型接受的 token 数。
所有环节在一块 A800 上串行执行。

---

## 1. 数据处理 —— `1_data/`

### `build_all8.py`(当前主用)

合并全部 8 个原始数据源,三个阶段均可断点续跑:

```bash
python 1_data/build_all8.py --stage flatten
python 1_data/build_all8.py --stage regenerate
python 1_data/build_all8.py --stage window
```

**`flatten`(拆单轮)** —— `to_turns()` 先把四种不同的原始格式统一成
`[{role, content}]`(有的用 `messages`+`choices`,有的是带占位符的提示词模板)。
`flatten_to_single_turn()` 再为**每条 assistant 消息**生成一个样本,之前的内容
作为上下文。这样一条多轮 agent trace 会产生多个训练样本(每个决策点一条),
而不是一条超长的样本。之后按 `(最后一个 user 轮, 答案)` 去重 —— data2 每行
都发了两份,多个源之间也有重叠。最后每个源按哈希序截取 `--per-source`
条(默认 4000),保证每次运行取到的子集完全一致。

**`regenerate`(重建)** —— vLLM 用 Qwen3-8B 逐条重新作答,
`enable_thinking=False`、`temperature=0`、`max_tokens=1024`,保留新答案。
这一步是让草稿对齐目标模型的关键:草稿学的是 Qwen3-8B **今天**会说的话,
而不是某个旧系统的历史输出。

同时记录 `finish_reason`:被 `max_new_tokens` 截断的答案,意味着真实答案
**超过** 1024 token 只是没看全,按规则应当丢弃。放不进上下文窗口的 prompt
会在生成前被跳过(vLLM 遇到一条超长输入会中止整批)。

**`window`(截断)** —— 执行两条长度规则:答案超过 1024 token 丢弃;渲染后的
对话超过 2048 token 则**从右向左截断**(丢弃 prompt 头部),因为这与草稿在线上
实际看到的窗口一致。最后按 95/5 输出 `train.jsonl` / `test.jsonl`。

产出:`data/specforge/v5-all8/`。

**data4 的注意点**:它的 prompt 平均 3.5 万字符,跳过规则会砍掉 3106 条里的
3015 条,只剩 77 条训练 / 3 条测试。若 data4 重要,需提高 `--max-model-len`
或把"跳过"改成"截断"。

### `conversations.py`

解析辅助函数,不含流程逻辑。原始数据有四种 schema:

| schema | 典型来源 |
|---|---|
| `messages` + `choices` | 多数源 |
| `model_prompt_tmpl` + `model_prompt_placeholder` + `choices` | 少数源 |
| 带 `tool_calls` 的多轮 agent trace | agent 类 |
| 只有 `messages`,没有目标 | 少数源 |

`to_turns()` 把四种统一成 `[{role, content}]`,并把 `choices[0]` 的答案作为
最后一条 assistant 轮追加进去。注意:部分源**整个文件共用一份**带 `{{context}}`
的 `model_prompt_tmpl`,每行的实际文本在 `model_prompt_placeholder` 里;
`messages[].content` 那份是**未替换的模板** —— 用它会让所有 prompt 完全相同。

`flatten_to_single_turn()` 为每条 assistant 消息生成一个样本,之前的轮次作为
上下文。`apply_window()` 执行长度规则。`sample_key()` 是去重标识:
`(最后一个 user 轮, 答案)`。

### `convert_speculators_dflash.py`

把 `speculators` 原生格式的 DFlash checkpoint 转成 SpecForge 的 legacy 布局:
删除 speculators 专有配置键、把 `aux_hidden_state_layer_ids` 提升为
`dflash_config.target_layer_ids`、去掉 `embed_tokens` / `lm_head`
(SpecForge 的草稿借用目标模型的这两个)。源目录不会被修改。

---

## 2. 训练 —— `2_train/`

### `run_pipeline.py` —— 主驱动

五个阶段,每阶段写一个 `.done` 标记,重跑时自动续上:

```bash
python 2_train/run_pipeline.py --suite configs/suite.yaml
python 2_train/run_pipeline.py --stages report          # 只重出报告
python 2_train/run_pipeline.py --runs bench-dflash-scratch
```

| 阶段 | 动作 |
|---|---|
| `configs` | 用 `gen_configs.py` 重新生成 run 配置 |
| `train` | 对每个还没有 checkpoint 的 run 调 `run_online.sh` |
| `export` | `specforge export --to hf` 到 `artifacts/checkpoints/<run>` |
| `bench` | 起服务测 τ(委托给 `bench/`) |
| `report` | 生成 markdown + JSON 汇总 |

`configs/suite.yaml` 是**什么要跑**的唯一真相源。
`run_pipeline.py` 不会自己发明某个参数 —— 清单或 run 配置里已经写了的,
它只负责读。

### `run_online.sh <config.yaml>`

依次启动 `mooncake_master`、`capture_server.py`,等 `/server_info` 就绪,
执行 `specforge train`,退出时回收两个服务。

SpecForge 官方的 capture server 是打补丁的 SGLang,本机编不出来
(需要 Rust 1.92 和 CUDA 13,而驱动是 12.9)。`capture_server.py` 用纯 PyTorch
复现了**接口契约** —— 两个 HTTP 端点、按特定布局把张量写进 Mooncake key。
它**没有**复现 SGLang 的 radix cache 隔离和 chunked-prefill 行为,所以
它产出的 checkpoint 在流程上是正确的,但**不能**和真实 SGLang capture 逐位对比。

### `gen_configs.py`

为每个(算法, 初始化方式)组合生成一份 YAML。所有算法差异集中在这两张表:

* `ALGORITHMS` —— 草稿配置文件名、batch size、是否需要 vocab mapping、aux 层号。
* `WARM_DRAFTS` —— 哪个 checkpoint 用哪个算法做热启动。

| 算法 | batch | 需要 vocab mapping | aux 层 |
|---|---|---|---|
| dflash | 2 | 否 | 1, 9, 17, 25, 33 |
| dflash2 | 2 | 否 | 2, 10, 18, 26, 34 |
| domino | 2 | 否 | 1, 9, 17, 25, 33 |
| dspark | 2 | 否 | 1, 9, 17, 25, 33 |
| dspine | 2 | 否 | 1, 9, 17, 25, 33 |
| eagle3 | 2 | **是** | 1, 17, 32 |
| peagle | **1** | **是** | 1, 17, 32 |

`peagle` 强制 `batch_size=1`;eagle3/peagle 需要 vocab mapping,因为在
disaggregated 模式下 producer 和 consumer 无法各自推导出同一份映射。
这两条在加载配置时就会校验,**写错会直接报错而不是静默训练**。

### 超参

所有 run 完全一致,保证横向可比:

| 设置 | 值 | 依据 |
|---|---|---|
| epochs | 3 | 同 3000 条跑 9 epoch 只多 +12% τ |
| learning rate | 2e-4 | 实测比 6e-4 高 11% τ(scratch 场景) |
| batch × accum | 2 × 4 | 单卡 A800、长度 2048 下刚好 |
| warmup | 0.04 | |
| max_grad_norm | 1.0 | |
| seed | 42 | 让不同 run 之间可 diff |
| max_length | 2048 | 与线上窗口一致 |

框架自带的 DFlash 配置都用 6e-4,但那些是从零训练、语料规模大得多。

---

## 3. 评测 —— `3_bench/`

### `bench_one.py`

用 SGLang 起一个草稿模型,**一个服务测完所有评测集**(加载目标模型要几分钟,
必须复用):

```bash
python 3_bench/bench_one.py \
    --draft artifacts/checkpoints/bench-dflash-warmstart \
    --algorithm dflash --out result.json
```

算法 → SGLang 参数的映射**并不统一**,这是本仓库最容易出错的地方:

| SpecForge 算法 | SGLang 算法 | 额外参数 |
|---|---|---|
| dflash | `DFLASH` | block size 来自草稿配置 |
| dflash2 | `DFLASH` | block 8 或 16,来自草稿配置 |
| domino | `DFLASH` | 靠 `projector_type: domino` 路由;需关闭 cuDNN |
| dspark | `DSPARK` | gamma = 草稿的 block_size,`num_draft_tokens = gamma + 1` |
| dspine | *(独立)* | 见 `bench_dspine.py` |
| eagle3 | `EAGLE3` | 树的大小 = `num-steps` × `eagle-topk` |
| peagle | `EAGLE3` | 改写配置后走 EAGLE3 路径 |

`prepare_peagle_config()` 把 PEagle 草稿改写成 EAGLE3 的形状(架构名、
`target_hidden_size`、嵌套的 `eagle_config`),因为两者的**张量本来就一样**。

### `measure_tau.py`

测量客户端。**必须用 chat template 渲染 prompt** —— SGLang 的 `/generate`
不会自动套模板,直接喂裸文本会让 τ 大约腰斩。只读
`spec_accept_length` 和 `spec_accept_rate`;本版本 SGLang 的 token 计数字段
恒为 0。

支持两种输入:已渲染好的 `prompt` 字段(来自 `load_prompts.py`),或
`conversations`。

### `bench_dspine.py`

DSpine 需要独立的 SGLang 插件(`sglang-dspine 0.2.0`),它绑定 sglang 0.5.22
和 torch 2.14.1,与其余算法用的 0.5.20 环境**无法共存**。所以这个脚本从
`venv-dspine` 起服务,客户端也用同一个解释器;只有 prompt 文件和结果格式共用。

### `run_bench_queue.sh`

带单模型超时的顺序执行器。坏掉的服务会一直卡在 `/health` 同时占着 GPU,
所以每个模型都在 `timeout` 下运行,模型之间强制杀掉所有 sglang 进程并轮询到
显存归零。

```bash
./3_bench/run_bench_queue.sh                 # 清单里全部
./3_bench/run_bench_queue.sh bench-dspark-all8
```

### `load_prompts.py`

把各类评测集转成 chat prompt。两大类:

* `datasets/<name>/` —— 七个通用基准(aime25、math500、gsm8k、humaneval、
  mbpp、lcb、mtbench),五种不同 schema。
* `src1..src8` —— 八个业务源,从合并测试集里拆出来,用来回答
  "这个草稿最擅长哪类任务"。并非八个都能用:`src4` 经长度过滤后只剩 3 条,
  因此按任务的报告只覆盖另外七个(在 `RESULTS_zh.md` 里编号 A–G)。


---

## 4. DSpine 格式转换(推理侧)

### `3_bench/convert_dspine_serve.py`

SpecForge 和推理插件对同一条 code bridge 用了不同命名:

| SpecForge(训练侧) | sglang-dspine(推理侧) |
|---|---|
| `transfer_codes` | `selected_prefix_bridge.codebook` |
| `injection.bias` | `selected_prefix_bridge.gate.bias` |
| `injection.message.weight` | `selected_prefix_bridge.gate.weight` |
| `injection.predecessor_gate.weight` | `selected_prefix_bridge.predecessor_gate.weight` |
| `injection.receiver_gate.weight` | `selected_prefix_bridge.prev.weight` |
| `injection.readouts.{i}` | `selected_prefix_bridge.early_read_weights[i]` |
| `injection.writes.{i}` | `selected_prefix_bridge.feature_write_weights[i]` |
| *(无对应)* | `selected_prefix_bridge.out.weight`(填零) |

这个映射是**从张量形状推断**的,没有文档依据。`out.weight` 在训练侧
没有对应张量,只能填零,所以 DSpine 的推理数值仅供参考。

---


## 5. 服务与兼容层

### `serving/capture_server.py`

SpecForge 官方的 capture server 是打补丁的 SGLang(需要 Rust 1.92 和 CUDA 13,
本机驱动是 12.9,编不出来)。这个用纯 PyTorch 复现了**接口契约**:
两个 HTTP 端点、按特定布局把张量写进 Mooncake key。

关键契约点:`store_id` 必须回显请求里的值;特征名必须符合算法契约
(`hidden_state`、`attention_mask`、`loss_mask`、`input_ids`、`target`);
`hidden_states[i+1]` 才是第 i 层输出(索引 0 是 embedding)。

它**没有**复现 SGLang 的 radix cache 隔离和 chunked-prefill,所以产出的
checkpoint 在流程上正确,但不能和真实 SGLang capture 逐位对比。

### 环境兼容层 —— `shims/`

使用方法:把目录放在 `PYTHONPATH` 最前面。Python **只会 import 一个**
`sitecustomize`,所以一个 shim 若还需要另一个,必须显式链式调用
(`cudnn_off` 和 `no_torchcodec` 都是这么做的)。

| shim | 解决的问题 | 做法 |
|---|---|---|
| `sglang_compat` | CUDA-13 算子在 12.9 驱动上直接中止 | 隐藏非贪心验证算子;设 `SGLANG_COMPAT_TRITON_TREE=1` 时为 EAGLE 树算子装上 Triton 替代实现 |
| `no_torchcodec` | `torchcodec` 在 import 时抛 `OSError`,逃出了 SGLang 的 `except (ImportError, RuntimeError)` | 注册**带真实 `ModuleSpec`** 的桩模块,让 `import` 和 `find_spec` 都能通过 |
| `cudnn_off` | Domino 的 GRU 头在本机无法初始化 cuDNN | 在 torch 之前设 `cudnn.enabled = False`,并链式调用 `sglang_compat` |

详细原因和走过的弯路见 `PITFALLS_zh.md` 第 7–9 节。

---

## 6. 路径、配置与目录

**`paths.py`** —— 唯一出现绝对路径的地方。其他 Python 文件都
`import paths as _p`,shell 脚本则调 `python paths.py --print <KEY>`。
要换模型、数据、venv 或 shim 目录,改 `paths.py` 的默认值,或者写
`local.env` 覆盖。

`configs/runs/` 下的 run 配置里,**路径全部相对 SpecForge 检出目录**
(`specforge train` 按当前目录解析,而它总是在框架检出里启动),所以这些
配置在任何具有相同相对布局的机器上都能直接用。

```
configs/
├── suite.yaml                 清单:每个 run、每个评测集
├── runs/bench-<算法>-<初始化>.yaml   run 配置(SpecForge train --config 的输入)
└── qwen3-8b-*.json            草稿架构(训练器的 draft_model_config)
    ├── qwen3-8b-dflash.json       上游同名配置
    ├── qwen3-8b-dflash2-b16.json  block 16
    ├── qwen3-8b-dspark-b16.json   block 16
    └── qwen3-8b-dspine.json       从发布 checkpoint 派生

artifacts/                     git 忽略,约 205 GB(默认位于 $SF_SPECFORGE 下)
├── datasets/v4-data1/         单源数据集
├── checkpoints/<run>/         导出的 HF 草稿(可直接服务)
├── runs/<run>/                训练原始产出(FSDP 状态,仅用于续训)
├── pipeline/{configs,logs,reports,prompts}
└── benchmarks/                早期 τ 测量结果
```

run 配置可以直接喂给框架:

```bash
specforge train --config configs/runs/bench-dflash-scratch.yaml
```

---

## 7. 对框架版本的说明

本工具链是在 SpecForge 的一个中间版本上开发和实测的。那个版本有两个缺陷,
后来上游的重构把它们都消除了:

1. `PEagleDraftModel.from_pretrained` 未处理 `output_loading_info=True`
   (transformers 此时返回 `(model, info)` 元组),PEagle 热启动会崩在
   `'tuple' object has no attribute '_rebuild_rotary_embedding'`。新版把
   P-EAGLE 改成继承 `DraftPreTrainedModel`,该基类已正确处理。
2. EAGLE3 的 `build_server_collator` 用 `concatenate_features`,要求批内序列
   等长,`batch_size > 1` 时立刻报 `Sizes of tensors must match`。新版删除了
   这个 collator,统一走 `pad_and_concatenate_features`。

**因此本仓库不含对 SpecForge 的补丁,也不假定上述任一实现细节存在。**
如果你用的是更旧的 SpecForge,这两处需要自行确认;`RESULTS.md` 的数字是在
含这两个修复的版本上测的。

---

## 8. 结果

实测结果单独放在 [RESULTS_zh.md](RESULTS_zh.md)(English:
[RESULTS.md](RESULTS.md)),包含环境、数据准备流程、训练与测试的完整设置。
本文档只描述工具本身。
