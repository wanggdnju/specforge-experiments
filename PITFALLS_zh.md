# 踩过的坑

搭建这套流程时遇到的每一个问题:现象、原因、如何避免重犯。
大致按排查耗时的长短排序。

---

## 1. 没套 chat template,τ 直接腰斩

**现象.** 一个明显能用的草稿模型,τ 只有 1.0–2.0。参考用的预训练模型测出 2.03,
而模型卡上暗示应该接近 5。

**原因.** `3_bench/measure_tau.py` 发出去的是 `system + "\n\n" + user` 裸文本。
SGLang 的 `/generate` 原样分词,**不会**套 chat template。而草稿是在
`<|im_start|>system ... <|im_end|><|im_start|>user ... <|im_end|><|im_start|>assistant`
加上 Qwen3 的空 thinking 块上训练的,裸文本让它完全偏离训练分布。
Qwen3 还需要 `enable_thinking=False` 才能和训练对齐。

**修复.** 一律用
`tokenizer.apply_chat_template(convs[:-1], add_generation_prompt=True, enable_thinking=False)`
渲染。实测效果:参考模型 2.03 → **4.51**。

**规则.** 评测草稿的 prompt,必须和训练数据的构造方式一致。
`3_bench/measure_tau.py` 现在默认这样做;`--raw-text` 保留旧的错误路径,仅供诊断。

---

## 2. 阶段"成功"但其实什么都没测

**现象.** 流水线给 8 个模型都报了 `bench: OK`,每个不到两分钟,
而报告里 τ 全是 `None`。

**原因.** 两层失败被吞掉:

* `3_bench/bench_one.py` 写的是 `{"status": "OK", "sets": {...}}`,
  但里面每个集都是 `"status": "FAIL"` —— 外层状态没有反映内层。
* 队列脚本把成功检查管道给了 `tee`,而 `tee` 会掩盖退出码,
  于是失败的检查也被算作 `DONE`。

**规则.** 阶段状态必须从实际产出推导,绝不能想当然。
把检查管道给 `tee` 时,`$?` 是 `tee` 的退出码,不是检查的。

---

## 3. 编排器空转 40 轮

**现象.** 无人值守循环(`run_all.sh`,现已移除)跑满了 `MAX_PASSES`,
而同样 3 个 run 一直在失败。

**原因.** 训练阶段设了 `progress=1`,依据是"尝试过训练",而不是"产出了 checkpoint"。
于是一个永远失败的配置看起来永远有进展。

**修复.** 只有产物真实存在才计为进展(尝试之后用 `is_trained <name>` 判断)。

---

## 4. `block_size: null` 静默用错尺寸训练

**现象.** dspark-warmstart 训练、导出、评测全部"成功",但 τ 是 1.34 而不是 ~5。

**原因.** `block_size` 只写在 `dflash_config` 里,没有放到顶层。
模型构建器先读 `config.block_size`,对 *dflash* 会回退到 `dflash_config` 的值,
但 dspark 的路径**默认成 7 且不报错**。训练用 gamma 7,推理用 16。

**怎么发现的.** 看初始训练准确率:错的那次从 0.036 起步,正确的是 0.610。
**对比 step-10 的 acc 是最快的体检手段。**

**规则.** 动过配置之后,验证**生效值**而不是文件内容:
查草稿配置顶层的 `block_size`、`dflash_config.block_size`,
以及(dspark)`num_draft_tokens == block_size + 1`。

---

## 5. 硬编码参数覆盖了 manifest

**现象.** `ValueError: DSpark speculative_num_draft_tokens must equal
gamma + 1 (= 8 for gamma=7), but got 16`。

**原因.** `3_bench/bench_one.py` 里硬编码了 `--speculative-dspark-block-size 7`,
而 suite 清单写的是 16。评测脚本里的硬编码会**静默压过**清单。

**修复.** dspark 现在**不传** block-size 参数,gamma 从草稿配置读 ——
这是唯一能同时适配 b7 和 b16 草稿的做法。

**规则.** 描述模型的参数属于模型的配置,不属于评测脚本。
如果脚本必须知道某个值,就从产物里读。

---

## 6. 参数列表 bug:`unrecognized arguments: 16`

**现象.** 所有 SGLang 服务瞬间挂掉:
`sglang serve: error: unrecognized arguments: 16`。

**原因.** 两个独立的 bug 叠加:

1. bash 里 `${var:+--flag "$var"}` 不加引号会被分词,
   把 `--draft-tokens 16` 拆成三个 token,`16` 被单独留下。
2. Python 里 `[a for a in extra if not a.startswith("--speculative-num-draft-tokens")]`
   只删了 flag,保留了**它的值**,列表里剩下一个孤立的 `16`。

**修复.** 可选 flag 用 bash 数组拼(`EXTRA+=(--flag "$val")`);
过滤 `--flag value` 这种成对的参数时,值也要一起跳过。

**规则.** 过滤 `--flag value` 必须成对处理。只按 flag 字符串过滤会留下孤儿值。

---

## 7. Python 只会加载一个 `sitecustomize`

**现象.** 往 `PYTHONPATH` 加了第二个 shim 目录之后,CUDA-13 的兼容处理失效,
服务挂在 `TreeSpeculativeSamplingTargetOnly ... CUDA driver version is insufficient`。

**原因.** `sitecustomize` 只会被导入一次,来自 `PYTHONPATH` 上**第一个**
包含它的目录。新 shim 把 `sglang_compat` 完全遮蔽了。

**修复.** 需要另一个 shim 时,在代码里显式链式调用:

```python
_shim = "<绝对路径>/sglang_compat/sitecustomize.py"
exec(compile(open(_shim).read(), _shim, "exec"), {"__name__": ...})
```

**规则.** `PYTHONPATH` 上只有第一个 `sitecustomize` 会执行。
要链式调用,不能假设两个都会加载。

---

## 8. `torchcodec` 抛 `OSError`,而 SGLang 不捕获

**现象.** `OSError: Could not load this library:
.../torchcodec/libtorchcodec_image.so`,服务在加载模型时挂掉。

**原因.** SGLang 对可选依赖的 import 用的是
`except (ImportError, RuntimeError)`,但这个构建抛的是 `OSError`。

**两个错误修法,然后才是对的.**

1. *屏蔽模块*(在 finder 里 `raise ModuleNotFoundError`)会搞坏 `transformers` ——
   它的 audio 路径会主动 import torchcodec,结果连 `AutoProcessor` 都没了。
2. *裸 `types.ModuleType` 桩*会搞坏 `importlib.util.find_spec`,
   报 `ValueError: torchcodec.__spec__ is None`。

**正确修法.** 注册桩模块时**给一个真实的 spec**:

```python
pkg.__spec__ = importlib.machinery.ModuleSpec("torchcodec", loader=None)
decoders.__spec__ = importlib.machinery.ModuleSpec("torchcodec.decoders", loader=None)
sys.modules["torchcodec"] = pkg
sys.modules["torchcodec.decoders"] = decoders
```

import 能过,`find_spec` 能过,也不会真去解码音频。

**规则.** 给可选依赖打桩时,务必带上 `ModuleSpec`。
手搓的模块没有 `__spec__`,会搞坏所有可用性探测。

---

## 9. shim 必须在需要它的库**之前**加载

**现象.** 重建时报 torchcodec 的 `OSError`,尽管 shim 函数明明写着。

**原因.** `from vllm import LLM` 写在 `_block_torchcodec()` 上面。

**规则.** 任何 import 期生效的兼容处理,都必须放在它所保护的库第一次 import 之前。
`1_data/build_all8.py` 里的顺序是 shim → cu13 → vllm。

---

## 10. vLLM 遇到一条超长 prompt 会中止整批

**现象.** 重建停在 1000 / 30212 行,报
`VLLMValidationError: ... prompt contains at least 8193 input tokens`。

**原因.** 有一条 prompt 超过了 `max_model_len`。vLLM 直接抛异常,而不是跳过。

**修复.** 在 `generate()` 之前按 token 数过滤:
`len(tok.encode(text)) <= max_model_len - max_new_tokens`。

**需要知道的后果.** 过滤是**丢弃**这些行。data4 的 prompt 平均 3.5 万字符,
3106 条里丢了 3015 条,最后只剩 77 条训练样本。
如果某个源需要长 prompt,应该调大 `--max-model-len` 或改成截断,而不是丢弃。

---

## 11. 热启动静默没生效

**现象.** 一个"热启动"的 run,step-10 的准确率和它的 scratch 版本几乎一样
(0.036 vs 0.040)—— 预训练权重等于没起作用。

**原因.** `draft_checkpoint_path` 指向的目录,config 里的架构名训练器能接受,
但**张量名**是推理布局(`selected_prefix_bridge.*`)而不是训练布局
(`injection.*`)。加载时找不到可映射的键,模型停在了初始化状态。

**怎么发现.** 对比热启动和 scratch 的 step-10 准确率,二者必须显著不同。
第二个检查:对比最终权重,

```python
# 热启动应该离预训练权重近得多,而不是离 scratch 近
mean_abs_diff(warm, pretrained)      # ~0.004
mean_abs_diff(scratch, pretrained)   # ~0.18
```

**规则.** "训练没报错"不等于"权重加载了"。
一定要验证初始准确率和 scratch 不同。

---

## 12. 导出用了错的配置,而且不报错

**现象.** 一个按 block 16 训练的 run,导出的 checkpoint 里 `block_size: 8`。

**原因.** `stage_export` 按算法名推导草稿配置
(`configs/qwen3-8b-{algo}.json`),而不是用该 run 实际训练时用的配置
(`configs/qwen3-8b-{algo}-b16.json`)。更糟的是,`artifacts/` 下生成的 run YAML
和 `configs/runs/` 里入库的那份**已经漂移**,导出读到了旧的那份。

**修复.** 导出时从该 run 自己的 YAML 读 `model.draft_model_config`;
热启动的 run 没有这个键,就用 init checkpoint 自带的 `config.json`。

**规则.** 绝不在导出时重新推导训练参数 —— 要么写进 run 配置,要么从 checkpoint 读。
另外,生成的文件一旦复制到入库位置,就**一定会**漂移:复制,不要两边手改。

---

## 13. 架构名是框架相关的

checkpoint 带着一个框架的命名过来,必须改成另一个框架认的名字。下面每一个都踩过:

| checkpoint | 原本写着 | 必须写成 | 原因 |
|---|---|---|---|
| `Qwen3-8B-Domino-b16` | `DFlashDraftModel` | `DominoDraftModel` | SpecForge 会校验策略对应的架构 |
| `Qwen3-8B-DSpine-b16` | `DFlashDraftModel` | `DSpineDraftModel`(训练)/ `DFlashDraftModel`(推理) | 训练和推理要的名字不同 |
| `dspark_qwen3_8b_block7` | `Qwen3DSparkModel` | `DSparkDraftModel` | `Qwen3DSparkModel` 没注册 |
| `Qwen3-8B-speculator.peagle` | `PEagleDraftModel` | `LlamaForCausalLMEagle3` | SGLang 硬性要求这个精确字符串(`weight_utils.py:467`) |

**规则.** 同一个 checkpoint 可能需要**两份**配置 —— 一份训练、一份推理。
放在两个目录里、权重用软链共享,而不是反复改同一个文件。

---

## 14. 推理侧有训练侧不做的硬校验

这些配置键训练时被忽略,但推理要求在**顶层**(不在 `dflash_config` 里面):

| 键 | 谁需要 |
|---|---|
| `markov_rank` | dspark:`ValueError: DSPARK requires markov_rank > 0` |
| `block_size` | dspark:推导 gamma |
| `target_layer_ids` | dspine / dflash 的 aux 层解析 |
| `max_position_embeddings` | 推导 context 长度 |
| `target_hidden_size` | EAGLE3 的 FC 输入宽度(`NoneType * int`) |
| `num_hidden_layers` | 从权重数出来;写错会重新初始化层 |

另外 `peagle` 需要嵌套的 `eagle_config.eagle_aux_hidden_state_layer_ids` 字典 ——
SGLang 读的是 `config.eagle_config["..."]`,不是扁平的那个键。

**规则.** 导出之后、花时间跑评测之前,先把导出的 `config.json`
和同架构的已知可用模型 diff 一遍。

---

## 15. `spec_accepted_tokens` / `spec_drafted_tokens` 恒为 0

这个 SGLang 构建里,这两个响应字段从来不会被填。只有
`spec_accept_length` 和 `spec_accept_rate` 有意义。

**规则.** 相信 τ 和接受率,忽略那两个 token 计数。

---

## 16. τ 不能跨 block size 比较

| 策略 | 每步草稿 token 数 |
|---|---|
| dflash / domino / dspine | 16 |
| dflash2 / dspark | 8(上游权重) |
| eagle3 / peagle | 4(1×1 树)或 16(4×2 树) |

16 预算下 τ=8,其实**不如** 8 预算下 τ=5。两种归一化有帮助:

* **接受率** —— 被接受的草稿 token 占比,报告里已有
* **(τ − 1) / (block − 1)** —— 预算的期望接受比例

**规则.** τ 只在相同 block size 内比较。跨算法请用接受率。

---

## 17. 训练侧准确率会把线上 τ 高估约 3.5 倍

某个 run 的训练日志报 `unary_greedy_accepted_length` 11.4,
而线上实测 τ 只有 3.22。训练指标是 teacher-forcing 下的逐位置估计;
线上是自回归生成、带 exposure bias 的链式结果。

**规则.** 只用线上 τ 评判草稿。训练准确率仅用于发现"没收敛"和"热启动失败"。

---

## 18. 微调可能**摧毁**一个很强的草稿

实测(权重加载已验证):

| 算法 | 预训练(域外) | data1 微调后(域外) | 变化 |
|---|---|---|---|
| dflash | 3.53 | 4.61 | +31% |
| dspark | 4.84 | 5.04 | +4% |
| domino | 7.39 | 7.05 | −5% |
| dflash2 | 4.07 | 4.04 | −1% |
| **dspine** | **8.18** | **5.68** | **−31%** |
| **eagle3** | 2.29 | 1.18 | **−48%** |

DSpine **域内和域外双双下降**(5.33 → 2.82)。它的权重平均只变了 0.004,
所以这是真实的灾难性遗忘,不是加载 bug。

**规则.** 预训练先验更强 ≠ 微调更好。要把预训练、微调、从零三个版本都测一遍,
不要假设微调一定有用。

---

## 19. 域内数字会掩盖"这个草稿擅长什么任务"

8 个通用基准(gsm8k、mbpp 等)和业务数据是**不同分布**。
八源训练在通用集上显得更差(−11%),但在 8 个业务源里赢了 7 个
(均值 3.14 → 4.62)。

**规则.** 在你实际服务的分布上评测。通用集只作次要参考。

---

## 20. 环境细节,值得记下来省得重查

* 这台机器**没装 git**。`codeload.github.com/<repo>/tar.gz/refs/heads/main`
  能用,而 `raw.githubusercontent.com` 和 API 都被墙。
* `pip download` 可用;HF CLI 走 `HF_ENDPOINT=https://hf-mirror.com`
  比 `curl` 快约 10 倍。直接 `curl` 下大文件会被限速到 ~16 KB/s。
* `setsid nohup ... &` 在这个环境里经常丢失子进程。
  优先用工具自带的后台执行,或前台 `timeout`。
* 杀掉训练进程会残留 `mooncake_master`、`capture_server` 和 `EngineCore`;
  这些残留进程占着显存,导致**下一次** vLLM 启动报
  "free memory less than desired"。重新启动前确认 `nvidia-smi` 归零。
* 失败的训练尝试会在 run 目录留下 `control/` 和 `consumer-state/`,
  SpecForge 拒绝复用(`has artifacts from an existing attempt`)。
  重试前先删掉 run 目录。
