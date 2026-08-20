# kvpress-ascend / SqueezeAttention-ascend

**vllm-ascend v0.23.0 的 KV cache 压缩 monkeypatch 适配包**（kvpress 与 SqueezeAttention 机制移植）。
两个包都通过 `.pth` 自动注入，**不改动 vllm-ascend 任何源码**；每次推理输出心跳日志，证明 patch 进入核心代码并打印核心参数。

- `kvpress-ascend/` — kvpress 打分式压缩（SnapKV / StreamingLLM / Random / PerLayer）
- `SqueezeAttention-ascend/` — SqueezeAttention 2D 管理（层重要性 KMeans 预算 + 逐层流式窗口）
- `README.md` / `RISK_REGISTER.md` / `tests/` / 自检 CLI（每包自带）

---

## 1. 安装（两台包都要装）

```bash
pip install ./kvpress-ascend
pip install ./SqueezeAttention-ascend
```

装完 site-packages 里会出现 `kvpress_ascend.pth` 与 `squeeze_ascend.pth`，解释器启动时自动加载（**未 export 时零导入，不碰 torch/vllm**）。

## 2. 拉起方式（核心）

先 export 开关，再正常拉起 vllm serve：

```bash
# 启用 kvpress 压缩
export kvpress=1

# 或启用 SqueezeAttention 压缩（二选一，详见“双包共存”）
# export squeeze=1

vllm serve /softwarePlatform/c00879303/Qwen3.5-27B-w8a8-mtp \
  --served-model-name "qwen3.5" \
  --host 0.0.0.0 \
  --port 1144 \
  --data-parallel-size 1 \
  --tensor-parallel-size 4 \
  --max-model-len 262144 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 128 \
  --gpu-memory-utilization 0.9 \
  --compilation-config '{"cudagraph_capture_sizes":[1,4,8,12,16,24,32,48,56,64,72,84,96,108,112,128,160,172,196,200,212,232,272,288,312,328,344,360,384,400,416,432,448,480,512], "cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --speculative_config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3, "enforce_eager": true}' \
  --trust-remote-code \
  --async-scheduling \
  --allowed-local-media-path / \
  --quantization ascend \
  --enable-prefix-caching \
  --mm-processor-cache-gb 0 \
  --additional-config '{"enable_cpu_binding":true}' \
  --hf-overrides '{"text_config": {"rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_type": "yarn", "rope_theta": 10000000, "partial_rotary_factor": 0.25, "factor": 4.0, "original_max_position_embeddings": 262144}}}'
```

> 开关别名：`export kvpress=1` / `kvpress_ascend=1` / `KVPRESS_ASCEND=1`；
> `export squeeze=1` / `squeeze_ascend=1` / `SQUEEZE_ASCEND=1`（大小写均可，任意非 0 值即启用）。

### 双包共存（不要同时 export 两个）

两个包改写同一批 seam。同开时由解释器启动顺序决定归属（`.pth` 按名字排序，`kvpress_ascend.pth` 在前 → **默认 kvpress 生效**，squeeze 打印 `DEFERRED: owner=kvpress installed first`，只观测不改写）。要让 squeeze 生效：

```bash
export SQUEEZE_ASCEND_POLICY=primary   # 或 KVPRESS_ASCEND_POLICY=defer，或不 export kvpress
```

### 每次推理的验证日志（心跳）

默认每步打印一行，`seams=N/N FAIL=-` 即证明 patch 进了核心代码，并带核心参数：

```
[kvpress-ascend] INFO step=123 reqs=4 seams=4/4 hit=56 FAIL=- core=snapkv ratio=0.500 window=64 sink=4 mode=mean prefill=2 decode=2 viewed=4 compressed=3 mid=2 reanchor=1
[kvpress-ascend] INFO COMPRESS req=abc phase=complete press=snapkv ratio=0.500 orig=262144 n_kept=131072 layers=48/48 dry_run=False

[squeeze-ascend] INFO step=123 reqs=4 seams=3/3 hit=18 FAIL=- core=squeeze ini=0.210 start=4 class3=0.08 clusters=3 prefill=1 decode=3 viewed=4 clustered=1 anchored=2
[squeeze-ascend] INFO CLUSTER req=abc mode=squeeze layers=48 class3_layers=12 ini_size=0.210 kv_class3=0.080 budgets_min=0.080 budgets_max=0.275 dry_run=False
```

性能测试可关：`KVPRESS_ASCEND_STEP_LOG=0` / `SQUEEZE_ASCEND_STEP_LOG=0`；调试加 `*_LOG=debug`。

## 3. 常用旋钮（速查）

| 包 | 环境变量 | 默认 | 含义 |
|---|---|---|---|
| kvpress | `KVPRESS_ASCEND_PRESS` | snapkv | snapkv / streaming / random / per_layer |
| kvpress | `KVPRESS_ASCEND_RATIO` | 0.5 | 压缩比（移除比例） |
| kvpress | `KVPRESS_ASCEND_WINDOW` | 64 | SnapKV 观测窗口 |
| kvpress | `KVPRESS_ASCEND_MID_PREFILL_BUDGET` | 65536 | 长上下文渐进压缩锚点间隔 |
| kvpress | `KVPRESS_ASCEND_DECODE_REANCHOR_WINDOW` | 8192 | decode 重锚点间隔 |
| kvpress | `KVPRESS_ASCEND_DRY_RUN` | 0 | 只打分不改写 |
| squeeze | `SQUEEZE_ASCEND_INI_SIZE` | 0.21 | 基础每层 KV 预算（prompt 比例） |
| squeeze | `SQUEEZE_ASCEND_KV_CLASS3` | =INI_SIZE | class3 预算；不等于 INI_SIZE 即开启 squeeze 聚类模式 |
| squeeze | `SQUEEZE_ASCEND_START_SIZE` | 4 | sink token 数 |
| 两包 | `*_POLICY` | auto | auto / primary / defer（共存裁决） |
| 两包 | `*_STEP_LOG` | 1 | 每步心跳 |

完整清单见各包 `README.md` 的 env 表。

## 4. 机制摘要（如何在不改 vllm-ascend 的前提下生效）

- **视图改写（view rewrite）**：物理 KV cache 内容与调度器块表**一律不动**，只改写每步重建的 attention 元数据（`_build_attention_metadata` 返回的 per-layer `block_tables` + `seq_lens` 三元组）。
  - kvpress：`[保留块] + [真行 m..]` 视图行 + `view_len`；保留块 = 块级打分 top-k（末非对齐块强制保留、sink/窗口强制、slack 不变量）。
  - squeeze：`[sink 块] + [recent 块]` 窗口视图行，随 decode 滑动。
- **捕获点**：kvpress 在 `AscendAttentionBackendImpl.forward`（TND query 滚动窗口，含 C8 变体）；squeeze 包装 decoder layer forward（cos-sim 层输入/输出）。
- **触发**：prefill 完成 + mid-prefill 渐进锚点（长上下文防“压缩永远不触发”）+ decode 重锚点；完成判定兼容 num_computed 晚更新（G20 补检）。
- **与你的配置兼容**：`--enable-prefix-caching`（视图不碰内容，hash 有效）、`qwen3_5_mtp`（draft 独立 KV group，不读视图）、`FULL_DECODE_ONLY` 图回放（每步从当前 metadata 取参）、TP4（每 rank 独立）、`--quantization ascend`（非 INT8 KV，可直接打分）。
- **物理边界（如实说明）**：worker 侧优化不回收块内存（vLLM V1 无 worker→调度器还块通道），省的是注意力计算/带宽与单请求有效 KV 容量。

## 5. 配套方法论文档（vllm-ascend 集成技能库）

`docs/skills/vllm-ascend-debugging/` 是本次两期适配沉淀的完整方法论（与本地 skill 同步）：

- `SKILL.md` — 框架先行方法论（调度框架 → 选缝 → 机制设计 → 模拟调试 → 真机排查）、
  两条技术路线决策表（视图改写 vs 物理 compact）、视图落地协议、bug 类目 G1-G33/K1-K11、
  心跳口径与多优化包共存裁决；
- `references/vllm-ascend-v023-seam-map.md` — v0.23.0 已验证 seam/API 表 + 视图行增量同步协议；
- `references/runtime-scheduling-framework.md` — 运行调度框架（进程/流水线/状态时序/数据流）+ 双路线与共存章节；
- `references/vllm-ascend-qwen35-facts.md` — Qwen3.5/qwen3_next 架构事实 + 本仓库启动命令逐项相容性；
- `references/triattention-ascend-core-adaptation.md` — **triattention → vllm-ascend 核心适配逻辑全解**
  （物理 compact 路线的完整参考实现：触发信号/事件回传/allocate_slots 补丁/输入修正 V1+V2/
  原地 KV 压缩原语/回收与分配同步/Ascend 打分/观测性模板）；
- `references/bug-catalog.md` — 实战 bug 目录（A/B/C 三组，含二期 C1-C12）。

## 6. 离线验证（无 NPU 可跑）

```bash
cd kvpress-ascend && python -m kvpress_ascend.simulate --suite      # 25 项测试
cd SqueezeAttention-ascend && python -m squeeze_ascend.simulate --suite   # 20 项测试
python -m kvpress_ascend.simulate        # L2 场景 + 端到端可见集不变量
python -m squeeze_ascend.simulate
```

覆盖：布局公式（L0）、seam 契约（L1）、步骤驱动仿真与“视图槽集合 == 参考保留集∪全部新 token”端到端不变量（L2，跨块多步多请求）、fail-soft 注入、心跳/核心参数日志断言。CANN 算子/图回放/真实性能项登记在各包 `RISK_REGISTER.md`，真机首跑清单见各包 README。
