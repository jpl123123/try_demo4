# vllm-ascend v0.23.0 + Qwen3.5-27B（qwen3_next 架构）事实速查

> 全部条目从 vllm-ascend-releases-v0.23.0 源码逐行核实（2025-08，kvpress/squeeze 适配期）。
> 做任何 Qwen3.5 / qwen3_next / MTP 相关集成（KV 压缩、注意力、投机解码、采样）前先看本表。

## 1. 架构事实（与 llama/qwen2 系的关键差异）

| 事实 | 值 / 证据 |
|---|---|
| 模型类 | `vllm.model_executor.models.qwen3_5` / `qwen3_5_mtp`（上游 vllm 提供，本仓库不 vendored） |
| 注意力类 | `Qwen3NextAttention`（上游）→ ascend 侧整类替换为 `AscendQwen3NextAttention.forward`（`vllm_ascend/patch/worker/patch_qwen3_5.py:40-93`）；**不是标准 `Attention` 模块**，`forward(positions, output, hidden_states)` 直接写 `output[:]` |
| 混合层 | `layer_type ∈ {"linear_attention"(GDN), "full_attention"}`；GDN 层走 `AscendGatedDeltaNetAttention`（独立 state cache，**不是** `AscendAttentionBackendImpl`） |
| 解码层 | `Qwen3_5DecoderLayer.forward(hidden_states, residual, positions, **kwargs)`（residual 风格，`patch_qwen3_5.py:94-148` 整类替换）；**post-attention 的 residual 加法在 `post_attention_layernorm(hidden, residual)` 内融合，中间态不可截获** |
| MTP | `Qwen3_5MultiTokenPredictor`，`qwen3_5_mtp_forward`（`patch_qwen3_5.py:151-192`）：每 MTP 层取 `spec_step_idx % num_mtp_layers`，draft 是**独立 KV group** |
| 稀疏注意力 | `Qwen3NextAttention` 内 `self.attn(q, k, v)` 三段调用——ascend 后端 impl 的 TND 入口在 `AscendAttentionBackendImpl.forward`（attention_v1.py:1479），**全模型统一的 TND query/key/value 捕获点** |
| GDN 补丁 | `_GDN_PATCH_TARGET` = `QwenGatedDeltaNetAttention`（qwen_gdn_linear_attn）：`_split_ba_for_tp` / `get_state_shape` / `get_attn_backend` / `forward` / `_forward_core` 全套替换 |
| KV cache group | 多 group：full-attention group + GDN group + MTP draft group；`_build_attention_metadata` 逐 group `cm = copy(cm_base)`（model_runner_v1.py:3274-3322） |

## 2. 用户生产启动命令（262144 长上下文，压缩插件目标配置）

```bash
vllm serve /softwarePlatform/c00879303/Qwen3.5-27B-w8a8-mtp \
  --served-model-name "qwen3.5" --host 0.0.0.0 --port 1144 \
  --data-parallel-size 1 --tensor-parallel-size 4 \
  --max-model-len 262144 --max-num-batched-tokens 4096 --max-num-seqs 128 \
  --gpu-memory-utilization 0.9 \
  --compilation-config '{"cudagraph_capture_sizes":[1,4,8,12,16,24,32,48,56,64,72,84,96,108,112,128,160,172,196,200,212,232,272,288,312,328,344,360,384,400,416,432,448,480,512], "cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --speculative_config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3, "enforce_eager": true}' \
  --trust-remote-code --async-scheduling --allowed-local-media-path / \
  --quantization ascend --no-enable-prefix-caching --mm-processor-cache-gb 0 \
  --additional-config '{"enable_cpu_binding":true}' \
  --hf-overrides '{"text_config": {"rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_type": "yarn", "rope_theta": 10000000, "partial_rotary_factor": 0.25, "factor": 4.0, "original_max_position_embeddings": 262144}}}'
```

> 注：早期版本启动命令带 `--enable-prefix-caching`；本项目实际以 `--no-enable-prefix-caching` + KV 卸载（offload）为准，以下结论按该配置核对。

对 KV 压缩插件的直接结论：

| 配置 | 结论 |
|---|---|
| `--no-enable-prefix-caching`（**本项目生产配置**） | prefix caching 关闭 → **物理 compact 无需 force**；物理驱逐由 **KV 卸载（offload）** 承担；视图改写不受影响（开/关都兼容） |
| `--enable-prefix-caching`（若开启） | 视图改写安全（内容不动）；物理 compact 会破坏 hash，需 force |
| `qwen3_5_mtp`（`AscendStep3p5MTPProposer`） | draft 独立 KV group + 元数据在 sample_tokens 里重建 → **不读 group-0 视图**，无需 cm 重写 |
| `FULL_DECODE_ONLY` | `update_graph_params` 每步从 `attn_metadata[key].seq_lens/block_tables` 取参（attention_v1.py:745-762，非 SWA 刷新 block_tables）→ 每步重建的视图天然生效；**捕获期假元数据跳过** |
| `--max-num-batched-tokens 4096` | 262144 prompt 必然分块 prefill → 完成判定用 `before+sched`，必须配 mid-prefill 渐进锚点（G17） |
| `--quantization ascend` | 非 INT8 KV → 可直接读缓存打分；`kv_c8` 才会走 `AscendC8AttentionBackendImpl` |
| TP4 | 每 rank 独立优化自己的分片；head/kv_head 数为 TP 分片后数量（以缓存张量 shape 为地面真值） |

## 3. 集成时的 seam 落点（qwen3.5 专属注意事项）

| 目标 | 落点 | 注意 |
|---|---|---|
| 捕获 TND query | `AscendAttentionBackendImpl.forward`（+C8 变体） | 全模型统一；GDN 层不经过它；draft forward 用 `_EXTRA_CTX.is_draft_model` 排除 |
| 层重要性（cos-sim） | 包装 decoder layer forward（`Qwen3_5DecoderLayer` 或 ascend 替换类） | residual 风格：`residual=None` 时首层输入即 residual；residual 参数位置用 `inspect.signature` 探测；post-attention 中间态不可截获 → 用「层输入 vs 层输出」近似（记 RTR） |
| 目标层过滤 | `kv_cache_config.kv_cache_groups[g].layer_names` | 只取 spec 带 `block_size` 的 group；排除名字含 `mtp/draft/encoder`；GDN group 的 spec 不是 FullAttentionSpec |
| 视图行 buffer | 每层独立持久 buffer，增量同步 | 见 seam-map §6.4 同步协议 |
| 层模块解析 | layer_name `model.layers.N.self_attn.attn` → 从 runner.model 按路径走 | `layers` 是 ModuleList：路径段 `N` 用 `module[int(N)]`，不能 `getattr` |

## 4. triattention 参考实现（速查指针）

triattention（`tri_3_5-fix-partial-rope-qwen35-v0.23.0`）是**已通过 vllm-ascend v0.23.0 补丁形式成功实现的参考集成**（物理 compact 路线：调度/压缩触发/驱逐回收/输入修正全链路）。
它是本项目的**参考实现（ground truth）**：做任何新集成迷茫时，先参照它的实现逻辑。
**其逐模块详解只写在 `triattention-ascend-core-adaptation.md`**（补丁面全景、调度侧触发与回收闭环、
worker proxy 惰性安装、输入元数据修正 V1+V2、原地 KV 压缩原语、跨进程事件回传、Ascend 打分、
观测性模板、可复用工程模式），本文件不再展开。
