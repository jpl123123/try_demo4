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
  --quantization ascend --enable-prefix-caching --mm-processor-cache-gb 0 \
  --additional-config '{"enable_cpu_binding":true}' \
  --hf-overrides '{"text_config": {"rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_type": "yarn", "rope_theta": 10000000, "partial_rotary_factor": 0.25, "factor": 4.0, "original_max_position_embeddings": 262144}}}'
```

对 KV 压缩插件的直接结论：

| 配置 | 结论 |
|---|---|
| `--enable-prefix-caching` | 视图改写安全（内容不动）；物理 compact 会破坏 hash，需 force |
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

## 4. 与 triattention 参考实现的对应关系（物理路径）

triattention（tri_3_5-fix-partial-rope-qwen35-v0.23.0）走**物理 compact + 回收**路线，其已验证的 vllm V1 补丁面（可复用为新项目的 seam 清单）：

| triattention 补丁 | 对应机制 |
|---|---|
| `Scheduler.__init__/schedule/update_from_output` 包装 | 调度侧压缩信号、effective len、事件回传 |
| `KVCacheManager.allocate_slots`（`delay_cache_blocks=True` + 临时改写 `request.num_computed_tokens`） | 压缩后避免继续提交 prefix-cache hash |
| `EngineCore.step_with_batch_queue` | async 边界屏障（压缩批不越过队列） |
| `NPUWorker.init_device/execute_model` + runner proxy 惰性安装 | worker 侧入口 |
| `set_ascend_forward_context` / model_runner 版 | 图模式守卫 |
| `ModelRunnerOutput.kv_connector_output.kv_cache_events`（declared dataclass 字段） | 跨进程压缩事件回传（普通 setattr 会被 cloudpickle 丢） |
| 输入元数据修正（seq_lens/slot mapping/block table 视图） | 与 view 模式同源的“读视图修正”，但配合物理搬移 |
| `TRIATTN_RUNTIME_*` env 全家桶 + logging master switch | 观测性模板 |

物理路线与视图路线共用 90% 的“元数据读视图修正”知识，差异只在写路径与调度器同步；新任务先按 §2.3 决策轴选路线，再取对应 seam 表。

> **完整实现参考**：本表只是速查；triattention 核心适配逻辑的逐模块详解
> （信号/触发守卫/effective len 跟踪/事件回传三优先级/allocate_slots 补丁/
> async 边界屏障/内存检查放宽/proxy 惰性安装/图模式守卫/V1+V2 输入修正/
> KV 原地 compact 三种放置语义/物理回收与分配同步/Ascend 打分后端/fast-recency
> 精度保护/观测性模板）见 `triattention-ascend-core-adaptation.md`。
