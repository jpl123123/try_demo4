# TriAttention → vllm-ascend 核心适配逻辑全解（物理 compact 路线参考实现）

> 本文档从 tri_3_5-fix-partial-rope-qwen35-v0.23.0 源码逐模块核实（2025-08）。
> 它是"**尾部块物理搬移 + 资源回收**"路线的**最完整参考实现**；与视图改写路线
> （kvpress-ascend / SqueezeAttention-ascend）对照使用，见 seam-map §6.5 决策表。
> 模块地图：`triattention/vllm/runtime/`（scheduler/worker/runner/kv_compaction/
> input_patch_*/runner_output_bridge/kv_allocation_sync/worker_reclaim_sync/...）。
> 入口：`triattention/vllm/plugin.py` + `integration_monkeypatch.py`（install 全补丁）。

---

## 1. 补丁面全景（三进程拓扑）

```
API server ── spawn ──► engine-core（调度进程）
                          ├─ Scheduler.__init__ / schedule / update_from_output   ← 触发信号 + 事件回传 + 回收
                          ├─ KVCacheManager.allocate_slots                        ← delay_cache_blocks（防 hash 污染）
                          ├─ EngineCore.step_with_batch_queue                     ← async 压缩边界屏障
                          └─ vllm.v1.core.kv_cache_utils 内存检查放宽             ← 压缩让 max_model_len 内存检查降级为警告
                     ── spawn ──► worker（每 NPU rank）
                          ├─ NPUWorker.init_device / execute_model                ← runner proxy 惰性安装
                          ├─ set_ascend_forward_context（两处）                    ← 图模式守卫（skip_compiled）
                          ├─ NPUModelRunner 被 TriAttentionModelRunner 代理        ← 压缩执行 + 元数据修正 + 事件挂载
                          └─ Ascend V1/V2 输入准备补丁（input_patch_ascend_backend）← seq_lens/positions/slot mapping/max_seq_len
```

关键设计：**native 类身份不变**（Scheduler/NPUWorker 仍是原类），全部行为经类方法包装注入；
worker 侧 proxy **惰性安装**——只在首个携带压缩信号的 step 才把 model_runner 包一层，
普通路径零开销。

## 2. 调度侧（engine-core）核心逻辑

### 2.1 触发信号（CompressionSignal）

每次 `schedule()` 后为每个请求构造信号：`should_compress = length_triggered or kv_triggered`。

- **长度触发**：`estimated_cache_len >= kv_budget + divide_length`
  （`protect_prefill` 时再加 `prefill_len`，信号阈值与有效预算语义一致）。
- **KV 压力触发**（`ENABLE_KV_USAGE_TRIGGER=1`）：`kv_cache_manager.usage` 迟滞比较
  `kv_usage_trigger(0.98)` / `kv_usage_release(0.90)`，armed 状态跨步保持。
- 信号字段：`req_id / should_compress / reason / estimated_cache_len / step /
  kv_usage / protect_prefill / prefill_len / scheduled_tokens / force`。
- **force 语义**：worker 本地硬边界触发（再拖会写穿当前块表容量）不可延迟，必须本步执行。

### 2.2 信号构建的守卫链（_build_signals）

1. `fast_recency_long_context_guard`：`effective_tokens >= guard_tokens(16384)` 且
   fast-recency-only 时跳过（每请求只告警一次）——长上下文精度保护。
2. **defer prefill compression（Ascend 默认开）**：prefill 步不触发，等完整 prompt 做完再压
   （`defer_prefill_compression_on_ascend=True` 是 Ascend 最稳模式）。
3. **prefill 阶段限次**：`prefill_max_compressions_on_ascend=1`（每请求 prefill 期最多压几次）。
4. `protect_prefill`：prefill 段整体保留，只压尾部。

### 2.3 prefill/decode 判定（is_prefill_phase_for_limit）

- **最可靠信号**：请求出现在 `scheduler_output.scheduled_new_reqs`（分块 prefill 会跨多个 chunk
  持续出现）→ 一定是 prefill。
- spec decode 步（`scheduled_spec_decode_tokens` 非空）→ 不是 prefill。
- `scheduled_tokens > 1` 且非 spec → prefill；否则按 `num_computed < prefill_len`。
- **明确不用压缩后的 effective length 判定**——它故意低于 prompt 长度，用了会把 decode 卡在
  prefill 限制后面（与视图路线"回归式状态清理"同源的教训）。

### 2.4 effective length 跟踪（EffectiveCacheLenTracker）

`num_computed_tokens` 单调增长是**请求进度**；压缩后 effective cache len 会**收缩**。
tracker 桥接两者：`observe_num_computed` 在无压缩时让 effective 跟随增长；
`apply_compression(cache_len_after)` 把 effective 置为压缩后长度；回退防御路径处理 rollback。

### 2.5 压缩事件回传与物理回收（update_from_output 三优先级）

worker 的压缩结果必须回到 engine-core 才能触发真正的块回收。V1 async 路径下
`execute_model` 返回 None、真正的 ModelRunnerOutput 由 sample_tokens 产生，普通
`setattr` 会被 cloudpickle 丢弃 → **必须用 vLLM 声明过的跨进程字段**：

```
优先级 1：ModelRunnerOutput.kv_connector_output.kv_cache_events
          （declared dataclass 字段；_TriattentionEventBag 显式实现 __reduce__/getstate/setstate 保 picklable）
优先级 2：model_runner_output.triattention_compression_events（进程内兜底）
优先级 3：scheduler_output.triattention_compression_events（同进程兜底）
```

engine-core 收到 applied 事件后 `_apply_compression_events`：
1. 按 `req_id → 压缩锚点/有效长度` 更新 tracker 与 request 状态；
2. 物理回收：`_free_reclaimed_blocks`——**先 `_maybe_evict_cached_block` 清掉旧 prefix-cache
   身份，再 `block_pool.free_blocks(reversed(removed))`**（复用前清 hash，防脏块）；
3. 回写 `scheduler_stats.kv_cache_usage`（对齐 post-reclaim 使用率）。

### 2.6 allocate_slots 补丁（防压缩后 hash 污染）

> 配置前提：本项目生产配置为 `--no-enable-prefix-caching`（物理驱逐由 KV 卸载承担）；allocate_slots 的 prefix-cache 防御在关闭时是空转但无害，保留以兼容未来开启。

`KVCacheManager.allocate_slots` 包装：对已物理压缩的请求，**临时**把
`request.num_computed_tokens` 改为 effective 值并传 `delay_cache_blocks=True`，
让 vLLM 分配槽但**跳过后续 prefix-cache commit**（物理搬移后原前缀 hash 链已失效），
finally 里还原。`prepare_request_effective_num_computed` 覆盖 WAITING（被抢占）请求。

### 2.7 EngineCore.step_with_batch_queue 边界屏障

普通 decode 保持 async 加速；**预测到本步是压缩边界批**时：不让 batch queue 越过该批、
先把边界批 drain 再调度新工作（防压缩批仍在队列时新批读到陈旧状态）。
`enable_async_compression_boundary` 开关控制（默认关，开则压稳换延迟）。

### 2.8 KV 内存检查放宽

`_check_enough_kv_cache_memory` / `check_enough_kv_cache_memory` 改为：不足时打警告而非
ValueError（压缩会在生成期把实际用量压回限制内）——否则 262144 max_model_len 直接启动失败。

## 3. Worker 侧核心逻辑

### 3.1 runner proxy 惰性安装（TriAttentionWorker）

- `NPUWorker.init_device` 包装：基类初始化后，若 `_should_early_install_proxy`
  （Ascend 环境且 `early_install_proxy_on_ascend=1`）则立刻装；否则标记待装。
- `execute_model` 包装：`should_install_triattention_runner_proxy(scheduler_output)` 为真
  （存在 `triattention_signals` 且预算/阈值/阶段守卫通过，或曾见过带信号的步）才
  `_ensure_triattention_runner_proxy()`——**普通路径零开销**。
- proxy 安装 = `self.model_runner = TriAttentionModelRunner(base_runner, config)`：
  base runner 原样保留（`isinstance` 守卫幂等），proxy 负责 override 输入、压缩、事件挂载。

### 3.2 Ascend 检测与默认参数

- `is_ascend_environment_available()`：sys.modules 含 `vllm_ascend*`、或
  `ASCEND_VISIBLE_DEVICES/ASCEND_RT_VISIBLE_DEVICES/NPU_VISIBLE_DEVICES/
  VLLM_TARGET_DEVICE/DEVICE_TARGET` 命中（结果缓存）。
- `is_ascend_runtime(obj)`：类模块名以 `vllm_ascend.` 开头 / device_config.device 含 npu/ascend。
- `apply_ascend_fast_recency_defaults`（auto 默认）：Ascend 上
  `min_reclaim_blocks_on_ascend=16`、`score_max_layers_on_ascend=8`、fast-recency 时
  min_reclaim=8——都是降 TPOT 尖峰的实测默认。

### 3.3 图模式守卫（set_ascend_forward_context 补丁）

`vllm_ascend.ascend_forward_context.set_ascend_forward_context` 与
`vllm_ascend.worker.model_runner_v1.set_ascend_forward_context` 两处包装：
当「多请求 + 有效长度 override」在 Ascend 图模式下不可靠时，把第 12 个位置参数
`skip_compiled=True` 强制传进去（eager + 跳过 compiled 路径），并临时置
`model_config.enforce_eager=True`（用完还原）——图模式下 override 修正天然生效。

## 4. 输入元数据修正（effective overrides，读视图修正的物理版）

压缩后 worker 的注意力必须只看有效前缀。triattention 用 **per-request effective
base/delta** 表达，V1/V2 各一套：

```
V1（NPUModelRunner）：
  seq_lens_np[batch_idx] = effective_base[req] + num_scheduled_tokens[req]   ← CPU 改写
  positions 设备端减 pos_delta（sparse delta 按 req_idx 展开 / packed 单请求快路径）
  slot mapping 用位移后 positions 计算（新 token 落压缩后槽位）
  build_attn_metadata 的 max_seq_len = min(原值, 有效 seq_lens 最大值)

V2（input_buffers 路径）：
  update_seq_lens_cpu / build_attn_metadata(max_seq_len) / compute_slot_mappings 三处包装，
  同上语义
```

- **单请求快速路径**：`single_seq_base` / `single_pos_delta`（避免每步张量拷贝）。
- **消费断言**：`assert_effective_overrides_consumed()`——override 状态必须被消费，
  防"算了没用上"的静默失效。
- **多请求 + Ascend 图模式守卫**：见 §3.3；`force_eager_multi_req_on_ascend_effective_overrides`
  默认开。
- 边界：worker 本地块表容量 clamp（block 边界 decode 时槽位置不得越过当前可见容量）——
  防压缩锚点与新分配块不同步导致 OOB（与 B8/B12 同族）。

## 5. KV 压缩执行（kv_compaction.py，物理路线的核心原语）

### 5.1 保留集构建（build_keep_token_indices）

- `effective_budget = kv_budget`；`protect_prefill && !include_prefill_in_budget` 时再加
  `prefill_len`；cap 到 total。
- 普通：保留最后 `effective_budget` 个 token（recency）。
- `protect_prefill`：保留 `[0, prefill_len)` + 尾部 `effective_budget - prefill_len` 个；
  预算不足放不下 prefill 段时返回 None（**不可能**信号，不硬压）。

### 5.2 缓存布局兼容（_split_kv_axes）

支持三种形态：
- split cache：`(key_cache, value_cache)` 各 `[num_blocks, bs, H, D]`（vllm-ascend）；
- combined `[2, num_blocks, bs, H, D]`（dim0 是 KV）；
- combined `[num_blocks, 2, bs, H, D]`（dim1 是 KV；TritonAttention 风格）；
- 两维都等于 2 的歧义形态：`register_kv_layout_axis_hint`（按 data_ptr/offset/shape/stride
  注册 axis），否则显式报错而不是猜。

### 5.3 gather（打分输入）

- **dense view 快路径**：`_consecutive_block_span` 检测块 id 连续（range gate + 差分全 1），
  连续则 `cache[start:end].reshape(-1, H, D)` 直接当 `[T,H,D]` 视图——**零拷贝**；
  带结果缓存（按 data_ptr/numel/device 键，上限 8192）。
- 慢路径：`key_cache[src_blocks, src_off]` 逐 token gather（CPU 前置越界校验）。

### 5.4 原地 compact（shared / per-head）

三种放置语义（`preserve_dropped_tokens=True` 默认）：
1. **全排列保持**：perm = `[kept..., dropped...]`，dst = 全 0..total-1——保留完整 token
   多重集、**不写零尾**（逻辑长度仍是 total 时写零尾会让 dropped 以零 K 参与 softmax 污染
   生成质量——注释明示的坑）。
2. **prefix_only**：只写 `[0, keep_count)` 有序保留前缀——仅当同一步物理回收尾块时用。
3. **fill-hole 最小拷贝**（`preserve_dropped_tokens=False`）：只把「留在前缀外的保留 token」
   搬进前缀空位，允许前缀无序，拷贝数最少。

per-head 变体：`keep_tensor (num_kv_heads, keep_count)`，按 head 独立索引；
gather/scatter 用 `[src_blocks, src_off, head_idx]` 三维索引。

**守卫**：keep 索引 ∈ [0,total)、无重复（TopK 假设，出错即抛）、
所有设备算子前 CPU 校验 + `TRIATTN_DEBUG_VALIDATE_COMPACTION_CONTENT=1` 内容回读断言
（prefix 内容 == gathered 期望，key/value 各一条错误码）。

## 6. 物理块回收与分配同步（worker_reclaim_sync / kv_allocation_sync）

- 压缩后按 `effective_len` 推导可回收 tail blocks；`worker_reclaim_sync` 负责
  worker 本地块表 tail 清理 + 与 engine-core 的 allocation 状态同步。
- `kv_allocation_sync`：`prepare_request_effective_num_computed`（覆盖被抢占 WAITING 请求）、
  `resolve_request_effective_num_computed`、`update_request_effective_kv_offset`——
  scheduler 侧 `_sync_effective_kv_offsets_before_schedule` 在每次 schedule 前把
  request 的 num_computed 语义对齐到 effective 基线。
- 每步最大压缩数 `max_compressions_per_step_on_ascend=4`（限流，防多请求同步压造成 TPOT 尖峰）。

## 7. Ascend 打分后端（scoring_backend=auto → torch/torch_npu）

- CUDA 用 Triton kernel；**Ascend 用 PyTorch/torch_npu**（CUDA Triton 不能直跑 NPU）。
- 临时把 K、Q 统计、RoPE 频率、频率尺度提升到 **float32** 再打分（KV 本体保持模型 dtype）。
- `score_chunk_max_tokens=4096` 分块流式（控峰值内存）；`score_max_layers=8`（Ascend 默认，
  层 cap 降 TPOT）；`score_layer_stride` 隔层打分。
- 语义：`per_head_selection_semantics = hf_aligned_global_per_head`（对齐 HF 选择）；
  `layer_perhead_aggregation/per_layer_aggregation = max`；`sparse_normalize_scores`。
- stats 文件必须与模型匹配（`TRIATTN_RUNTIME_SPARSE_STATS_PATH`，否则按模型名找打包 stats）。

## 8. fast recency 与精度保护

- `FAST_RECENCY_ONLY=1`：只保留最近 `KV_BUDGET` 个 token（诊断/极简模式，零打分）。
- `FAST_RECENCY_ACCURACY_GUARD`（默认开）：sparse stats 可用时优先真 TriAttention 打分。
- `FAST_RECENCY_LONG_CONTEXT_GUARD`：长上下文（≥16384）禁止纯 recency（掉点保护）。
- **zero-copy recency**（`enable_zero_copy_recency`，Ascend 默认）：预算与块对齐时
  **不搬 KV**，直接 tail remap（视图语义 + 回收），省一次搬移。

## 9. 观测性体系（可直接照抄的模板）

| 层级 | 内容 |
|---|---|
| logging master switch | `TRIATTN_RUNTIME_LOGGING=0` 一键关全部（错误仍打）；子开关级联关闭 |
| execution path markers | `LOG_EXECUTION_PATH=1`（+`CORE_ONLY`）——请求是否进入 runner/worker/hook/selector |
| core trace / selector debug | `LOG_CORE_TRACE=1` / `LOG_SELECTOR_DEBUG=1`（贵，显式开） |
| profile | `PERF_PROFILE` / `E2E_PROFILE` / `PHASE_PROFILE`（base_exec_ms/override_prep_ms/各阶段 ms） |
| build id | `RUNTIME_BUILD_ID` 打进启动日志——避免线上容器加载旧源码（G18 同类） |
| 启动日志 | scheduler/worker proxy 安装时打全参数（budget/divide_length/各守卫/score 旋钮/build） |

## 10. 可直接复用的工程模式（物理路线项目清单）

1. **跨进程事件必须走 declared 字段**（`kv_connector_output.kv_cache_events`），普通
   setattr 会被 cloudpickle 丢；事件对象显式实现 `__reduce__/__getstate__/__setstate__`。
2. **复用物理块前先清 prefix-cache 身份**（`_maybe_evict_cached_block` 再 free）。
3. **所有设备算子前 CPU 越界校验**（gather 槽、保留索引、块 id）——AIV 断言救不回。
4. **压缩后不写零尾**（软 max 语义：零 K 参与 softmax 污染生成）。
5. **惰性安装 proxy**（首个信号步才包装 runner）——普通路径零开销。
6. **多请求 override 走 sparse 热路径**（避免每步 dense 拷贝）+ 消费断言防静默失效。
7. **降级/守卫日志每请求只报一次**；让位/DRY_RUN 打 INFO 不打 ERROR。
8. **Ascend 默认参数族**（score cap 8 层、reclaim 16 块、defer prefill、每步限 4 压）——
   新物理路线项目直接沿用再调参。
9. **信号-事件闭环**：scheduler 发信号 → worker 压缩 → 事件回传 → scheduler 回收 →
   下一轮 effective 基线——任何一环断裂（事件丢/计数错位）都会表现为"压缩了但 KV 不降"，
   排查时按 §9 观测层逐环看。
