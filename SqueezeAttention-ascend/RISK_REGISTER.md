# Runtime Risk Register — SqueezeAttention-ascend

Items the offline (CPU) simulation cannot fully cover, with real-machine
verification methods and fail-soft fallbacks.

## 1. NPU kernel / CANN semantics

| Risk | Why simulation can't cover | Real-machine verification | Fallback |
|---|---|---|---|
| FIA reads of window view rows (sink + recent blocks, zero-padded, partial last block) | CPU harness emulates the same read rule | Long-context accuracy A/B; NPU logs for index/561002 errors | CPU guards on block ids; `skipped_bad_row` diagnostics |
| Graph replay rebinds `block_tables`/`seq_lens` per step (FULL_DECODE_ONLY) | No CANN runtime | First decode steps after completion under graph mode | Shallow-copy metadata with identical field types/shapes; exceptions degrade to unoptimized path |
| Window slide assembly inside graph replay steps | Not simulated on device | Stability over long decode | Full-resync path re-asserts marker invariants |

## 2. Scheduling / state timing

| Risk | Why | Verification | Fallback |
|---|---|---|---|
| `num_computed` update timing (sample_tokens) vs one-step-late completion (G20) | Simulated | Heartbeat `reqs_compressed` matches completed requests | Complement check implemented |
| Preemption / row rebuild racing with buffer sync | Simulated via first-block signature | Long-run stability with prefix caching + preemptions | Full row re-sync on signature mismatch |
| MTP draft forwarding (independent KV group) | By design | Acceptance-rate A/B | Views never touch draft groups |
| Prefix-cache hash validity | Views never touch content | Production config is `--no-enable-prefix-caching` (eviction via KV offload) → no hash interaction; if enabled, hit-rate log unchanged | Guarded by design |

## 3. Importance signal approximation

| Deviation | Why | Impact / verification |
|---|---|---|
| `cos_sim(layer input, layer output)` instead of upstream `cos_sim(residual, residual + attn_output)` | vllm-ascend fuses the post-attention residual add into the layer; the intermediate state is not materialized | Both signals rank "how much the layer changes the representation"; expected to be stable. A/B with uniform mode at same total budget isolates the clustering effect |
| Per-rank cos-sim means under TP | Each rank sees the same hidden stream per layer (TP splits heads, not tokens) | Means are comparable across ranks; budgets computed per rank — same input → same budgets |
| cos-sim captured on CPU (`.detach().cpu()`) at layer exit | Only during prefill steps (not hot decode path) | One small H2D per layer per prefill step; bounded by `SQUEEZE_ASCEND_MID_PREFILL_BUDGET` if overhead matters |

## 4. Performance

| Risk | Mitigation |
|---|---|
| Window slide shifts (every ~block_size tokens per row) | Full-assembly only on shift; append-only tail copies otherwise; shifts amortized |
| Sink part + recent part reads are small (budget-bounded) | Window size is a fraction of prompt; decode memory bounded by the sliding window (no decode re-anchor needed) |
| Per-layer view buffers (layers × rows × max_blocks int32) | Persistent, no per-step alloc; `SQUEEZE_ASCEND_LAYERS` caps layers |
| KMeans at completion | One pass over `layers` scalar values per request; negligible |

## 5. Fail-soft inventory

Counters: `skipped_error`, `skipped_ubatch`, `skipped_cp`, `skipped_short`,
`skipped_no_stats`, `dry_run`. All hooks try/except; the heartbeat `FAIL=`
field only lists uninstalled seams (layer_hook is installed lazily at the
first inference step and is not a failure).

## 6. Real-machine verification checklist (DoD handover)

1. `seams=3/3 FAIL=-` in heartbeats; `hit` grows; `CLUSTER`/`COMPRESS` per
   request; `skipped_error == 0`.
2. Accuracy A/B: baseline vs uniform mode vs squeeze mode at equal total budget.
3. `viewed_layers`/`reqs_compressed`/`reqs_clustered` counters grow.
4. Coexistence test with kvpress-ascend: exactly one adapter active.
5. `python -m squeeze_ascend.simulate --suite` green on the machine.
