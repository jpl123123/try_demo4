# Runtime Risk Register — kvpress-ascend

Every item below is something the offline (CPU) simulation cannot fully cover;
each lists why, how to verify on the real machine, and the fail-soft fallback.

## 1. NPU kernel / CANN semantics

| Risk | Why simulation can't cover | Real-machine verification | Fallback |
|---|---|---|---|
| FIA reads of view rows: block-table row semantics (`view_row[p//bs]` slot `p%bs`), zero-padded tails, partial last block | CPU harness emulates the same read rule with numpy | Long-context accuracy A/B; NPU logs for 561002/index errors | `skipped_bad_row` diagnostics + debug log; view rows are always within `[0, num_blocks)` (CPU-guarded) |
| `npu_fused_infer_attention_score` shape checks on replaced `block_tables`/`seq_lens` tensors (graph replay rebinds per-step) | No CANN runtime | First decode step after compression under FULL_DECODE_ONLY | Metadata replacement is a shallow copy with same field types/shapes; any exception degrades to the unoptimized path |
| C8 / INT8 KV caches | Not simulated | `--quantization kv_c8` runs: expect `skipped_c8` | Scoring skipped; positional presses still work |

## 2. Scheduling / state timing

| Risk | Why | Verification | Fallback |
|---|---|---|---|
| `num_computed` update timing (sample_tokens) vs one-step-late completion (G20) | Simulated | Heartbeat counters: `reqs_compressed` matches completed requests | Complement check + mid anchors (G17) |
| Preemption / row rebuild (move/swap/add_row) racing with buffer sync | Simulated (first-block signature) | Long-run stability with preemptions (production: `--no-enable-prefix-caching`) | Full row re-sync on signature mismatch |
| MTP draft forward reading group-0 views | Verified by design: draft has its own KV group; draft metadata rebuilt in sample_tokens | Acceptance rate A/B with/without patch | None needed (draft never sees views) |
| Prefix-cache hash validity | View mode never touches content; hash keyed on original rows | Production config is `--no-enable-prefix-caching` (eviction via KV offload) → no hash interaction; if prefix caching is enabled, hit-rate log unchanged | If a future change rewrites rows, hashes would break — guarded by design |

## 3. Performance

| Risk | Mitigation |
|---|---|
| Per-step per-layer buffer sync (incremental tail copies) | Append-only sync: only new tail blocks copied per step; full re-sync only on anchors/preemption; buffers persistent |
| Query window memory: layers × requests × window × heads × hd | Reduce `KVPRESS_ASCEND_WINDOW`; windows freed when requests finish; TP splits heads per rank |
| Per-layer view buffers: layers × rows × max_blocks int32 | Bounded by decode re-anchoring + `KVPRESS_ASCEND_LAYERS`; persistent, no per-step alloc |
| Anchor-time scoring (QK^T per layer) is a one-shot cost per request | Mid anchors add scoring passes; tune `KVPRESS_ASCEND_MID_PREFILL_BUDGET` up for lower overhead |
| Hot-path `.item()`/sync | Scoring/selection run at anchors (per-request events), not per token; S4 reads CPU seq_lens which are already CPU tensors |

## 4. Approximation vs upstream kvpress

| Deviation | Why | Impact |
|---|---|---|
| Token top-k → block top-k | Block is the shared physical unit | Slightly larger effective kept set (slack invariant) |
| SnapKV softmax over all visible keys | vllm cache holds the full sequence; kvpress normalizes over early keys only | Score scale differs; ranking expected similar; A/B on real data |
| SnapKV uses post-RoPE queries (captured TND) | vllm computes attention with post-RoPE states | Matches the actual attention logits |
| avg_pool1d smoothing applied per layer | Same kernel as kvpress | Negligible |
| No decode-time (ToVa-family) presses | Out of scope for v1 | Decode bounded by re-anchoring instead |

## 5. Fail-soft inventory

All hooks wrap in try/except; counters: `skipped_error`, `skipped_ubatch`,
`skipped_cp`, `skipped_no_kv`, `skipped_no_q`, `skipped_c8`, `skipped_short`,
`skipped_bad_row`, `skipped_all_layers`, `dry_run`. A failing hook never
raises into the engine; the heartbeat's `FAIL=` field only lists seams that
were not installed.

## 6. Real-machine verification checklist (DoD handover)

1. `seams=4/4 FAIL=-` in every process heartbeat; `hit` grows each step.
2. `COMPRESS` lines per request with correct `orig/n_kept/layers`.
3. `skipped_error == 0` over a long run; any `skipped_*` explained.
4. Accuracy A/B (same seed): LongBench-style, ratio 0.25/0.5.
5. Prefix-cache hit rate unchanged vs baseline.
6. MTP acceptance rate unchanged vs baseline.
7. Performance: TTFT/TPOT delta within expectations; no `.item()` in hot path
   (profile if suspicious).
8. `python -m kvpress_ascend.simulate --suite` green on the machine.
