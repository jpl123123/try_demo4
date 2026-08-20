# kvpress-ascend

Monkeypatch adapter that ports **kvpress** KV-cache compression to
**vllm-ascend v0.23.0** — with **zero changes to vllm-ascend source**.

The adapter implements kvpress's *score → top-k keep* mechanism on vllm-ascend's
block-based KV cache as a **read-side view rewrite**: physical cache content is
never modified, so **`--enable-prefix-caching` stays valid** and the scheduler
keeps its original block accounting.

---

## Install & enable

```bash
pip install ./kvpress-ascend          # inside the kvpress-ascend directory
export kvpress=1                      # or KVPRESS_ASCEND=1 / kvpress_ascend=1
vllm serve ...                        # your normal vllm-ascend command
```

The `.pth` file in site-packages imports the package at interpreter startup in
**every** process (API server / engine-core / each NPU worker rank). With the
gate off the package imports nothing (no torch/vllm). Both packages can be
installed; see *Coexistence* below.

### Per-step heartbeat (the "did it really enter core code?" proof)

With `KVPRESS_ASCEND_STEP_LOG=1` (default) every inference step prints:

```
[kvpress-ascend] INFO step=123 reqs=4 seams=4/4 hit=56 FAIL=- core=snapkv ratio=0.500 window=64 sink=4 mode=mean prefill=2 decode=2 viewed=4 compressed=3 mid=2 reanchor=1
```

`seams=4/4 FAIL=-` proves all four hooks are installed; `hit=...` counts actual
entries into the wrapped code; `core=...` shows the press and its parameters.
At every compression event a per-request line is logged with the core
parameters:

```
[kvpress-ascend] INFO COMPRESS req=abc phase=complete press=snapkv ratio=0.500 orig=262144 n_kept=131072 layers=48/48 dry_run=False
```

Set `KVPRESS_ASCEND_STEP_LOG=0` for quiet performance runs
(`KVPRESS_ASCEND_LOG=debug` for full tracing).

## Supported presses

| Press | Env | Scoring data |
|---|---|---|
| `snapkv` (default) | `KVPRESS_ASCEND_PRESS=snapkv` | last-`window` post-RoPE TND queries (captured from the Ascend attention backend) × cached keys QK^T, softmax, window-mean, avg-pool smoothing; observation window forced-kept |
| `streaming` | `KVPRESS_ASCEND_PRESS=streaming` | positional: sink + recent kept, middle pruned (StreamingLLM) |
| `random` | `KVPRESS_ASCEND_PRESS=random` | uniform random scores |
| `per_layer` | `KVPRESS_ASCEND_PRESS=per_layer` + `KVPRESS_ASCEND_PER_LAYER_RATIOS=[...]` | per-layer ratios wrapping any press |

Selection granularity is **blocks** (kvpress token-level top-k → block-level
top-k with token-score aggregation `mean|max`), because the block cache is the
smallest shared unit (all KV heads share a physical block).

## Environment variables

| Var | Default | Meaning |
|---|---|---|
| `kvpress` / `KVPRESS_ASCEND` | unset | Master gate (any truthy value) |
| `KVPRESS_ASCEND_PRESS` | snapkv | snapkv \| streaming \| random \| per_layer |
| `KVPRESS_ASCEND_RATIO` | 0.5 | Compression ratio (fraction of KV tokens removed) |
| `KVPRESS_ASCEND_WINDOW` | 64 | SnapKV observation window (tokens) |
| `KVPRESS_ASCEND_N_SINK` | 4 | StreamingLLM sink tokens |
| `KVPRESS_ASCEND_BLOCK_MODE` | mean | token→block score aggregation |
| `KVPRESS_ASCEND_PER_LAYER_RATIOS` | - | JSON list of per-layer ratios |
| `KVPRESS_ASCEND_LAYERS` | all | layer range, e.g. `0-31` |
| `KVPRESS_ASCEND_MID_PREFILL` | 1 | progressive mid-prefill anchors (long contexts) |
| `KVPRESS_ASCEND_MID_PREFILL_BUDGET` | 65536 | tokens between mid anchors |
| `KVPRESS_ASCEND_DECODE_REANCHOR` | 1 | re-compress during decode when the tail grows |
| `KVPRESS_ASCEND_DECODE_REANCHOR_WINDOW` | 8192 | max new tokens between decode re-anchors |
| `KVPRESS_ASCEND_STEP_LOG` | 1 | per-step heartbeat |
| `KVPRESS_ASCEND_DRY_RUN` | 0 | score + log only, never apply views |
| `KVPRESS_ASCEND_POLICY` | auto | auto \| primary \| defer (vs squeeze-ascend) |
| `KVPRESS_ASCEND_LOG` | info | debug \| info \| warning |

## Mechanism (how kvpress is expressed on the block cache)

```
prefill (chunked)                         decode
  ├─ S1 capture: per-(req, layer) rolling  ├─ S4 per step: view rows =
  │    post-RoPE query window               │   [kept blocks] + [true row m..]
  ├─ S5 anchor (mid / completion):          │   + per-layer seq_lens = view_len
  │    score → block scores → top-k kept    └─ (FIA / paged attention read only
  │    (last partial block always kept,          the view; write path untouched)
  │     slack invariant, latest tokens visible)
  └─ S4 from the next step: metadata views
```

* **S1** `AscendAttentionBackendImpl.forward` (and C8 variant) — TND query capture.
* **S4** `NPUModelRunner._build_attention_metadata` — per-layer metadata replaced
  (shallow copy) with view `block_tables` (persistent per-layer buffers, synced
  incrementally) + `seq_lens`/`seq_lens_cpu`/`seq_lens_list` = `view_len`.
* **S5** `NPUModelRunner.execute_model` — per-step context, completion/mid/decode
  anchors (with the one-step-late completion complement), compression pass,
  heartbeat.

Guards: graph capture (`capturing`), draft forwards (`is_draft_model`),
profiling, ubatch (list metadata), PCP, quantized (C8) caches — all skipped
with explicit counters (`skipped_*`) instead of failures. Every hook is
fail-soft: an error logs and the step continues unoptimized.

## Compatibility with the target launch command

| Config in the target command | Interaction |
|---|---|
| `--enable-prefix-caching` | **Safe**: view rewrite never touches physical cache content, so block hashes stay valid |
| `--speculative_config qwen3_5_mtp` | **Safe**: the MTP drafter has its own KV group; its metadata is rebuilt inside sample_tokens and never reads the group-0 views |
| `--compilation-config cudagraph_mode=FULL_DECODE_ONLY` | **Safe**: graph replay refreshes `block_tables`/`seq_lens` from the per-layer metadata every step; capture-time fake metadata is skipped |
| `--tensor-parallel-size 4` | Each rank optimizes its own TP-split cache independently (scores are local per rank) |
| `--quantization ascend` | Non-quantized KV cache → scoring reads bf16/fp16 keys directly (C8 INT8 KV would be skipped) |

## Coexistence with SqueezeAttention-ascend

Both adapters rewrite the same seams. With both gates on, ownership is decided
deterministically at interpreter startup by which `.pth` file runs first
(site-packages processes `.pth` files in name order, so
`kvpress_ascend.pth` < `squeeze_ascend.pth` → **kvpress owns the seams by
default** and squeeze logs `DEFERRED: owner=kvpress installed first`). The
winner sets the shared `KV_ASCEND_OWNER` marker; the loser only observes.

To let SqueezeAttention own the seams: `export KVPRESS_ASCEND_POLICY=defer`
(or `SQUEEZE_ASCEND_POLICY=primary`), or simply don't `export kvpress`.
Explicit policies always override ownership.

## Offline verification (no NPU needed)

```bash
cd kvpress-ascend
python -m kvpress_ascend.simulate          # L2 scenario + visible-set invariant
python -m kvpress_ascend.simulate --suite  # full offline suite (25 tests)
python tests/run_tests.py                  # same suite
```

The suite covers: layout formulas (L0), seam contracts on field-faithful fakes
(L1), the end-to-end invariant *view slots == reference visible set (kept +
all new tokens)* across chunked prefill, mid anchors, decode re-anchors,
multi-request batches and add_row/preemption resync (L2), fail-soft injection,
and heartbeat/core-parameter logging.

## Limits (honest)

* **Worker-side only**: blocks freed by compression are not returned to the
  scheduler's allocator (vLLM V1 has no worker→scheduler block-return path);
  the win is attention compute/bandwidth and effective KV capacity per request.
* **Block granularity**: kvpress token-level top-k becomes block-level top-k
  (mean|max aggregation); head-uniform retention is forced by shared physical
  blocks.
* **Prefix-cache hits** on a compressed request reuse the original row; the
  view machinery handles cache-hit rows via the append/resync protocol.
* **Decode-time (ToVa-style) presses** are not ported in v1; decode memory is
  bounded via decode re-anchoring instead.
* SnapKV scoring uses post-RoPE queries/keys (the real attention logits) and a
  full-length softmax normalization (kvpress normalizes over the early keys
  only) — see RISK_REGISTER.md.

## Real-machine checklist (first run)

1. Heartbeat shows `seams=4/4 FAIL=-` on every worker rank and `COMPRESS`
   lines per request; `skipped_error` stays 0.
2. Compare output quality on a long-context benchmark with/without the patch
   (same seed). Expect some degradation at ratio 0.5 (compression is lossy).
3. Check `viewed_layers`/`compressed` counters grow; `skipped_*` counters
   explain any request that was not compressed.
4. Watch NPU logs for `gather_v3`/AIV index errors — if any appear, set
   `KVPRESS_ASCEND_LOG=debug` and report the `skipped_bad_row` diagnostics.
