# SqueezeAttention-ascend

Monkeypatch adapter that ports **SqueezeAttention** (2D KV-cache management:
layer-wise optimal budget + token sparsification) to **vllm-ascend v0.23.0** —
with **zero changes to vllm-ascend source**.

Implemented as a **read-side window view rewrite**: per layer, the attention
metadata exposes only `[sink tokens] + [recent tokens]` of the block cache.
Physical cache content is never modified, so **`--enable-prefix-caching` stays
valid** and the scheduler's block accounting is untouched.

---

## Install & enable

```bash
pip install ./SqueezeAttention-ascend     # inside the SqueezeAttention-ascend dir
export squeeze=1                           # or SQUEEZE_ASCEND=1 / squeeze_ascend=1
vllm serve ...                             # your normal vllm-ascend command
```

The `.pth` file imports the package at interpreter startup in every process;
with the gate off nothing is imported (no torch/vllm). Both packages can be
installed; see *Coexistence*.

### Per-step heartbeat (the "did it really enter core code?" proof)

With `SQUEEZE_ASCEND_STEP_LOG=1` (default) every inference step prints:

```
[squeeze-ascend] INFO step=123 reqs=4 seams=3/3 hit=18 FAIL=- core=squeeze ini=0.210 start=4 class3=0.08 clusters=3 prefill=1 decode=3 viewed=4 clustered=1 anchored=2
```

`seams=3/3 FAIL=-` proves all hooks are installed; `core=...` shows the 2D
budget parameters. At clustering time the class composition is logged:

```
[squeeze-ascend] INFO CLUSTER req=abc mode=squeeze layers=48 class3_layers=12 ini_size=0.210 kv_class3=0.080 budgets_min=0.080 budgets_max=0.275 dry_run=False
```

and at every budget application:

```
[squeeze-ascend] INFO COMPRESS req=abc phase=complete mode=squeeze ini_size=0.210 start=4 prompt=262144 windows_min=20972 windows_max=72090 layers=48 dry_run=False
```

## Mechanism (ported from SqueezeAttention)

```
prefill                                          decode
  ├─ S6 layer hooks: per-(req, layer) accumulate  ├─ S4 per step: window view
  │    cos-sim(layer input, layer output)          │   rows = [sink blocks] +
  ├─ S5 completion:                                 │   [recent blocks] (slides)
  │    1D KMeans(clusters) on layer means           └─ per-layer seq_lens = view_len
  │    -> class3 (highest cos) gets KV_CLASS3,
  │       others share the leftover budget
  │       (total conserved: layers*ini_size)
  │    -> per-layer window = ratio * prompt_len
  └─ S4 applies window views per layer
```

* **S6** decoder-layer forward wrap (installed lazily at the first step after
  the model is loaded) — cos-sim capture; residual-style layers (Qwen3.5,
  Llama V1) are detected from the forward signature.
* **S4** `NPUModelRunner._build_attention_metadata` — per-layer window view
  rows in persistent buffers (sink part written once; recent part slides with
  append-only tail copies and rare full-assembly shifts) + `seq_lens` triple.
* **S5** `NPUModelRunner.execute_model` — per-step context, mid-prefill anchors
  (provisional uniform windows), completion clustering, heartbeat.

KMeans is a pure-numpy 1D implementation (no sklearn dependency), deterministic
and degenerate-safe (indistinguishable layer means → uniform budget).

## Environment variables

| Var | Default | Meaning |
|---|---|---|
| `squeeze` / `SQUEEZE_ASCEND` | unset | Master gate (any truthy value) |
| `SQUEEZE_ASCEND_INI_SIZE` | 0.21 | base per-layer KV budget (fraction of prompt) |
| `SQUEEZE_ASCEND_KV_CLASS3` | = INI_SIZE | class-3 budget; equal to INI_SIZE = uniform mode, different = squeeze mode |
| `SQUEEZE_ASCEND_START_SIZE` | 4 | sink tokens always kept |
| `SQUEEZE_ASCEND_CLUSTERS` | 3 | KMeans clusters |
| `SQUEEZE_ASCEND_MID_PREFILL` | 1 | provisional windows during long prefill |
| `SQUEEZE_ASCEND_MID_PREFILL_BUDGET` | 65536 | tokens between mid anchors |
| `SQUEEZE_ASCEND_LAYERS` | all | layer range, e.g. `0-31` |
| `SQUEEZE_ASCEND_STEP_LOG` | 1 | per-step heartbeat |
| `SQUEEZE_ASCEND_DRY_RUN` | 0 | compute budgets only, never apply views |
| `SQUEEZE_ASCEND_POLICY` | auto | auto \| primary \| defer (vs kvpress-ascend) |
| `SQUEEZE_ASCEND_LOG` | info | debug \| info \| warning |

## Compatibility with the target launch command

| Config | Interaction |
|---|---|
| `--enable-prefix-caching` | **Safe**: window views only change per-step metadata; cache content and hashes untouched |
| `--speculative_config qwen3_5_mtp` | **Safe**: MTP draft has its own KV group; views apply to target full-attention layers only |
| `--compilation-config cudagraph_mode=FULL_DECODE_ONLY` | **Safe**: graph replay refreshes metadata per step; capture skipped |
| `--tensor-parallel-size 4` | Each rank captures its own layer hidden states; budgets computed per rank from the same cos-sim signal (local means) |
| Qwen3.5 hybrid GDN layers | GDN/linear-attention layers have their own KV spec and are excluded from the window views automatically |

## Coexistence with kvpress-ascend

Both adapters rewrite the same seams. With both gates on, ownership is decided
deterministically at interpreter startup by `.pth` name order
(`kvpress_ascend.pth` < `squeeze_ascend.pth`, so kvpress owns the seams by
default and this package logs `DEFERRED: owner=kvpress installed first`). The
winner sets the shared `KV_ASCEND_OWNER` marker.

To let SqueezeAttention own the seams: `export SQUEEZE_ASCEND_POLICY=primary`
(or `KVPRESS_ASCEND_POLICY=defer`), or don't `export kvpress`. Explicit
policies always override ownership.

## Offline verification (no NPU needed)

```bash
cd SqueezeAttention-ascend
python -m squeeze_ascend.simulate          # L2 scenario + window invariant
python -m squeeze_ascend.simulate --suite  # full offline suite (20 tests)
python tests/run_tests.py                  # same suite
```

The suite covers: KMeans/budget math (L0), seam contracts (L1), the
end-to-end invariant *view slots == sink ∪ recent reference* across chunked
prefill, mid anchors, completion clustering (2D budgets), decode window
sliding, multi-request batches (L2), fail-soft injection and
heartbeat/core-parameter logging.

## Limits (honest)

* **Worker-side only**: the window shrinks what attention reads; freed blocks
  are not returned to the scheduler allocator (V1 has no worker→scheduler
  return path). The win is attention compute/bandwidth and per-request
  effective KV capacity.
* **Block granularity**: the window is expressed in blocks (sink `⌈start/bs⌉`
  blocks + recent blocks); the visible token count is within one block of the
  nominal budget.
* **Importance signal adaptation**: upstream measures
  `cos_sim(residual, residual + attn_output)`; vllm-ascend fuses the residual
  add inside the layer, so the wrapped layer hook measures
  `cos_sim(layer input, layer output)` (after the full layer incl. MLP). Both
  rank layers by how much they change the hidden state; see RISK_REGISTER.md.
* **Uniform fallback**: if cos-sim capture is unavailable (e.g. unusual model
  structure), all layers get `INI_SIZE` (logged as `skipped_no_stats`).

## Real-machine checklist (first run)

1. Heartbeat `seams=3/3 FAIL=-` on every rank; `CLUSTER` + `COMPRESS` lines
   per completed request; `skipped_error == 0`.
2. Accuracy A/B on long-context benchmarks for uniform mode
   (`KV_CLASS3 = INI_SIZE`) and squeeze mode.
3. `viewed_layers` grows; window slides during decode (heartbeat `viewed=...`).
4. With kvpress-ascend also installed, confirm only one adapter is active
   (the other prints `DEFERRED` once).
