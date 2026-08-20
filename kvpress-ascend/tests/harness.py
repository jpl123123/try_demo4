"""L1/L2 simulation harness for kvpress-ascend.

Fakes are field-faithful to vllm-ascend v0.23.0 (see
vllm_ascend/worker/block_table.py, vllm_ascend/attention/attention_v1.py,
vllm_ascend/worker/model_runner_v1.py).  The step driver replicates the real
per-step order:

    execute_model (S5 wrapper) ->
      _update_states (row growth) ->
      _prepare_inputs (positions, slot mapping, seq lens, q lengths) ->
      _build_attention_metadata (S4 wrapper) ->
      per-layer backend forward (S1 wrapper: capture + KV write) ->
      [finally] compression pass + heartbeat ->
      sample_tokens (num_computed update)

Levels: L1 = seam contracts with fakes; L2 = multi-step behavior with the
end-to-end visible-set invariant.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import numpy as np
import torch

from kvpress_ascend.runtime.capture import make_backend_forward_wrapper
from kvpress_ascend.runtime.pass_engine import make_execute_model_wrapper
from kvpress_ascend.runtime.view import make_build_attn_metadata_wrapper

BLOCK_SIZE = 16
NUM_LAYERS = 4
NUM_BLOCKS = 64  # physical blocks in the fake cache
HEADS = 4
KV_HEADS = 2
HD = 8


class FakeAscendMetadata:
    """Mirror of the AscendMetadata fields our seams touch."""

    def __init__(self, attn_state, seq_lens, seq_lens_cpu, seq_lens_list,
                 block_tables, slot_mapping, actual_seq_lengths_q):
        self.attn_state = attn_state
        self.seq_lens = seq_lens
        self.seq_lens_cpu = seq_lens_cpu
        self.seq_lens_list = seq_lens_list
        self.block_tables = block_tables
        self.slot_mapping = slot_mapping
        self.actual_seq_lengths_q = actual_seq_lengths_q
        self.num_actual_tokens = 0
        self.causal = True


class FakeBlockTable:
    """Mirror of vllm_ascend BlockTable (np rows + gpu mirror)."""

    def __init__(self, bs=BLOCK_SIZE, max_reqs=8, max_blocks=32, max_tokens=1024):
        self.block_size = bs
        self.max_num_reqs = max_reqs
        self.block_table = self._buf((max_reqs, max_blocks), np.int32)
        self.num_blocks_per_row = np.zeros(max_reqs, dtype=np.int32)
        self.slot_mapping = self._buf((max_tokens,), np.int64)
        self._next_block = 0

    def _buf(self, shape, dtype):
        np_arr = np.zeros(shape, dtype=dtype)
        return SimpleNamespace(np=np_arr, gpu=torch.from_numpy(np_arr.copy()))

    def alloc_blocks(self, n):
        ids = list(range(self._next_block, self._next_block + n))
        self._next_block += n
        return ids

    def grow_row(self, row_idx: int, n_blocks: int):
        """append_row semantics: append fresh physical blocks."""
        ids = self.alloc_blocks(n_blocks)
        start = int(self.num_blocks_per_row[row_idx])
        self.block_table.np[row_idx, start : start + n_blocks] = ids
        self.num_blocks_per_row[row_idx] = start + n_blocks

    def commit_block_table(self, num_reqs: int):
        self.block_table.gpu[:num_reqs] = torch.from_numpy(self.block_table.np[:num_reqs].copy())

    def get_device_tensor(self, num_reqs=None):
        return self.block_table.gpu if num_reqs is None else self.block_table.gpu[:num_reqs]

    def compute_slot_mapping(self, num_reqs, query_start_loc, positions):
        """Fake kernel: slot = row[pos//bs]*bs + pos%bs (matches ascend)."""
        row_np = self.block_table.np
        qsl = query_start_loc.numpy() if torch.is_tensor(query_start_loc) else query_start_loc
        pos_np = positions.numpy() if torch.is_tensor(positions) else positions
        slots = []
        for i in range(num_reqs):
            lo, hi = int(qsl[i]), int(qsl[i + 1])
            for p in pos_np[lo:hi]:
                slots.append(int(row_np[i, p // self.block_size]) * self.block_size + p % self.block_size)
        arr = np.array(slots, dtype=np.int64)
        self.slot_mapping.np[: arr.size] = arr
        self.slot_mapping.gpu[: arr.size] = torch.from_numpy(arr)
        return arr


class FakeAttention:
    """vllm Attention module: writes KV via slot_mapping, then returns output."""

    def __init__(self, layer_name, key_cache, value_cache):
        self.layer_name = layer_name
        self.kv_cache = (key_cache, value_cache)

    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output=None,
                output_scale=None, output_block_scale=None):
        assert output is not None
        num_tokens = query.shape[0]
        sm = attn_metadata.slot_mapping[:num_tokens]
        if key is not None and value is not None:
            kv_heads = key.shape[1]
            hd = key.shape[2]
            flat_k = self.kv_cache[0].reshape(-1, kv_heads, hd)
            flat_v = self.kv_cache[1].reshape(-1, kv_heads, hd)
            sm_t = sm.to(torch.int64)
            flat_k[sm_t] = key
            flat_v[sm_t] = value
        # deterministic "attention output" (content does not matter for the
        # visible-set invariant; the write path above is what matters)
        output[:num_tokens] = query.reshape(num_tokens, -1)[:, : output.shape[1]]
        return output


class FakeMultiGroupBlockTable:
    """Mirror of MultiGroupBlockTable: indexable by kv-cache group id."""

    def __init__(self, table: FakeBlockTable):
        self._t = table

    def __getitem__(self, idx: int):
        return self._t

    def commit_block_table(self, num_reqs: int):
        self._t.commit_block_table(num_reqs)


class FakeInputBatch:
    def __init__(self, table: FakeBlockTable, max_reqs=8):
        self.req_ids = []
        self.req_id_to_index = {}
        self.num_computed_tokens_cpu = np.zeros(max_reqs, dtype=np.int64)
        self.num_prompt_tokens = np.zeros(max_reqs, dtype=np.int64)
        self.block_table = FakeMultiGroupBlockTable(table)
        self.num_reqs = 0


class FakeRunner:
    def __init__(self, layer_names, table: FakeBlockTable, bs=BLOCK_SIZE,
                 num_blocks=NUM_BLOCKS, heads=HEADS, kv_heads=KV_HEADS, hd=HD):
        self.layer_names = list(layer_names)
        self.bs = bs
        self.heads = heads
        self.kv_heads = kv_heads
        self.hd = hd
        self.device = torch.device("cpu")
        self.use_cp = False
        self.attn_state = "PrefillNoCache"
        self.actual_seq_lengths_q = []
        self.table = table
        self.input_batch = FakeInputBatch(table)
        key_cache = torch.randn(num_blocks, bs, kv_heads, hd)
        value_cache = torch.randn(num_blocks, bs, kv_heads, hd)
        self.kv_cache_config = SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(
                    layer_names=self.layer_names,
                    kv_cache_spec=SimpleNamespace(
                        block_size=bs, num_kv_heads=kv_heads, head_size=hd
                    ),
                )
            ]
        )
        self.compilation_config = SimpleNamespace(
            static_forward_context={
                name: FakeAttention(name, key_cache, value_cache) for name in self.layer_names
            }
        )
        self.requests = {}

    # ---- engine-side methods (wrapped by the adapter's wrappers) ------- #
    def _build_attention_metadata(self, num_tokens, num_reqs, max_query_len,
                                  for_cudagraph_capture=False, **kwargs):
        q_lens = list(self.actual_seq_lengths_q[:num_reqs])
        attn_metadata = {}
        for layer_name in self.layer_names:
            seqs_np = np.array(
                [int(self.input_batch.num_computed_tokens_cpu[i]) + q_lens[i] for i in range(num_reqs)],
                dtype=np.int64,
            )
            seqs = torch.tensor(seqs_np)
            meta = FakeAscendMetadata(
                attn_state=self.attn_state,
                seq_lens=seqs,
                seq_lens_cpu=seqs,
                seq_lens_list=seqs_np.tolist(),
                block_tables=self.table.get_device_tensor(),
                slot_mapping=self.table.slot_mapping.gpu,
                actual_seq_lengths_q=q_lens,
            )
            attn_metadata[layer_name] = meta
        return attn_metadata, None

    def execute_model(self, scheduler_output, intermediate_tensors=None):
        """Driver step: update states -> prepare -> build meta -> forward."""
        num_scheduled = scheduler_output.num_scheduled_tokens
        req_ids = self.input_batch.req_ids
        n = len(req_ids)
        sched_np = np.array([int(num_scheduled.get(r, 0)) for r in req_ids], dtype=np.int64)

        # _update_states: grow rows for scheduled tokens
        for i, rid in enumerate(req_ids):
            before = int(self.input_batch.num_computed_tokens_cpu[i])
            need = (before + int(sched_np[i]) + self.bs - 1) // self.bs
            have = int(self.table.num_blocks_per_row[i])
            if need > have:
                self.table.grow_row(i, need - have)
        self.table.commit_block_table(n)

        # _prepare_inputs: positions + slot mapping
        total = int(sched_np.sum())
        positions = []
        for i, rid in enumerate(req_ids):
            base = int(self.input_batch.num_computed_tokens_cpu[i])
            for j in range(int(sched_np[i])):
                positions.append(base + j)
        pos_t = torch.tensor(positions, dtype=torch.int64)
        qsl = np.concatenate([[0], np.cumsum(sched_np)]).astype(np.int64)
        qsl_t = torch.from_numpy(qsl)
        self.table.compute_slot_mapping(n, qsl_t, pos_t)

        self.actual_seq_lengths_q = sched_np.tolist()
        self.attn_state = (
            "DecodeOnly"
            if int(sched_np.sum()) == n and all(
                int(self.input_batch.num_computed_tokens_cpu[i]) >= int(self.input_batch.num_prompt_tokens[i])
                for i in range(n)
            )
            else "ChunkedPrefill"
        )

        attn_metadata, _ = self._build_attention_metadata(total, n, 0)
        self._last_attn_metadata = attn_metadata  # like execute_model_state

        if total == 0:
            return None  # like vLLM's EMPTY early return

        # _model_forward: per layer
        rng = np.random.default_rng(42)
        for layer_name in self.layer_names:
            q = torch.randn(total, self.heads, self.hd)
            k = torch.randn(total, self.kv_heads, self.hd)
            v = torch.randn(total, self.kv_heads, self.hd)
            out = torch.zeros(total, self.heads * self.hd)
            attn = self.compilation_config.static_forward_context[layer_name]
            attn.forward(
                SimpleNamespace(layer_name=layer_name),
                q, k, v,
                attn.kv_cache,
                attn_metadata[layer_name],
                out,
            )
        return None

    def sample_tokens(self):
        """num_computed update (executes AFTER execute_model returns)."""
        for i, rid in enumerate(self.input_batch.req_ids):
            sched = int(self.actual_seq_lengths_q[i])
            self.input_batch.num_computed_tokens_cpu[i] += sched


class SchedulerOutput:
    def __init__(self, num_scheduled: dict, finished=()):
        self.num_scheduled_tokens = num_scheduled
        self.total_num_scheduled_tokens = sum(num_scheduled.values())
        self.finished_req_ids = list(finished)


# --------------------------------------------------------------------- #
# driver helpers
# --------------------------------------------------------------------- #
def add_request(runner: FakeRunner, req_id: str, prompt_len: int):
    idx = len(runner.input_batch.req_ids)
    runner.input_batch.req_ids.append(req_id)
    runner.input_batch.req_id_to_index[req_id] = idx
    runner.input_batch.num_prompt_tokens[idx] = prompt_len
    runner.input_batch.num_computed_tokens_cpu[idx] = 0
    runner.input_batch.num_reqs = len(runner.input_batch.req_ids)


def run_step(runner: FakeRunner, sched: dict) -> None:
    """One full inference step (execute_model + sample_tokens)."""
    so = SchedulerOutput(sched)
    runner.execute_model(so)
    runner.sample_tokens()


def install_wrappers():
    """Apply the real adapter wrappers to the fakes (idempotent per class)."""
    if not getattr(FakeAttention.forward, "_kvpress_ascend_patched", False):
        FakeAttention.forward = make_backend_forward_wrapper(FakeAttention.forward)  # type: ignore[assignment]
        FakeAttention.forward._kvpress_ascend_patched = True  # type: ignore[attr-defined]
    if not getattr(FakeRunner._build_attention_metadata, "_kvpress_ascend_patched", False):
        FakeRunner._build_attention_metadata = make_build_attn_metadata_wrapper(  # type: ignore[assignment]
            FakeRunner._build_attention_metadata
        )
        FakeRunner._build_attention_metadata._kvpress_ascend_patched = True  # type: ignore[attr-defined]
    if not getattr(FakeRunner.execute_model, "_kvpress_ascend_patched", False):
        FakeRunner.execute_model = make_execute_model_wrapper(FakeRunner.execute_model)  # type: ignore[assignment]
        FakeRunner.execute_model._kvpress_ascend_patched = True  # type: ignore[attr-defined]


def make_runner(layer_names=None, cfg_env=None, bs=BLOCK_SIZE, num_blocks=NUM_BLOCKS):
    """Fresh runner with a clean adapter state (env hygiene included)."""
    import os

    from kvpress_ascend.envs import reset_config
    from kvpress_ascend import registry

    for k in list(os.environ):
        low = k.lower()
        if low in ("kvpress", "kvpress_ascend") or low.startswith("kvpress_ascend_"):
            os.environ.pop(k, None)
    if cfg_env:
        for k, v in cfg_env.items():
            os.environ[k] = v
    reset_config()
    registry.reset()
    layer_names = layer_names or ["model.layers.%d.self_attn.attn" % i for i in range(NUM_LAYERS)]
    table = FakeBlockTable(bs=bs, max_reqs=8, max_blocks=64, max_tokens=4096)
    runner = FakeRunner(layer_names, table, bs=bs, num_blocks=num_blocks)
    install_wrappers()
    from kvpress_ascend.runtime.context import ensure_runner_state

    ensure_runner_state(runner)
    return runner


def layer_idx_of(name: str) -> int:
    return int(name.split("layers.")[1].split(".")[0])


def view_row_from_meta(meta, i: int) -> np.ndarray:
    """The view row the attention would read for request i (FIA semantics)."""
    row = meta.block_tables[i].numpy()
    return row


def view_slots(meta, i: int) -> np.ndarray:
    """Slots FIA would read for request i: view_row[p//bs]*bs + p%bs, p < view_len."""
    row = view_row_from_meta(meta, i)
    vlen = int(meta.seq_lens_list[i])
    bs = BLOCK_SIZE
    return (np.repeat(row, bs)[:vlen] * bs + np.arange(vlen) % bs).astype(np.int64)


def reference_slots(kept, true_row: np.ndarray, orig_len: int, true_len: int, bs: int) -> np.ndarray:
    """Reference visible set: kept tokens (original order) + all new tokens."""
    slots = []
    for b in sorted(int(x) for x in kept):
        n = min(bs, max(0, orig_len - b * bs))
        slots.extend([int(true_row[b]) * bs + j for j in range(n)])
    for p in range(orig_len, true_len):
        slots.append(int(true_row[p // bs]) * bs + p % bs)
    return np.array(slots, dtype=np.int64)
