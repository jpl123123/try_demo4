"""L1/L2 simulation harness for squeeze-ascend.

Fakes are field-faithful to vllm-ascend v0.23.0; the decoder layers mimic the
Qwen3.5 residual-style forward so the real layer-hook wrapper (S6) is
exercised end-to-end.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import torch

from squeeze_ascend.runtime.pass_engine import make_execute_model_wrapper
from squeeze_ascend.runtime.view import make_build_attn_metadata_wrapper

BLOCK_SIZE = 16
NUM_LAYERS = 4
NUM_BLOCKS = 64
HIDDEN = 32


class FakeAscendMetadata:
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

    def grow_row(self, row_idx, n_blocks):
        ids = self.alloc_blocks(n_blocks)
        start = int(self.num_blocks_per_row[row_idx])
        self.block_table.np[row_idx, start : start + n_blocks] = ids
        self.num_blocks_per_row[row_idx] = start + n_blocks

    def commit_block_table(self, num_reqs):
        self.block_table.gpu[:num_reqs] = torch.from_numpy(self.block_table.np[:num_reqs].copy())

    def get_device_tensor(self, num_reqs=None):
        return self.block_table.gpu if num_reqs is None else self.block_table.gpu[:num_reqs]

    def compute_slot_mapping(self, num_reqs, query_start_loc, positions):
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


class FakeMultiGroupBlockTable:
    def __init__(self, table):
        self._t = table

    def __getitem__(self, idx):
        return self._t

    def commit_block_table(self, num_reqs):
        self._t.commit_block_table(num_reqs)


class FakeLayer(torch.nn.Module):
    """Qwen3.5-style residual layer with a per-layer random projection.

    out = residual + factor * (hidden @ W_i); the random W_i makes the
    cos-sim(residual, out) depend on the factor (non-collinear outputs).
    """

    def __init__(self, idx, factor):
        super().__init__()
        self.idx = idx
        self.factor = factor
        rng = np.random.default_rng(100 + idx)
        self.W = torch.from_numpy(rng.standard_normal((HIDDEN, HIDDEN)) * 0.5).float()

    def forward(self, hidden_states, residual=None, positions=None, **kwargs):
        if residual is None:
            residual = hidden_states
        normed = hidden_states @ self.W
        out = residual + self.factor * normed
        return out, residual


class FakeModel(torch.nn.Module):
    def __init__(self, factors):
        super().__init__()
        self.model = SimpleNamespace()
        self.model.layers = torch.nn.ModuleList(
            [FakeLayer(i, f) for i, f in enumerate(factors)]
        )


class FakeRunner:
    def __init__(self, layer_names, table, factors, bs=BLOCK_SIZE):
        self.layer_names = list(layer_names)
        self.bs = bs
        self.device = torch.device("cpu")
        self.use_cp = False
        self.attn_state = "PrefillNoCache"
        self.actual_seq_lengths_q = []
        self.table = table
        self.input_batch = SimpleNamespace(
            req_ids=[],
            req_id_to_index={},
            num_computed_tokens_cpu=np.zeros(8, dtype=np.int64),
            num_prompt_tokens=np.zeros(8, dtype=np.int64),
            block_table=FakeMultiGroupBlockTable(table),
            num_reqs=0,
        )
        self.kv_cache_config = SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(
                    layer_names=self.layer_names,
                    kv_cache_spec=SimpleNamespace(block_size=bs),
                )
            ]
        )
        self.compilation_config = SimpleNamespace(
            static_forward_context={name: SimpleNamespace(kv_cache=None) for name in self.layer_names}
        )
        self.requests = {}
        self.model = FakeModel(factors)

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
            attn_metadata[layer_name] = FakeAscendMetadata(
                attn_state=self.attn_state,
                seq_lens=seqs,
                seq_lens_cpu=seqs,
                seq_lens_list=seqs_np.tolist(),
                block_tables=self.table.get_device_tensor(),
                slot_mapping=self.table.slot_mapping.gpu,
                actual_seq_lengths_q=q_lens,
            )
        return attn_metadata, None

    def execute_model(self, scheduler_output, intermediate_tensors=None):
        num_scheduled = scheduler_output.num_scheduled_tokens
        req_ids = self.input_batch.req_ids
        n = len(req_ids)
        sched_np = np.array([int(num_scheduled.get(r, 0)) for r in req_ids], dtype=np.int64)
        for i, rid in enumerate(req_ids):
            before = int(self.input_batch.num_computed_tokens_cpu[i])
            need = (before + int(sched_np[i]) + self.bs - 1) // self.bs
            have = int(self.table.num_blocks_per_row[i])
            if need > have:
                self.table.grow_row(i, need - have)
        self.table.commit_block_table(n)

        total = int(sched_np.sum())
        positions = []
        for i, rid in enumerate(req_ids):
            base = int(self.input_batch.num_computed_tokens_cpu[i])
            for j in range(int(sched_np[i])):
                positions.append(base + j)
        pos_t = torch.tensor(positions, dtype=torch.int64)
        qsl = np.concatenate([[0], np.cumsum(sched_np)]).astype(np.int64)
        self.table.compute_slot_mapping(n, torch.from_numpy(qsl), pos_t)

        self.actual_seq_lengths_q = sched_np.tolist()
        self.attn_state = (
            "DecodeOnly"
            if total == n and all(
                int(self.input_batch.num_computed_tokens_cpu[i]) >= int(self.input_batch.num_prompt_tokens[i])
                for i in range(n)
            )
            else "ChunkedPrefill"
        )

        attn_metadata, _ = self._build_attention_metadata(total, n, 0)
        self._last_attn_metadata = attn_metadata

        if total == 0:
            return None

        # _model_forward: per layer with the residual-style contract
        rng = np.random.default_rng(7)
        hidden = torch.randn(total, HIDDEN)
        residual = None
        for layer_name in self.layer_names:
            idx = int(layer_name.split("layers.")[1].split(".")[0])
            layer = self.model.model.layers[idx]
            hidden, residual = layer(
                hidden_states=hidden, residual=residual, positions=positions
            )
        return None

    def sample_tokens(self):
        for i, rid in enumerate(self.input_batch.req_ids):
            self.input_batch.num_computed_tokens_cpu[i] += int(self.actual_seq_lengths_q[i])


class SchedulerOutput:
    def __init__(self, num_scheduled: dict, finished=()):
        self.num_scheduled_tokens = num_scheduled
        self.total_num_scheduled_tokens = sum(num_scheduled.values())
        self.finished_req_ids = list(finished)


def add_request(runner, req_id, prompt_len):
    idx = len(runner.input_batch.req_ids)
    runner.input_batch.req_ids.append(req_id)
    runner.input_batch.req_id_to_index[req_id] = idx
    runner.input_batch.num_prompt_tokens[idx] = prompt_len
    runner.input_batch.num_computed_tokens_cpu[idx] = 0
    runner.input_batch.num_reqs = len(runner.input_batch.req_ids)


def run_step(runner, sched):
    runner.execute_model(SchedulerOutput(sched))
    runner.sample_tokens()


def install_wrappers():
    if not getattr(FakeRunner._build_attention_metadata, "_squeeze_ascend_patched", False):
        FakeRunner._build_attention_metadata = make_build_attn_metadata_wrapper(  # type: ignore[assignment]
            FakeRunner._build_attention_metadata
        )
        FakeRunner._build_attention_metadata._squeeze_ascend_patched = True  # type: ignore[attr-defined]
    if not getattr(FakeRunner.execute_model, "_squeeze_ascend_patched", False):
        FakeRunner.execute_model = make_execute_model_wrapper(FakeRunner.execute_model)  # type: ignore[assignment]
        FakeRunner.execute_model._squeeze_ascend_patched = True  # type: ignore[attr-defined]


def make_runner(cfg_env=None, factors=None, bs=BLOCK_SIZE):
    from squeeze_ascend.envs import reset_config
    from squeeze_ascend import registry

    for k in list(os.environ):
        low = k.lower()
        if low in ("squeeze", "squeeze_ascend") or low.startswith("squeeze_ascend_"):
            os.environ.pop(k, None)
    if cfg_env:
        for k, v in cfg_env.items():
            os.environ[k] = v
    reset_config()
    registry.reset()
    factors = factors or [0.9, 0.9, 0.1, 0.1]
    layer_names = ["model.layers.%d.self_attn.attn" % i for i in range(len(factors))]
    table = FakeBlockTable(bs=bs, max_reqs=8, max_blocks=64, max_tokens=4096)
    runner = FakeRunner(layer_names, table, factors, bs=bs)
    install_wrappers()
    from squeeze_ascend.runtime.context import ensure_runner_state

    ensure_runner_state(runner)
    return runner


def view_slots(meta, i):
    """Slots FIA would read for request i through the window view."""
    row = meta.block_tables[i].numpy()
    vlen = int(meta.seq_lens_list[i])
    bs = BLOCK_SIZE
    return (np.repeat(row, bs)[:vlen] * bs + np.arange(vlen) % bs).astype(np.int64)


def reference_window_slots(true_row, true_len, bs, window, start_size):
    """Reference visible set: sink tokens + recent tokens (block boundaries)."""
    from squeeze_ascend import core

    _, _, recent_first = core.window_view_layout(true_len, bs, window, start_size)
    sink_blocks = (start_size + bs - 1) // bs if start_size > 0 else 0
    m = (true_len + bs - 1) // bs
    slots = []
    for b in range(0, sink_blocks):
        n = min(bs, max(0, true_len - b * bs))
        slots.extend([int(true_row[b]) * bs + j for j in range(n)])
    for b in range(recent_first, m):
        n = min(bs, max(0, true_len - b * bs))
        slots.extend([int(true_row[b]) * bs + j for j in range(n)])
    return np.array(slots, dtype=np.int64)
