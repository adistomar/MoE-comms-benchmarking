# Copyright (c) 2026. FlashInfer NVLink one-sided all-to-all dispatch/combine bencher.
"""
Isolated bencher for FlashInfer's NVLink one-sided MoE all-to-all
(flashinfer.moe_ep NVLinkOneSidedAlltoAll, FlashInfer >= 0.7.2), set up and called like vLLM's
`flashinfer_nvlink_one_sided` all2all backend (FlashInferNVLinkOneSidedManager in
vllm/distributed/device_communicators/all2all.py and FlashInferNVLinkOneSidedPrepareAndFinalize
in vllm/model_executor/layers/fused_moe/prepare_finalize/flashinfer_nvlink_one_sided.py):

  communication : create_communication(BootstrapConfig, MoEEpCommParams(num_experts, top_k,
                  max_tokens_per_rank=per_rank_cap, hidden, bf16), NVLinkOneSidedConfig(cft=False,
                  lamport=False)) -- vLLM keeps the fence transport, BF16 combine. One MNNVL
                  workspace per rank holds the round state, the dispatch receive rows and the
                  combine rows.
  dispatch      : dispatch(hidden, topk_ids int32, topk_weights, runtime_max_tokens_per_rank =
                  the largest per-rank token count of the batch). Each token is written once into
                  every rank that owns one of its experts; a rank receives ep_size *
                  runtime_max_tokens_per_rank rows (rank-major: row = src_rank * T + token) with
                  their GLOBAL top-k ids, padding rows' ids set to -1.
  combine       : combine(expert_output, output=out). Each rank's row is read back and summed on
                  the token's source rank. The expert writes its output straight into the
                  communication's combine-input buffer (get_combine_input_buffer), which the
                  combine then reads without a copy -- vLLM's allocate_fused_expert_output path.

With --expert vllm, each layer runs vllm_fused_moe on all received rows (global expert ids,
local expert range of this rank, routing weights applied) into the combine-input buffer. With
the identity expert, combine sends back whatever is in that buffer (comm only).
"""

import torch
import torch.distributed as dist

from common import Config


class FlashInferOneSidedBencher:
    name = "flashinfer"

    def __init__(self, cfg: Config, group):
        self.cfg = cfg
        self.group = group
        self.device = torch.device("cuda", cfg.local_rank)
        self.num_sms = -1  # FlashInfer picks its own grid (N/A in the CSV)
        self.max_tokens_per_rank = cfg.per_rank_cap
        self._built = False
        self.expert = None  # None = identity expert; else a common.VllmExperts

    def build(self):
        import flashinfer
        from flashinfer.moe_ep import (
            BootstrapConfig, MoEEpCommParams, NVLinkOneSidedAlltoAll, NVLinkOneSidedConfig,
            create_communication,
        )
        cfg = self.cfg
        # vLLM exchanges the MNNVL handles over its CPU (gloo) group; do the same.
        self.cpu_group = dist.new_group(backend="gloo")
        bootstrap = BootstrapConfig(world_size=cfg.ep_size, rank=cfg.rank,
                                    device=cfg.local_rank, process_group=self.cpu_group)
        params = MoEEpCommParams(num_experts=cfg.num_experts, top_k=cfg.topk,
                                 max_tokens_per_rank=cfg.per_rank_cap, hidden_size=cfg.hidden,
                                 dtype=torch.bfloat16)
        config = NVLinkOneSidedConfig(cft=False, lamport=False)
        with torch.device("cpu"):  # vLLM: bootstrap metadata stays on the CPU
            self.comm = create_communication(bootstrap, params, backend=config)
        assert isinstance(self.comm, NVLinkOneSidedAlltoAll), type(self.comm)
        self.flashinfer_version = flashinfer.__version__
        self._built = True

    def setup_batch(self, hidden, topk_idx, topk_weights):
        assert self._built
        cfg = self.cfg
        self.hidden = hidden.contiguous()
        self.topk_ids = topk_idx.to(torch.int32).contiguous()
        self.topk_weights = topk_weights.to(torch.float32).contiguous()
        n = hidden.shape[0]
        # vLLM's runtime_max_tokens_per_rank: the largest local token count over the EP ranks
        # (every rank passes the same value; it fixes the receive rows of the captured graph).
        counts = torch.tensor([n], dtype=torch.int32, device=self.device)
        dist.all_reduce(counts, op=dist.ReduceOp.MAX, group=self.group)
        self.runtime_max_tokens_per_rank = max(1, int(counts.item()))
        self.out = torch.empty(n, cfg.hidden, dtype=torch.bfloat16, device=self.device)
        # Received rows (all ep_size * T of them, padding rows carry -1 ids) are all handed to
        # the expert, as vLLM's Standard activation format does.
        self.recv_rows = torch.full((1,), cfg.ep_size * self.runtime_max_tokens_per_rank,
                                    dtype=torch.int32, device=self.device)

    def dispatch(self):
        return self.comm.dispatch(
            self.hidden, self.topk_ids, self.topk_weights,
            runtime_max_tokens_per_rank=self.runtime_max_tokens_per_rank,
        )

    def step(self, copy_received_to_combine_input=False) -> torch.Tensor:
        """One MoE layer: dispatch -> expert -> combine, paired (one in-flight round).
        Returns the combined [local_tokens, hidden] bf16 output."""
        cfg = self.cfg
        received = self.dispatch()
        combine_input = self.comm.get_combine_input_buffer(torch.bfloat16)
        if self.expert is not None:
            self.expert(received.hidden_states, received.topk_weights, received.topk_ids,
                        self.recv_rows, cfg.rank * cfg.num_local_experts, combine_input,
                        self.hidden.shape[0] * cfg.ep_size, write_unrouted_rows=False)
        elif copy_received_to_combine_input:
            # Identity expert with real data (validation only): send each received row back.
            combine_input.copy_(received.hidden_states)
        return self.comm.combine(combine_input, output=self.out)

    def decode_step(self, num_layers: int) -> torch.Tensor:
        """One decode step: `num_layers` paired dispatch->combine (no once-per-step metadata:
        the count exchange is inside each dispatch). Returns the last layer's output."""
        for _ in range(num_layers):
            out = self.step()
        return out

    # -- correctness -----------------------------------------------------------
    def validate(self):
        """Identity-expert round trip on random per-element tokens: combine sums one copy of
        token t per destination rank, so the result must equal (#distinct dest ranks) * x[t]
        element-wise. Tested with all ranks populated (B=2*ep) and 0-token ranks (B=1)."""
        assert self.expert is None, "validate() checks the identity-expert layer"
        world, rank, dev = self.cfg.ep_size, self.cfg.rank, self.device
        H, K, E = self.cfg.hidden, self.cfg.topk, self.cfg.num_experts
        epr = E // world
        out = []
        for B in (world * 2, 1):
            n = B // world + (1 if rank < B % world else 0)
            gen = torch.Generator(device=dev).manual_seed(880803 + B + rank * 131)
            x = (torch.randn(n, H, generator=gen, device=dev).to(torch.bfloat16)
                 if n > 0 else torch.empty(0, H, device=dev, dtype=torch.bfloat16))
            idx = (torch.stack([torch.randperm(E, generator=gen, device=dev)[:K] for _ in range(n)])
                   if n > 0 else torch.empty(0, K, device=dev, dtype=torch.int64))
            combined = self.functional_roundtrip(x, idx)
            torch.cuda.synchronize()
            if n > 0:
                m = torch.tensor([torch.unique(idx[t] // epr).numel() for t in range(n)],
                                 device=dev, dtype=torch.float32).view(n, 1)
                got, ref = combined.to(torch.float32), m * x.to(torch.float32)
                ok = bool(torch.allclose(got, ref, rtol=0.03, atol=0.05))
                detail = (f"n={n} max|got-m*x|={float((got - ref).abs().max()):.4f} "
                          f"m(#dest_ranks)={m.view(-1)[:min(n, 4)].to(torch.int64).tolist()}")
            else:
                ok, detail = True, "n=0 (idle rank still participates in the A2A)"
            out.append((f"FlashInfer one-sided dispatch->combine B={B:<4d}", ok, detail))
        return out

    def functional_roundtrip(self, hidden, topk_idx):
        """The identity-expert layer with each received row sent back unchanged -> [n,H] ==
        (#distinct dest ranks for t) * x[t] (run.py's cross-impl check)."""
        n = hidden.shape[0]
        self.setup_batch(hidden, topk_idx,
                         torch.ones(n, self.cfg.topk, device=self.device, dtype=torch.float32))
        return self.step(copy_received_to_combine_input=True)[:n]

    def destroy(self):
        if getattr(self, "comm", None) is not None:
            self.comm.destroy()
            self.comm = None
