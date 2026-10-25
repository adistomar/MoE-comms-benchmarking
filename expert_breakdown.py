"""Per-kernel cost of Megatron's vllm_fused_moe in the two input layouts the e2e run feeds it.

One GPU, no communication. Models rank 0 of an EP group (experts [0, E/EP)) for Super's MoE
(512 experts, top-22, latent 1024, ffn 2688, squared ReLU), with uniform routing:

  gathered (nvls / nvl_a2a): all B global rows, global expert ids, valid_tokens = B
  gathered_skip (nvl_a2a)  : the same, with write_unrouted_rows=False
  compact  (deepep)        : only the rows routed to this rank, local ids (-1 elsewhere),
                             valid_tokens = number of such rows (~0.3 B at EP=64)

Both layouts hold the same (token, local expert) pairs, so the GEMM work is identical, and
both use num_tokens_hint = B, as Megatron does. Each layer's call is captured in a CUDA graph
and replayed; a torch.profiler pass splits the time by kernel.

    python3 expert_breakdown.py --ep 64 --per-rank-cap 2048 --batch-sizes 1024,4096,8192
"""

import argparse
import collections
import inspect
import sys

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ep", type=int, default=64)
    p.add_argument("--per-rank-cap", type=int, default=2048)
    p.add_argument("--batch-sizes", default="64,256,1024,2048,4096,8192,16384,32768,65536,131072")
    p.add_argument("--megatron-path", required=True)
    p.add_argument("--reps", type=int, default=50)
    args = p.parse_args()
    sys.path.insert(0, args.megatron_path)
    from megatron.core.inference.moe.fused_moe import ActivationType
    from megatron.core.inference.moe.vllm_fused_moe import VllmFusedMoeBuffers, vllm_fused_moe

    E, K, H, FFN = 512, 22, 1024, 2688
    L = E // args.ep  # local experts
    rows = args.ep * args.per_rank_cap  # Megatron's buffer height (gathered and compact alike)
    dev = torch.device("cuda", 0)
    gen = torch.Generator(device=dev).manual_seed(0)
    fc1 = (torch.randn(L, FFN, H, generator=gen, device=dev) * H ** -0.5).to(torch.bfloat16)
    fc2 = (torch.randn(L, H, FFN, generator=gen, device=dev) * FFN ** -0.5).to(torch.bfloat16)
    VllmFusedMoeBuffers.allocate_buffers(max_tokens=rows, topk=K, fc1_output_size=FFN,
                                         hidden_size=H, num_local_experts=L, device=dev)
    hidden = torch.randn(rows, H, generator=gen, device=dev).to(torch.bfloat16)
    has_skip = "write_unrouted_rows" in inspect.signature(vllm_fused_moe).parameters
    out = torch.empty(rows, H, dtype=torch.bfloat16, device=dev)

    print(f"# EP={args.ep} local_experts={L} buffer_rows={rows} reps={args.reps}")
    print("layout,B,valid_rows,total_us,kernel,kernel_us")
    for B in [int(b) for b in args.batch_sizes.split(",")]:
        assert B <= rows
        scores = torch.rand(B, E, generator=gen, device=dev)
        w, ids = torch.topk(scores, K, dim=-1)
        w = torch.softmax(w, dim=-1)
        is_local = ids < L
        routed = is_local.any(dim=1)
        layouts = {}
        # gathered: global ids for every row
        r_g = torch.full((rows, K), -1, dtype=torch.int64, device=dev)
        p_g = torch.zeros(rows, K, dtype=torch.float32, device=dev)
        r_g[:B], p_g[:B] = ids, w
        layouts["gathered"] = (r_g, p_g, B, True)  # nvls (and nvl_a2a before the skip)
        layouts["gathered_skip"] = (r_g, p_g, B, False)  # nvl_a2a: unrouted rows skipped
        # compact: received rows only, local ids
        n = int(routed.sum())
        r_c = torch.full((rows, K), -1, dtype=torch.int64, device=dev)
        p_c = torch.zeros(rows, K, dtype=torch.float32, device=dev)
        r_c[:n] = torch.where(is_local[routed], ids[routed], torch.full_like(ids[routed], -1))
        p_c[:n] = w[routed]
        layouts["compact"] = (r_c, p_c, n, True)

        for name, (routing, probs, valid, write_unrouted) in layouts.items():
            if not write_unrouted and not has_skip:
                continue  # Megatron before write_unrouted_rows
            valid_t = torch.tensor([valid], dtype=torch.int32, device=dev)

            def layer():
                vllm_fused_moe(hidden, probs, fc1, fc2,
                               activation_type=ActivationType.SQUARED_RELU,
                               num_local_experts=L, local_expert_start=0,
                               valid_tokens=valid_t, routing_map=routing, out=out,
                               num_tokens_hint=B,
                               **({"write_unrouted_rows": write_unrouted} if has_skip else {}))

            for _ in range(3):
                layer()
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                layer()
            g.replay()
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(args.reps):
                g.replay()
            end.record()
            torch.cuda.synchronize()
            total_us = start.elapsed_time(end) * 1e3 / args.reps

            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
                for _ in range(args.reps):
                    layer()
                torch.cuda.synchronize()
            per_kernel = collections.Counter()
            for ka in prof.key_averages():
                if ka.self_device_time_total > 0:  # kernels (CPU ops report 0 here)
                    per_kernel[ka.key.split("(")[0][:60]] += ka.self_device_time_total / args.reps
            for kname, us in per_kernel.most_common():
                print(f"{name},{B},{valid},{total_us:.1f},{kname},{us:.1f}", flush=True)


if __name__ == "__main__":
    main()
