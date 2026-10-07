# Copyright (c) 2026. NVLS AGv bus-bandwidth saturation sweep.
"""
Measures the NVLS multimem all-gather-v kernel's bus bandwidth as a function of
output message size and its launch-geometry knobs, to answer: does the AGV kernel
(one CTA per token row, one 128-bit chunk per thread) saturate NVLink?

    bus_bw = (g-1)/g * output_msg_bytes / time        (NCCL bus-bandwidth convention)

Reference ceilings printed alongside (GB200 / NVLink5, per-GPU aggregate):
  * 900 GB/s per direction (18 links).
  * The multicast-AG *ingress-bound* ceiling is (g-1)/g * 900 = 675 GB/s at g=4:
    a multimem.st fans out to ALL group members including the sender, so every
    rank's NVLink ingress carries the full output (g shards), not (g-1) shards.
    bus_bw counts (g-1) shards, hence the (g-1)/g scaling of the ingress peak.

Two sweeps (both on the UNMODIFIED vendored kernel):
  A. message-size curve at the production grid cap (148 CTAs), for several row
     widths H (H sets threads/CTA: block = next_pow2(H*2B/16B), i.e. H=1024 -> 128
     threads ... H=8192 -> 1024 threads). H=1024 is the Nemotron-Super shape.
  B. CTA-count scaling at a fixed large (bandwidth-bound) message, num_blocks in
     {16..148}, for the narrowest and widest rows.

Launch:  torchrun --nproc-per-node 4 bench_agv_bw.py --out agv_bw.csv
"""

import argparse
import csv
import os

import torch
import torch.distributed as dist

from common import init_distributed, time_region
from nvls.symmetric_memory import SymmetricMemoryManager
from nvls.torch_symm_triton.variable_collectives import multimem_all_gather_v

PEAK_DIR_GBS = 900.0  # B200 NVLink-5 per-direction aggregate

HIDDENS = [1024, 2048, 4096, 8192]           # -> 128/256/512/1024 threads per CTA
OUT_MBS = [1, 4, 16, 64, 256, 1024, 2048, 4096]
NUM_BLOCKS_SWEEP = [16, 32, 64, 96, 128, 148]
NB_SWEEP_OUT_MB = 1024                        # bandwidth-bound point for sweep B
NB_SWEEP_HIDDENS = [1024, 8192]

MAX_OUT_MB = max(OUT_MBS)


def run_one(group, rank, world, symm, in_flat, hidden, out_mb, num_blocks,
            rank_off_t, ep_max_t, warmup, iters):
    """Time one AGV config; returns (tokens_global, time_us_max, algbw, busbw)."""
    row_bytes = hidden * 2  # bf16
    out_bytes = out_mb << 20
    tokens_global = out_bytes // row_bytes
    if tokens_global % world != 0 or tokens_global == 0:
        return None
    tokens_local = tokens_global // world

    buf = symm.maybe_get_tensor([tokens_global, hidden], torch.bfloat16)
    assert buf["handle"] is not None, "symmetric memory unavailable"
    out_t = buf["tensor"]

    inp = in_flat[: tokens_local * hidden].view(tokens_local, hidden)

    rank_off_t.fill_(rank * tokens_local)
    ep_max_t.fill_(tokens_local)

    def fn():
        multimem_all_gather_v(
            out_t, inp, buf["handle"],
            rank_token_offset=rank_off_t,
            ep_max_tokens=ep_max_t,
            per_rank_max_tokens=tokens_local,
            max_num_blocks=num_blocks,
        )

    _, max_us = time_region(fn, group, warmup=warmup, iters=iters, use_graph=True)
    t_s = max_us / 1e6
    algbw = out_bytes / t_s / 1e9
    busbw = (world - 1) / world * algbw
    return tokens_global, max_us, algbw, busbw


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="agv_bw.csv")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=20)
    args = p.parse_args()

    group, rank, world, local_rank = init_distributed()
    device = torch.device("cuda", local_rank)
    torch.manual_seed(1234 + rank)

    # One symmetric buffer big enough for the largest output, reused (re-viewed)
    # for every config. +16 MB slack for alignment.
    symm = SymmetricMemoryManager.get_buffer(
        "agv_bw", process_group=group, size_mb=MAX_OUT_MB + 16)

    # Largest local shard, allocated once; bf16 random.
    max_shard_elems = (MAX_OUT_MB << 20) // world // 2
    in_flat = torch.randn(max_shard_elems, device=device, dtype=torch.bfloat16)

    rank_off_t = torch.zeros(1, dtype=torch.int32, device=device)
    ep_max_t = torch.zeros(1, dtype=torch.int32, device=device)

    rows = []

    def record(sweep, hidden, out_mb, num_blocks, res):
        tokens_global, max_us, algbw, busbw = res
        chunks = -(-hidden * 2 // 16)  # 128-bit chunks per bf16 row
        block_threads = min(1 << max(0, chunks - 1).bit_length(), 1024)
        rows.append(dict(sweep=sweep, hidden=hidden, out_mb=out_mb,
                         num_blocks=num_blocks, tokens_global=tokens_global,
                         block_threads=block_threads,
                         time_us=round(max_us, 2), algbw_gbs=round(algbw, 2),
                         busbw_gbs=round(busbw, 2),
                         pct_peak=round(100 * busbw / PEAK_DIR_GBS, 1)))
        if rank == 0:
            r = rows[-1]
            print(f"[{sweep}] H={hidden:<5d} out={out_mb:>5d}MB blocks={num_blocks:>3d} "
                  f"t={r['time_us']:>10.1f}us  alg={r['algbw_gbs']:>7.2f}  "
                  f"bus={r['busbw_gbs']:>7.2f} GB/s  ({r['pct_peak']:>5.1f}% of 900)",
                  flush=True)

    if rank == 0:
        print(f"# world={world}  peak/dir={PEAK_DIR_GBS} GB/s  "
              f"multicast-AG ingress ceiling={(world-1)/world*PEAK_DIR_GBS:.0f} GB/s "
              f"(multimem.st delivers the sender's own copy back through the switch)",
              flush=True)

    # Sweep A: message-size curve at 148 CTAs.
    for hidden in HIDDENS:
        for out_mb in OUT_MBS:
            res = run_one(group, rank, world, symm, in_flat, hidden, out_mb, 148,
                          rank_off_t, ep_max_t, args.warmup, args.iters)
            if res:
                record("msg", hidden, out_mb, 148, res)

    # Sweep B: CTA-count scaling at a fixed bandwidth-bound message.
    for hidden in NB_SWEEP_HIDDENS:
        for nb in NUM_BLOCKS_SWEEP:
            res = run_one(group, rank, world, symm, in_flat, hidden,
                          NB_SWEEP_OUT_MB, nb, rank_off_t, ep_max_t,
                          args.warmup, args.iters)
            if res:
                record("cta", hidden, NB_SWEEP_OUT_MB, nb, res)

    if rank == 0 and rows:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"# wrote {len(rows)} rows -> {args.out}", flush=True)

    dist.barrier(group)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
