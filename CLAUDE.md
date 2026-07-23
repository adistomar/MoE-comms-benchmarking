# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A single-node micro-benchmark comparing **three MoE expert-parallel dispatch/combine
schemes** for **inference decode** within one NVLink domain (validated on 4× B200 / GB200,
EP=4). It isolates just the communication — no expert GEMM, no shared-expert machinery:

- **DeepEP-v2** (`bench_deepep.py`) — true expert-parallel **all-to-all** via `deep_ep.ElasticBuffer`.
- **Megatron NVLS** (`bench_nvls.py`) — dense EP: **AllGather-V → local experts → ReduceScatter-V**
  over NVLink-SHARP **multicast** (`multimem.*`), using the vendored `nvls/` package.
- **Megatron NCCL** (`bench_nccl.py`) — the *same dense algorithm* as NVLS but over plain
  `torch.distributed` AllGather/ReduceScatter (CUDA-graph path; ranks padded to equal counts).

`README.md` is the authoritative spec (measurement definition, precision caveats, DeepEP flag
rationale). Read it before changing benchmark semantics.

## Commands

There is no build system, linter, or test framework — this is a research benchmark, and
`--validate` is the correctness gate. All runs launch under `torchrun` on 4 GPUs.

```bash
# DeepEP path only: set up build+runtime env (idempotent; ~15 min first build, cached after).
# NVLS and NCCL paths need NONE of this — nvls/ is fully vendored, NCCL is stock torch.
source ./deepep_env.sh          # override DeepEP checkout with: export DEEPEP_DIR=/path/to/DeepEP

# Correctness FIRST (known-value + cross-impl checks, then exits — no timing):
torchrun --nproc_per_node=4 run.py --impl all --validate

# Quick smoke:
torchrun --nproc_per_node=4 run.py --impl all --batch-sizes 4 --num-layers 88 --reps 3 --warmup 2

# Full batch-size sweep -> CSV -> plot:
torchrun --nproc_per_node=4 run.py --impl all \
    --batch-sizes 1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192 \
    --num-layers 88 --reps 20 --warmup 6 --timing graph --out results.csv
python3 plot_results.py --csv results.csv --out results.png

# DeepEP num_sms sweep (NVLS/NCCL are num_sms-independent reference lines):
torchrun --nproc_per_node=4 run.py --impl all --batch-sizes 1,16,128 \
    --deepep-num-sms 4,16,64,128 --num-layers 88 --reps 10 --warmup 4 --out results_sms.csv
python3 plot_results.py --csv results_sms.csv --x num_sms --out results_sms.png
```

`--impl` values: `deepep | nvls | nccl | both` (deepep+nvls) `| all` (+nccl). To run only the
vendored/stock paths without a DeepEP checkout, use `--impl nvls`, `nccl`, or a comma of them.
Cluster batch template: `run.sbatch` (edit SBATCH headers / `CONTAINER_IMAGE`, then `sbatch run.sbatch`).

## Architecture

`run.py` (driver) builds one **bencher** per impl and times **one full decode step** —
`--num-layers` MoE layers (default 88), each a paired dispatch→(identity expert)→combine —
captured and replayed as a **single CUDA graph**, reported as **max latency across ranks**.
`common.py` holds the shared `Config`, distributed init, deterministic input/routing generation,
the global→per-rank token split, and `time_region` (CUDA-event timing, graph or eager).

**The bencher protocol is the core abstraction and the extension point.** `run.py` treats all
three impls uniformly through this duck-typed interface — to add a fourth scheme, implement the
same shape:

- attributes: `name` (str), `num_sms` (int; `-1` = N/A as with NCCL)
- `build()` — one-time (collective) allocation; `setup_batch(hidden, topk_idx, topk_weights)` per batch
- `decode_step(num_layers)` — the timed unit; `step()` — one layer's dispatch→combine
- `validate()` → list of `(name, ok, detail)` known-value checks
- `functional_roundtrip(hidden, topk_idx)` → `[n,H]` = `(#distinct dest ranks)·x`, the common
  quantity all three produce, used for the cross-impl equivalence check in `run.py`
- optional `set_num_sms(n)` (DeepEP sweeps it; others no-op) and `destroy()`

`nvls/` is a **vendored, isolated copy of Megatron's NVLS collectives** (zero Megatron deps):
`torch_symm_triton/` (the Triton `multimem` AGV-V/RSV-V kernels), `symmetric_memory.py`
(`SymmetricMemoryManager` buffer registry), `metadata.py` (fused once-per-step token-count kernel).

## Critical, non-obvious constraints

- **DeepEP must be imported before `init_process_group`.** `run.py` does `import deep_ep` up
  top for exactly this reason: importing it after torch inits NCCL links it against a different
  NCCL than the process group, and reading the comm handle segfaults during `ElasticBuffer`
  construction. Relatedly, `run.py` issues a `dist.barrier` to force NCCL comm creation *before*
  building benchers (torch creates the comm lazily; DeepEP's ctor needs it to already exist).

- **The vendored `nvls/` code has three deliberate local deltas from upstream Megatron — do not
  "fix" them away, and re-apply them if you re-vendor:**
  1. **Triton-3.6 int64 pointer-widen** (`# Required Triton-3.6 fix` in `nvls/metadata.py:76`
     and `nvls/torch_symm_triton/variable_collectives.py` at lines ~204/434). Triton 3.6
     specializes a small raw pointer int as i32, but `tt.int_to_ptr` needs i64 → without the
     widen the NVLS path **will not compile**.
  2. **Block cap raised 128 → 148** (`MAX_NUM_BLOCKS`, three sites in `variable_collectives.py`).
     The AGV/RSV kernels run one CTA per token, so this ceiling bounds how many SMs the comm
     occupies. NVLS is **fixed at 148** (≈ all B200 SMs) via `NVLS_MAX_BLOCKS` in `bench_nvls.py`
     and never swept — it is **independent** of DeepEP's `--deepep-num-sms` (no shared knob).
  3. Import isolation (`_compat.py` vendors Megatron's `null_decorator`).

- **`B` (`--batch-sizes`) is GLOBAL** — total decode tokens across *all* EP ranks, split as
  evenly as possible. For `B < ep` some ranks hold 0 tokens (still participate in the collective).

- **Timing model:** NVLS runs its `fused_metadata_update` **once per decode step** (first layer,
  routing-independent), not per layer. NCCL has no per-step metadata collective — equal per-rank
  counts are guaranteed by padding to the per-step max, discovered with one `all_reduce(MAX)` in
  `setup_batch` *outside* the CUDA graph (no host sync on the hot path).

- **Combine precision differs by design:** NVLS & DeepEP combine in **bf16** (NVLS accumulates
  the cross-rank sum in fp32 via `acc::f32`); NCCL combines in **fp32**. So NVLS↔DeepEP match on
  precision, but NVLS-vs-NCCL contrasts *both* transport and dtype. To make NVLS-vs-NCCL a pure
  transport comparison, switch NCCL's `reduce_scatter_tensor` to bf16 in `bench_nccl.py`.

## Repo-history artifacts (don't chase these)

Docstrings reference paths like `bench/run.py`, `bench/README.md`, and `docs/moe_dispatcher_deep_dive*.md`.
This repo was extracted from a larger tree where these files lived under `bench/`; here everything
is flat at the root and **there is no `docs/` directory**. Treat those paths as historical.
