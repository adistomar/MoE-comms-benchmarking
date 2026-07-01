# MoE decode dispatch/combine benchmark — DeepEP-v2 (A2A) vs Megatron-NVLS (AGv/RSv)

Isolated micro-benchmark comparing the two MoE expert-parallel **dispatch/combine**
schemes used for **inference decode** within a single NVLink domain:

- **DeepEP-v2** ("elastic" `ElasticBuffer`) — a true expert-parallel **all-to-all**
  over NCCL-Gin (device-side NCCL symmetric-memory windows; intra-node NVLink uses
  direct-peer PTX load/store, no `multimem`).
- **Megatron NVLS** (`NVLSAllGatherVDispatcher`) — dense/replicated EP: **AllGather-V**
  all tokens to all ranks → compute local experts → **ReduceScatter-V**, built on
  NVLink-SHARP **multicast** (`multimem.st` / `multimem.ld_reduce`).

The NVLS collectives are **vendored** here (`nvls/`, an isolated copy of Megatron's
`torch_symm_triton` + `symmetric_memory.py` + `metadata.py`; zero Megatron deps), so
the NVLS path runs standalone. DeepEP is called at the `ElasticBuffer` level and
requires a DeepEP source checkout to build (see Setup).

## What is measured

**One full decode step** = `--num-layers` MoE layers (default **88**, Nemotron-Super
depth), each doing a **paired dispatch → (identity expert) → combine**, captured and
replayed as a **single CUDA graph**. Reported latency is **milliseconds per decode
step** (all layers), taken as the **max across ranks** (critical path). Per-layer =
step / num_layers.

- **DeepEP** per layer: `dispatch_impl` + copy epilogue (incl. routing-dependent
  notify/count-exchange) then `combine_impl` + reduce epilogue.
- **NVLS** per step: `fused_metadata_update` **once** (token-count sum/prefix/max,
  routing-independent — Megatron runs it only at the first MoE layer) then, per layer,
  `multimem_all_gatherv_3tensor` (AGV-V) → `multimem_reduce_scatter_v` (RSV-V).

## Experimental setup (Nemotron-Super, 1 node, 4× B200)

| | value |
|---|---|
| experts / top-k / hidden | 512 / 22 / 1024 |
| parallelism | EP=4 (1 rank/GPU), TP=1, single NVLink domain |
| dtype | dispatch **bf16** both; combine disparate (NVLS fp32 RSV, DeepEP bf16) |
| batch axis | **GLOBAL** B ∈ {1,2,4,8,16,32,64,128} tokens across all 4 ranks (balanced; B<4 leaves some ranks with 0 tokens) |
| layers | 88 MoE layers per decode step (`--num-layers`) |
| routing | uniform: 22 distinct experts/token; identical tensors fed to both impls |
| no host sync | DeepEP `do_cpu_sync=False`; NVLS on-device metadata |

DeepEP flags: `use_fp8_dispatch=False` (bf16), `allow_hybrid_mode=False` (single
NVLink domain → flat/direct path, Gin dormant not disabled), `allow_multiple_reduction=True`
(ep=4 ≤ topk=22 ⇒ rank-layout combine), `do_expand=False` (pure collective — permute is
deferred to the GEMM, matching NVLS), `num_sms` swept.

## Results (this hardware)

Per **88-layer decode step**, CUDA-graph, ms (max across ranks):

| | DeepEP | NVLS |
|---|---|---|
| per decode step (88 layers) | ~3.0–3.2 ms | ~1.0–1.1 ms |
| per layer | ~35 µs | ~12 µs |

**NVLS ≈ 2.9× faster** for the full-model MoE comm per decode token, roughly flat
across batch size (decode at the tested batch sizes is latency/barrier-bound, not payload-bound). See
`results.png`.

## Setup

**Requirements.** A GPU node with a **single NVLink domain** (this was validated on
4× B200 / GB200) and an NGC-style PyTorch container (validated: CUDA 13, torch 2.11,
**Triton 3.6**, `torch.distributed._symmetric_memory` + multicast, NVRTC/ptxas). Multi-GPU
launched with `torchrun`. NVLS needs Hopper+ (SM ≥ 9) with NVLink + symmetric memory.

**1. Clone this repo and DeepEP side-by-side:**
```bash
git clone <THIS_REPO_URL> moe-comms-bench
git clone https://github.com/deepseek-ai/DeepEP.git DeepEP   # checkout the "elastic"
cd DeepEP && git checkout af9a0403 && cd ..                   # v2 ElasticBuffer commit
# layout:  ./moe-comms-bench   (this repo)   and   ./DeepEP   (sibling)
```
The DeepEP checkout must contain `deep_ep/buffers/elastic.py` (the v2 "elastic"
dispatcher). `deepep_env.sh` looks for DeepEP at `../DeepEP` by default; override with
`export DEEPEP_DIR=/path/to/DeepEP`.

**2. Get an interactive allocation in the container** (adjust account/partition/image):
```bash
srun -p batch --account=<ACCT> --qos=interactive -t 2:00:00 --nodes=1 --exclusive \
  --gpus-per-node=4 --container-image <CONTAINER.sqsh> \
  --container-mounts "/home:/home,/lustre:/lustre" --pty /bin/bash
```

**3. Set up the environment** (builds DeepEP on first run; NVLS needs nothing):
```bash
cd moe-comms-bench
source ./deepep_env.sh      # installs nvidia-nccl-cu13>=2.30.4 + nvshmem-cu13 wheels,
                            # orders the new NCCL first, builds DeepEP (~15 min first
                            # time; cached after), sets a persistent JIT cache.
```
The first build compiles DeepEP's `_C` extension into `$DEEPEP_DIR/deep_ep/` and caches
JIT kernels under `bench/.deepep_jit_cache` — both persist, so re-running in a fresh
(ephemeral) container just reinstalls the wheels and reuses the prebuilt extension.
If DeepEP's runtime-vs-linked NCCL check complains, `export EP_SUPPRESS_NCCL_CHECK=1`.

**4. Run:**
```bash
# quick smoke
torchrun --nproc_per_node=4 run.py --impl both --batch-sizes 4 --num-layers 88 --reps 3 --warmup 2

# full batch-size sweep (default) -> results.csv
torchrun --nproc_per_node=4 run.py --impl both \
    --batch-sizes 1,2,4,8,16,32,64,128 --deepep-num-sms 16 --num-layers 88 \
    --reps 20 --warmup 6 --timing graph --out results.csv
python3 plot_results.py --csv results.csv --out results.png

# DeepEP num_sms sweep -> results_sms.csv (each num_sms JIT-compiles once)
torchrun --nproc_per_node=4 run.py --impl both \
    --batch-sizes 1,16,128 --deepep-num-sms 4,16,64,128 --num-layers 88 \
    --reps 10 --warmup 4 --timing graph --out results_sms.csv
python3 plot_results.py --csv results_sms.csv --x num_sms --out results_sms.png
```
Or submit `run.sbatch` (edit the SBATCH headers / `CONTAINER_IMAGE` for your cluster,
then `cd moe-comms-bench && sbatch run.sbatch`).

`run.py` flags: `--impl {deepep,nvls,both}`, `--batch-sizes`, `--num-layers`,
`--deepep-num-sms` (comma list, even, clamped to device SM count),
`--timing {graph,eager}`, `--reps`, `--warmup`, `--out`.

## Required patch (Triton 3.6): int64 pointer-widen
The vendored NVLS multimem kernels (`nvls/torch_symm_triton/variable_collectives.py`,
`nvls/metadata.py`) take raw pointer *ints* and do `x.to(tl.pointer_type(...))`. Triton
3.6 specializes a scalar int arg as **i32** when its value fits in 32 bits (a low GPU
VA), but `tt.int_to_ptr` needs i64 → compile error. Fix (tagged `# Required Triton-3.6
fix`): widen each raw pointer int to i64 at kernel entry — value-preserving, and a no-op
on Triton versions that already type it i64. Without it the NVLS path will not compile.

## Files
- `run.py` — torchrun driver (times one full decode step as a CUDA graph).
- `common.py` — config, global→per-rank token split, routing/input gen, `time_region` (CUDA-event timing).
- `bench_deepep.py` / `bench_nvls.py` — the two benchers; each exposes `decode_step(num_layers)`.
- `nvls/` — vendored, isolated NVLS collectives (zero Megatron deps; only change vs upstream is the Triton-3.6 widen + import isolation).
- `plot_results.py` — plot `--x B` (default) or `--x num_sms`, in milliseconds.
- `deepep_env.sh` — DeepEP build+runtime env (idempotent; relocatable).
- `install_deepep_ngc.sh` — one-shot DeepEP wheel install + build (called by `deepep_env.sh`).
- `run.sbatch` — SLURM batch template.

## Caveat
- For NVLS dispatcher's combine, RSV is fp32 for high-precision reduction whereas Deep-EP v2 keeps activations in BF16 prior to reduction, so combine-side reduction for NVLS has 2x higher comm volume. I decided to leave this precision disparity as-is, as it anyway favors DeepEP.
