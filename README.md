# MoE decode dispatch/combine benchmark — DeepEP-v2 (A2A) vs Megatron NVLS & NCCL AllGather

Isolated micro-benchmark comparing three MoE expert-parallel **dispatch/combine**
schemes used for **inference decode** within a single NVLink domain:

- **DeepEP-v2** ("elastic" `ElasticBuffer`) — a true expert-parallel **all-to-all**
  over NCCL-Gin (device-side NCCL symmetric-memory windows; intra-node NVLink uses
  direct-peer PTX load/store, no `multimem`).
- **Megatron NVLS** (`NVLSAllGatherVDispatcher`) — dense/replicated EP: **AllGather-V**
  all tokens to all ranks → compute local experts → **ReduceScatter-V**, built on
  NVLink-SHARP **multicast** (`multimem.st` / `multimem.ld_reduce`).
- **Megatron NCCL** (`NCCLAllGatherDispatcher`, CUDA-graph path) — the **same dense
  algorithm** as NVLS (AllGather → local experts → ReduceScatter) but over **plain NCCL
  collectives** instead of NVLink multicast. The graphable path requires equal per-rank
  token counts, so ranks are **padded** to the per-step max. NVLS-vs-NCCL therefore
  isolates the *transport* (NVLink multicast vs NCCL ring/tree) at equal precision (fp32).

Both Megatron dispatchers are simulated standalone: the NVLS collectives are **vendored**
(`nvls/`, an isolated copy of Megatron's `torch_symm_triton` + `symmetric_memory.py` +
`metadata.py`; zero Megatron deps), and NCCL uses stock `torch.distributed`
AllGather/ReduceScatter. DeepEP is called at the `ElasticBuffer` level and requires a
DeepEP source checkout to build (see Setup).

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
- **NCCL** per layer: 3× `all_gather_into_tensor` (hidden bf16, routing int64, probs fp32)
  → `reduce_scatter_tensor` (fp32). No once-per-step metadata collective (equal per-rank
  counts are guaranteed by padding, discovered with one `all_reduce(MAX)` in setup,
  outside the graph).

## Experimental setup (Nemotron-Super, 1 node, 4× B200)

| | value |
|---|---|
| experts / top-k / hidden | 512 / 22 / 1024 |
| parallelism | EP=4 (1 rank/GPU), TP=1, single NVLink domain |
| dtype | dispatch **bf16** (all three); combine **fp32** for NVLS & NCCL, **bf16** for DeepEP |
| batch axis | **GLOBAL** B ∈ {1,…,8192} tokens across all 4 ranks (balanced; B<4 leaves some ranks with 0 tokens; NCCL pads to the per-step max) |
| layers | 88 MoE layers per decode step (`--num-layers`) |
| routing | uniform: 22 distinct experts/token; identical tensors fed to all impls |
| no host sync | DeepEP `do_cpu_sync=False`; NVLS on-device metadata; NCCL pad-count all-reduce done in setup (outside the graph) |
| comm SMs / blocks | DeepEP `num_sms` swept; **NVLS fixed at 148** CTA blocks (see below); NCCL uses NCCL-internal grid |

DeepEP flags: `use_fp8_dispatch=False` (bf16), `allow_hybrid_mode=False` (single
NVLink domain → flat/direct path, Gin dormant not disabled), `allow_multiple_reduction=True`
(ep=4 ≤ topk=22 ⇒ rank-layout combine), `do_expand=False` (pure collective — permute is
deferred to the GEMM, matching NVLS), `num_sms` swept.

**NVLS block cap.** The NVLS AGV/RSV kernels run **one CTA per token**, so the CTA-grid
ceiling `max_num_blocks` bounds how many SMs the comm can occupy. Upstream Megatron caps it
at 128; we **hardcode it to 148** (the B200 SM count we standardize on) in `nvls/torch_symm_triton/variable_collectives.py`. NVLS is **fixed at 148** (≈ all
SMs) and never swept; it is **independent** of DeepEP's `--deepep-num-sms` — there is no
shared knob, and sweeping DeepEP's SM count does not affect NVLS.

## Correctness check (`--validate`)

`torchrun --nproc_per_node=4 run.py --impl all --validate` runs known-value checks per
impl (random tensors, verified element-wise) then exits without timing. It confirms NVLS
AGV-V gathers each rank's tokens to the right global offset, NVLS RSV-V sums across ranks
and scatters to the right owner, NCCL's padded AllGather→ReduceScatter round-trips
correctly, and DeepEP's dispatch→combine round-trips to `m·x` (`m` = #destination ranks).
With ≥2 impls built it also **cross-checks** that all of them produce the same combine
output on identical inputs (all compute `m·x`; NVLS & NCCL match exactly in fp32, DeepEP
differs only by bf16 rounding). Prints `[PASS]/[FAIL]` per check and a verdict reduced
across ranks (exit 0/1); tested at a full batch (`B=2·ep`) and the 0-token-rank case (`B=1`).

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

**3. Set up the environment** (builds DeepEP on first run; NVLS/NCCL need nothing):
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
# correctness first (no timing) — see the Correctness section
torchrun --nproc_per_node=4 run.py --impl all --validate

# quick smoke
torchrun --nproc_per_node=4 run.py --impl all --batch-sizes 4 --num-layers 88 --reps 3 --warmup 2

# full batch-size sweep -> results.csv (all three impls; DeepEP num_sms defaults to 148)
torchrun --nproc_per_node=4 run.py --impl all \
    --batch-sizes 1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192 --num-layers 88 \
    --reps 20 --warmup 6 --timing graph --out results.csv
python3 plot_results.py --csv results.csv --out results.png

# DeepEP num_sms sweep -> results_sms.csv (each num_sms JIT-compiles once; NVLS/NCCL are
# num_sms-independent reference lines)
torchrun --nproc_per_node=4 run.py --impl all \
    --batch-sizes 1,16,128 --deepep-num-sms 4,16,64,128 --num-layers 88 \
    --reps 10 --warmup 4 --timing graph --out results_sms.csv
python3 plot_results.py --csv results_sms.csv --x num_sms --out results_sms.png
```
Or submit `run.sbatch` (edit the SBATCH headers / `CONTAINER_IMAGE` for your cluster,
then `cd moe-comms-bench && sbatch run.sbatch`).

`run.py` flags: `--impl {deepep,nvls,nccl,both,all}` (both = deepep+nvls; all = +nccl),
`--batch-sizes`, `--num-layers`, `--deepep-num-sms` (comma list, even, clamped to device
SM count), `--timing {graph,eager}`, `--reps`, `--warmup`, `--out`, `--validate`
(correctness checks then exit — see above).

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
- `bench_deepep.py` / `bench_nvls.py` / `bench_nccl.py` — the three benchers; each exposes `decode_step(num_layers)` + `validate()` + `functional_roundtrip()`.
- `nvls/` — vendored, isolated NVLS collectives (zero Megatron deps; only change vs upstream is the Triton-3.6 widen, the 128→148 block cap, and import isolation).
- `plot_results.py` — plot `--x B` (default) or `--x num_sms`, in milliseconds.
- `deepep_env.sh` — DeepEP build+runtime env (idempotent; relocatable).
- `install_deepep_ngc.sh` — one-shot DeepEP wheel install + build (called by `deepep_env.sh`).
- `run.sbatch` — SLURM batch template.

## Caveats
- **Combine precision.** NVLS and NCCL reduce in **fp32** (high-precision, 2× combine
  bytes); DeepEP keeps activations in **bf16**. NVLS-vs-NCCL is thus a clean transport
  comparison (same dense algorithm, same precision); the fp32-vs-bf16 gap to DeepEP is
  left as-is since it favors DeepEP.
- **NCCL padding.** The graphable NCCL path requires equal per-rank token counts, so ranks
  are padded to the per-step max (`ceil(B/ep)`). Under the balanced global-B split counts
  differ by ≤1, so padding is negligible; the padded rows are gathered/reduced then
  truncated (they never pollute real-token outputs).
