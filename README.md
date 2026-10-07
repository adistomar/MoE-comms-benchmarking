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
  contrasts the *transport* (NVLink multicast vs NCCL ring/tree) on the same dense
  algorithm — though they now differ in combine precision (NVLS bf16, NCCL fp32; see Caveats).

In addition to those fixed strategies, two **all-to-all-v** variants and one
**regime-adaptive** kernel are benchmarked:

- **A2Av** (`a2av_push`) — dense-layout all-to-all-v: hidden activations are **unicast**
  (`st.relaxed.sys` through `_SymmetricMemory.buffer_ptrs_dev`) only to the ranks hosting
  one of a token's top-k experts, while routing/probs stay multicast so the compute path is
  unchanged. The combine **pushes** bf16 expert outputs into per-source planes of each
  owner's receive buffer, which the owner then reduces locally in fp32. A builder kernel
  publishes compact per-destination index lists so the push touches only real tokens.
- **Dynamic** (`dynamic`) — **both of the above in one kernel set**, choosing per step
  **on the device**. See below; this is the interesting one.

Both Megatron dispatchers are simulated standalone: the NVLS collectives are **vendored**
(`nvls/`, an isolated copy of Megatron's `torch_symm_triton` + `symmetric_memory.py` +
`metadata.py`; zero Megatron deps), and NCCL uses stock `torch.distributed`
AllGather/ReduceScatter. DeepEP is called at the `ElasticBuffer` level and requires a
DeepEP source checkout to build (see Setup).

## The `dynamic` (regime-adaptive) kernels

AGv/RSv and A2Av win in *different* regimes, and a served model visits both regimes from
one step to the next — small decode steps and large prefill steps. `dynamic` therefore
picks per step, inside the kernel:

- **Switch signal**: `step_metadata[0]`, the token count **summed over all EP ranks**,
  which `fused_metadata_update` already publishes once per step. It is a reduction over
  every rank, so all ranks read the same value and take the same branch — required, or the
  release/acquire barriers would not pair up.
- **CUDA-graph safe by construction**: grid, block size, `num_warps` and every kernel
  argument are fixed at capture time; only a value *read from device memory* differs
  between replays. One captured graph runs AGv/RSv on a decode step and A2Av on a prefill
  step — no per-regime graphs, no host-side branch, no re-capture.
- **Dispatch** is one kernel with one barrier in both regimes (the two strategies already
  shared their launch geometry). The A2Av branch runs the send-index builder *inline*, so
  it rides the dispatch kernel's existing release barrier: no extra launch, no extra sync.
  That makes `dynamic`'s A2Av path measurably **faster than `a2av_push`** itself.
- **Combine** is two launches in both regimes: stage 1 pushes (A2Av) or returns
  immediately (AGv); stage 2 reduces the pushed planes (A2Av) or runs the in-switch
  reduce-scatter (AGv). Both branches share one tile shape because both emit a bf16 row,
  which also folds in the fp32→bf16 cast that the NVLS dispatcher pays as a separate op.
- **Thresholds** (`--dyn-dispatch-threshold`, `--dyn-combine-threshold`) are separate
  because dispatch and combine cross over at genuinely different step sizes. Set one to 0
  to pin its A2Av branch, or to a huge value to pin its AGv branch — which is how
  `--validate` exercises both paths and checks that they agree, and how
  `bench_phase_breakdown.py` measures each branch in isolation.
- **`--out-dtype {bf16,fp32}`** picks the expert-output (combine-source) buffer dtype and
  applies to `nvls` and `dynamic` alike, so the comparison stays apples-to-apples.
  `bf16` (default) is this bench's own convention; **`fp32` is what Megatron's fused-MoE
  unpermute actually produces** (it accumulates fp32 atomically), and is what the in-tree
  threshold defaults were measured with. The distinction is large: under fp32 the
  in-switch reduce moves 4 bytes per element per rank while the push wire stays bf16, and
  the NVLS path additionally pays a separate fp32→bf16 cast — so all-to-all-v takes over
  far earlier. With a bf16 buffer the combine crossover moves out past ~6k tokens, so the
  thresholds must be retuned for that case.

### Thresholds are strongly EP-dependent

Measured per phase with `bench_phase_breakdown.py` (GB200, 512 experts, top-k 22, MoE
hidden 1024, fp32 expert-output buffer), taking the first step size at which the A2Av
branch strictly beats the AGv branch:

| EP | nodes | dispatch crossover | combine crossover |
|---|---|---|---|
| 4  | 1  |  5120 (1280/rank) |  1792 ( 448/rank) |
| 16 | 4  |  8192 ( 512/rank) |  8192 ( 512/rank) |
| 32 | 8  |  8192 ( 256/rank) | 32768 (1024/rank) |
| 64 | 16 | 16384 ( 256/rank) | 65536 (1024/rank) |

The two scale completely differently, and **neither is a constant**:

- **Dispatch** is roughly flat in global tokens across a 16x EP range, because its traffic
  advantage *grows* with EP and cancels the higher per-peer loop cost. A token reaches
  `avg_dest = ep·(1-(1-1/ep)^topk)` ranks, so unicast moves `avg_dest/ep` of the multicast
  bytes: **99.8% at EP=4** (top-k 22 already touches all four ranks, so there is no traffic
  saving at all) but only **29% at EP=64**.
- **Combine** scales ~linearly with EP, the wrong way. The push kernel walks WORLD_SIZE
  source ranks and the receive kernel sums WORLD_SIZE candidate planes per token, so its
  per-token cost is O(EP) while the in-switch reduce stays O(1) in instructions. At a
  256-token step the push combine is 2.4x worse than RSv at EP=4 but **21x worse at EP=64**.

Using the EP=4 numbers at EP=64 costs up to **6x** (`dynamic` measured at 0.17x of NVLS in
the 2k–16k band), so the defaults are resolved from world size — see
`A2AV_THRESHOLDS_BY_EP` / `default_a2av_thresholds()`. Passing `None` (bench) or `-1`
(Megatron config) selects the measured value for the running EP.

### Measured with the EP-aware defaults (fp32 expert-output buffer, per-layer µs)

`dynamic` vs NVLS, i.e. the speedup of the adaptive kernel over the fixed in-switch path:

| step tokens | EP=4 | EP=16 | EP=32 | EP=64 |
|---|---|---|---|---|
| 64    | 1.01x | 1.05x | (pending) | (pending) |
| 256   | 1.03x | 1.04x | (pending) | (pending) |
| 1024  | 0.99x | 1.02x | (pending) | (pending) |
| 4096  | 1.18x | 1.00x | (pending) | (pending) |
| 8192  | 1.50x | 1.07x | (pending) | (pending) |
| 32768 | 1.67x | 1.45x | (pending) | (pending) |
| 65536 | 1.68x | 1.47x | (pending) | (pending) |

i.e. `dynamic` tracks the lower envelope of the two fixed strategies at every EP: parity
where the in-switch collectives win, and a growing margin where all-to-all-v wins.

One structural cost is unavoidable: the combine needs two launches in both regimes
(stage 1 is an immediate return in the NVLS regime), because the cross-rank barrier
`symm_mem_sync` pairs CTAs by `blockIdx` rather than being grid-wide, so push and reduce
cannot share a kernel. That costs ~1.3 µs/layer, and it is offset by the fused dispatch
kernel being 5–25% faster than the stock AGv-3tensor kernel.

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
  `multimem_all_gatherv_3tensor` (AGV-V) → `multimem_reduce_scatter_v` (RSV-V; **bf16**
  buffer, fp32-accumulated).
- **NCCL** per layer: 3× `all_gather_into_tensor` (hidden bf16, routing int64, probs fp32)
  → `reduce_scatter_tensor` (fp32). No once-per-step metadata collective (equal per-rank
  counts are guaranteed by padding, discovered with one `all_reduce(MAX)` in setup,
  outside the graph).

## Experimental setup (Nemotron-Super, 1 node, 4× B200)

| | value |
|---|---|
| experts / top-k / hidden | 512 / 22 / 1024 |
| parallelism | EP=4 (1 rank/GPU), TP=1, single NVLink domain |
| dtype | dispatch **bf16** (all three); combine **bf16** for NVLS & DeepEP (NVLS accumulates the reduction in fp32 via `acc::f32`), **fp32** for NCCL |
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
output on identical inputs (all compute `m·x`; NCCL is exact in fp32, NVLS & DeepEP combine
in bf16 and agree within bf16 rounding). Prints `[PASS]/[FAIL]` per check and a verdict reduced
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

`run.py` flags: `--impl` — a single name, a comma-separated list, or a shortcut. Names:
`deepep | nvls | nccl | a2av | a2av_rs | a2av_push | dynamic`. Shortcuts: `both`
(deepep+nvls), `all` (everything), **`compare` (nvls+a2av_push+dynamic — the three-way
regime comparison)**. Plus `--batch-sizes`, `--num-layers`, `--deepep-num-sms` (comma
list, even, clamped to device SM count), `--timing {graph,eager}`, `--reps`, `--warmup`,
`--out`, `--validate` (correctness checks then exit — see above), and the `dynamic`-only
`--dyn-dispatch-threshold`, `--dyn-combine-threshold`, `--dyn-out-dtype {bf16,fp32}`.

The three-way comparison, plus its plot:
```bash
torchrun --nproc_per_node=4 run.py --impl compare --dyn-out-dtype fp32 \
    --batch-sizes 1,16,256,1024,1280,2048,8192,32768 \
    --num-layers 88 --reps 30 --warmup 8 --out results_dynamic_fp32.csv
python3 plot_dynamic_compare.py --csv results_dynamic_fp32.csv \
    --out results_dynamic_fp32.png --dispatch-threshold 1280 --combine-threshold 1280
```

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
- `bench_deepep.py` / `bench_nvls.py` / `bench_nccl.py` / `bench_a2av.py` / `bench_dynamic.py` — the benchers; each exposes `decode_step(num_layers)` + `validate()` + `functional_roundtrip()`.
- `plot_dynamic_compare.py` — the AGv/RSv vs A2Av vs dynamic figure (absolute latency + speedup over NVLS, with the switch thresholds marked).
- `bench_phase_breakdown.py` / `plot_phase_breakdown.py` — times DISPATCH and COMBINE separately for `nvls` and for each pinned branch of `dynamic`. This is how the two crossovers were located, and how a coalescing defect in the reduce-scatter branch was found (a lane mapping that reduce-loaded chunks `(2c, 2c+1)` left every 32-byte sector half-used per instruction; unlike a local load, multimem results are not cached, so the second load refetched from every peer and doubled NVLink traffic — 1.8x combine slowdown at large steps).
- `nvls/` — vendored, isolated NVLS collectives (zero Megatron deps; only change vs upstream is the Triton-3.6 widen, the 128→148 block cap, and import isolation).
- `plot_results.py` — plot `--x B` (default) or `--x num_sms`, in milliseconds.
- `deepep_env.sh` — DeepEP build+runtime env (idempotent; relocatable).
- `install_deepep_ngc.sh` — one-shot DeepEP wheel install + build (called by `deepep_env.sh`).
- `run.sbatch` — SLURM batch template.

## Caveats
- **Combine precision.** NVLS and DeepEP combine in **bf16**; NCCL combines in **fp32**.
  NVLS's ReduceScatter-V moves bf16 operands but **accumulates the cross-rank sum in fp32**
  (`multimem.ld_reduce.add.acc::f32.v4.bf16x2`) — high precision at half the fp32 combine
  bytes. So NVLS↔DeepEP now match on combine precision (bf16), while NVLS-vs-NCCL differs in
  **both** transport and combine dtype; read that pair as a transport+precision contrast,
  not pure transport. (To restore a pure-transport NVLS-vs-NCCL comparison, switch NCCL's
  `reduce_scatter_tensor` to bf16 in `bench_nccl.py`.)
- **NCCL padding.** The graphable NCCL path requires equal per-rank token counts, so ranks
  are padded to the per-step max (`ceil(B/ep)`). Under the balanced global-B split counts
  differ by ≤1, so padding is negligible; the padded rows are gathered/reduced then
  truncated (they never pollute real-token outputs).
