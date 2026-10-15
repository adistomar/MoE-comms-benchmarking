# MoE decode dispatch/combine benchmark — DeepEP-v2, Megatron NVLS & NCCL, and A2A-V

Isolated micro-benchmark comparing four MoE expert-parallel **dispatch/combine**
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
- **A2A-V** (`nvls/torch_symm_triton/all_to_all_v.py`) — a **sparse all-to-all** built
  from the same symmetric-memory pieces as NVLS (row layout, per-step metadata, per-CTA
  barrier): dispatch **unicasts** each token's hidden row only to the ranks that own one of
  its experts (direct peer stores), and combine **pulls** the per-rank partial outputs back
  and sums them in fp32. It is a drop-in alternative to the AGV-V / RSV-V pair (see
  [A2A-V design](#a2a-v-design)).

Both Megatron dispatchers are simulated standalone: the NVLS collectives are **vendored**
(`nvls/`, an isolated copy of Megatron's `torch_symm_triton` + `symmetric_memory.py` +
`metadata.py`; zero Megatron deps), and NCCL uses stock `torch.distributed`
AllGather/ReduceScatter. A2A-V lives next to the vendored NVLS kernels and reuses their
barrier and metadata. DeepEP is called at the `ElasticBuffer` level and requires a
DeepEP source checkout to build (see Setup).

## What is measured

**One full decode step** = `--num-layers` MoE layers (default: the model's MoE block count,
**40** for Nemotron-3-Super and **48** for Nemotron-3-Ultra; see [Models](#models)), each
doing a **paired dispatch → (identity expert) → combine**, captured and
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
- **A2A-V** per step: `fused_metadata_update` **once** (same as NVLS) then, per layer,
  `a2a_dispatch_v` → `a2a_combine_v` (**bf16** partial buffer, fp32-accumulated).

## Experimental setup (GB300 NVL72, 4 GPUs/node)

| | value |
|---|---|
| model | `--model ultra` (default) or `--model super`; sets the rows below (see [Models](#models)) |
| experts / top-k / dispatched row width | 512 / 22 / 1024 (Super) or 2048 (Ultra) — the `moe_latent_size`, since both are latent MoEs |
| parallelism | EP=4–64 (1 rank/GPU, 1–16 nodes of one NVL72 rack), TP=1, single NVLink domain |
| dtype | dispatch **bf16** (all four); combine **bf16** for NVLS, DeepEP & A2A-V (all accumulate the sum in fp32: NVLS via `acc::f32`, A2A-V in registers), **fp32** for NCCL |
| batch axis | **GLOBAL** B ∈ {1,…,1M} tokens across all EP ranks (balanced; B<EP leaves some ranks with 0 tokens; NCCL pads to the per-step max) |
| layers | 40 (Super) / 48 (Ultra) MoE layers per decode step; `--num-layers` overrides |
| routing | uniform: 22 distinct experts/token; identical tensors fed to all impls |
| no host sync | DeepEP `do_cpu_sync=False`; NVLS / A2A-V on-device metadata; NCCL pad-count all-reduce done in setup (outside the graph) |
| comm SMs / blocks | DeepEP `num_sms` = device SM count (152 on GB300) unless swept; **NVLS and A2A-V**: CTA cap = 4× SM count at EP ≤ 8, SM count at EP ≥ 16 (see below); NCCL uses NCCL-internal grid |

## Models

`--model` selects a preset in `models.py`, copied from each model's Hugging Face
`config.json` ([Super](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16/blob/main/config.json),
[Ultra](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16/blob/main/config.json)):

| | Super (120B-A12B) | Ultra (550B-A55B) |
|---|---|---|
| blocks (Mamba / MoE / attention) | 88 (40 / **40** / 8) | 108 (48 / **48** / 12) |
| `hidden_size` | 4096 | 8192 |
| `moe_latent_size` (dispatched / combined row width) | **1024** | **2048** |
| `n_routed_experts` / `num_experts_per_tok` | 512 / 22 | 512 / 22 |

Only MoE blocks dispatch and combine, so a decode step has one dispatch → combine per MoE
block, not per block of the hybrid stack. Both models project tokens from `hidden_size` to
`moe_latent_size` before routing to experts, so the collectives move `moe_latent_size`-wide
rows. Super's config gives the layout as `hybrid_override_pattern` (`M` = Mamba, `E` = MoE,
`*` = attention); Ultra's gives the same as `layers_block_type`, which `models.py` stores in
the same letter format. The MoE count is computed from the pattern (and asserted against the
config's block count). Each run records `model` and `num_layers` in its CSV, and
`plot_results.py` uses them for the plot title.

DeepEP flags: `use_fp8_dispatch=False` (bf16), `allow_hybrid_mode=False` (single
NVLink domain → flat/direct path, Gin dormant not disabled), `allow_multiple_reduction=True`
(ep=4 ≤ topk=22 ⇒ rank-layout combine), `do_expand=False` (pure collective — permute is
deferred to the GEMM, matching NVLS), `num_sms` swept.

**NVLS / A2A-V block cap.** The NVLS AGV/RSV kernels and the A2A-V kernels run **one CTA
per token**, so the CTA-grid ceiling `max_num_blocks` bounds how many tokens are in flight
(the grid is `min(per_rank_cap, max_num_blocks)`; CTAs past the step's `ep_max_tokens` exit
at once). Upstream Megatron hardcodes 128. Here the wrappers in
`nvls/torch_symm_triton/variable_collectives.py` and `all_to_all_v.py` default to the
**device SM count** (`multi_processor_count`: 152 on GB300, 148 on B200), and the benchers
override it per EP via `default_num_blocks` in `bench_nvls.py`:

| EP | CTA cap (GB300) |
|---|---|
| ≤ 8 | 4 × SMs = 608 |
| ≥ 16 | SMs = 152 |

*Why more CTAs at small EP.* Each CTA has one token's rows in flight, so the bytes in flight
per GPU are roughly `num_blocks × (distinct ranks a token reaches) × row size` (2 KB for
Super's 1024-wide bf16 rows, 4 KB for Ultra's 2048). With top-22 of 512 experts a token reaches ≈4 ranks at EP=4, ≈7.6 at
EP=8, ≈12 at EP=16 and ≈16–19 at EP=32/64. At EP ≤ 8, one CTA per SM therefore has too
little data in flight to cover NVLink latency, and large batches are latency-bound. A CTA
sweep on GB300 (per-layer dispatch+combine, B=128K) showed:

| | 148 CTAs (~1/SM) | 592–1184 CTAs (4–8/SM) |
|---|---|---|
| A2A-V, EP=4 | 1.62 ms | 0.73 ms |
| A2A-V, EP=8 | 0.94 ms | 0.78 ms |
| A2A-V, EP=16 | 0.70 ms | 0.68 ms |
| NVLS AGV/RSV, EP=4 | 1.12 ms | 0.85 ms |

At EP ≤ 8, 4 CTAs/SM was never slower than 1/SM at any swept batch size. At EP=16, one CTA
per SM already keeps the link busy, and extra CTAs only add barrier traffic
(`num_blocks × EP` signals): 4 CTAs/SM was 5–10% *slower* at B=4K–16K. At small B the cap
does not matter (CTAs beyond the step's token count exit immediately). DeepEP has no equivalent knob: it launches one
persistent CTA per SM (`num_sms`) and loops over tokens inside the kernel, with one warp per
token and TMA copies, so its concurrency comes from warps rather than from the grid. Its
`num_sms` is independent of this cap (sweeping it does not affect NVLS or A2A-V).

**Signal pad.** The AGV/RSV and A2A-V barrier gives every CTA one 4-byte signal slot per
peer (`block_id * world_size + rank`), so each symmetric buffer's signal pad must hold
`num_blocks * EP * 4` bytes. PyTorch's default pad (9216 B = 2304 slots, torch ≥ 2.9)
overflows *silently* into the data buffer once `num_blocks * EP > 2304` (152 blocks at
EP ≥ 16, 608 blocks already at EP=4). `bench_nvls.py` and `bench_a2a.py` therefore call
`torch.distributed._symmetric_memory.set_signal_pad_size` (torch ≥ 2.10) with
`min(per_rank_cap, num_blocks) * EP * 4` bytes before allocating their buffers; `run.py`
prints the resulting grid and pad size.

## A2A-V design

`nvls/torch_symm_triton/all_to_all_v.py` provides `a2a_dispatch_v` and `a2a_combine_v`
(Triton kernels + Python wrappers); `bench_a2a.py` drives them.

**Layout and metadata (same as AGV-V).** Source rank `s` writes its local token `t` to row
`rank_token_offset[s] + t` of `[global_max, *]` symmetric buffers, using the per-step
`[valid_tokens, rank_token_offset, ep_max_tokens]` computed on device by
`fused_metadata_update`. One CTA processes one token at a time (grid-stride over local
tokens); CTAs with `pid >= ep_max_tokens` exit before the barrier on every rank, since
`ep_max_tokens` is the same everywhere.

**Dispatch (`a2a_dispatch_v`).** Each CTA loads its token's hidden row once (one 16-byte
chunk per thread) and walks the token's top-k entries: destination rank =
`expert_id // num_local_experts`, deduplicated with a 64-bit "already sent" mask (`-1`
entries are skipped). The row is stored to each distinct destination's buffer through the
symmetric-memory peer pointers (`buffer_ptrs_dev`), so a token goes only to the ranks that
own one of its experts. Routing ids and probs are still **multicast** (`multimem.st`) to
every rank, so every rank has the complete routing map for rows `[0, valid_tokens)` and can
tell rows routed to it from rows it did not receive ("holes" — the expert side skips rows
with no local expert anyway). A trailing per-CTA-pair barrier (release/acquire) publishes
the stores; one barrier covers all three buffers.

**Combine (`a2a_combine_v`).** A leading per-CTA-pair barrier (release/acquire) makes every
destination's partial outputs visible. For each local token the CTA recomputes the
destination ranks from its own routing and **pulls** row `rank_token_offset + t` from each
destination's bf16 partial buffer with predicated 128-bit loads: all of a token's loads are
issued together, and skipped entries (duplicates, `-1`) return zeros so the adds stay
branch-free. The rows are summed in fp32 in a fixed order (deterministic) and rounded once
to bf16. There is no trailing barrier: the next dispatch's barrier protects the partial
buffers.

**Contract with the expert side.** On rank `d`, partial row `r` must hold `d`'s
probability-weighted sum over its local experts for token `r`. Rows of tokens with no
expert on `d` are never read, so — unlike RSV-V's `multimem.ld_reduce`, which sums every
rank's row — they need no zero-fill.

**Buffer reuse.** A source cannot re-dispatch into a destination before that destination
has reached combine (so its experts are done with the receive buffers), and a destination
cannot overwrite its partials before every source has finished pulling (the next dispatch's
barrier waits on each source's dispatch, which starts after that source's combine).

**Limits.** EP ≤ 64 (the dedup mask is one int64; asserted in the wrappers); hidden rows must
be 16-byte aligned; routing ids must be int64; combine supports bf16 partials only.

## Correctness check (`--validate`)

`torchrun --nproc_per_node=4 run.py --impl nvls,deepep,nccl,a2a --validate` runs known-value
checks per impl (random tensors, verified element-wise) then exits without timing. It
confirms NVLS AGV-V gathers each rank's tokens to the right global offset, NVLS RSV-V sums
across ranks and scatters to the right owner, NCCL's padded AllGather→ReduceScatter
round-trips correctly, and DeepEP's dispatch→combine round-trips to `m·x` (`m` = #destination
ranks). For A2A-V, tokens are routed to random subsets of ranks (plus an all-`-1` padding
token) and it checks that routing and probs arrive bit-exact on every rank, routed hidden
rows arrive bit-exact, rows not routed to a rank are left untouched (NaN sentinels), and
combine returns `bf16(m·partial)` bit-exact. With ≥2 impls built it also **cross-checks**
that all of them produce the same combine output on identical inputs (all compute `m·x`;
NCCL is exact in fp32, NVLS, DeepEP & A2A-V combine in bf16 and agree within bf16 rounding).
Prints `[PASS]/[FAIL]` per check and a verdict reduced across ranks (exit 0/1); tested at a
full batch (`B=2·ep`) and the 0-token-rank case (`B=1`).

## Setup

**Requirements.** GPUs in a **single NVLink domain** (one node, or several nodes of one
NVL72 rack; validated on 4× B200 / GB200 and GB300 NVL72) and an NGC-style PyTorch container
(validated: CUDA 13, torch 2.11/2.12, **Triton 3.6/3.7**, `torch.distributed._symmetric_memory`
+ multicast, NVRTC/ptxas). Multi-GPU launched with `torchrun` or one task per GPU
(`launch_rank.sh`). NVLS and A2A-V need Hopper+ (SM ≥ 9) with NVLink + symmetric memory, and
torch ≥ 2.10 (for `set_signal_pad_size`).

**1. Clone this repo and DeepEP side-by-side:**
```bash
git clone <THIS_REPO_URL> moe-comms-bench
git clone https://github.com/deepseek-ai/DeepEP.git DeepEP   # checkout the "elastic"
cd DeepEP && git checkout af9a0403 && cd ..                   # v2 ElasticBuffer commit
# layout:  ./moe-comms-bench   (this repo)   and   ./DeepEP   (sibling)
```
The DeepEP checkout must contain `deep_ep/buffers/elastic.py` (the v2 "elastic"
dispatcher). `deepep_env.sh` looks for DeepEP at `../DeepEP` by default; override with
`export DEEPEP_DIR=/path/to/DeepEP`. Newer DeepEP (V2.5, `EPBuffer`) is not used: it builds
through DeepJIT, which needs elfutils headers (`libdw-dev`) the container does not ship, and
its EP dispatch/combine path is unchanged for this benchmark.

**2. Get an interactive allocation in the container** (adjust account/partition/image):
```bash
srun -p batch --account=<ACCT> --qos=interactive -t 2:00:00 --nodes=1 --exclusive \
  --gpus-per-node=4 --container-image <CONTAINER.sqsh> \
  --container-mounts "/home:/home,/lustre:/lustre" --pty /bin/bash
```

**3. Set up the environment** (builds DeepEP on first run; NVLS/NCCL/A2A-V need nothing):
```bash
cd moe-comms-bench
source ./deepep_env.sh      # installs nvidia-nccl-cu13==2.30.4 + nvshmem-cu13 wheels,
                            # orders the new NCCL first, builds DeepEP (~15 min first
                            # time; cached after), sets a persistent JIT cache.
```
NCCL is pinned to 2.30.4 (the version `af9a0403` targets, and the one the mcore-inference
container's torch ships); `>=2.30.4` would now resolve to 2.32.x. The first build compiles
DeepEP's `_C` extension into `$DEEPEP_DIR/deep_ep/` and caches JIT kernels under
`$EP_JIT_CACHE_DIR` (default `~/.cache/deepep_jit`, off Lustre) — both persist, so re-running
in a fresh (ephemeral) container just reinstalls the wheels and reuses the prebuilt
extension. Run the first build from a **single task** (multi-task jobs refuse to build).
If DeepEP's runtime-vs-linked NCCL check complains, `export EP_SUPPRESS_NCCL_CHECK=1`.

**4. Run:**
```bash
# correctness first (no timing) — see the Correctness section
torchrun --nproc_per_node=4 run.py --model super --impl nvls,deepep,a2a --validate

# quick smoke (--num-layers defaults to the model's MoE block count: 40 Super, 48 Ultra)
torchrun --nproc_per_node=4 run.py --model super --impl nvls,deepep,a2a --batch-sizes 4 \
    --reps 3 --warmup 2

# full batch-size sweep -> results/results.csv (DeepEP num_sms defaults to the device SM count)
torchrun --nproc_per_node=4 run.py --model ultra --impl nvls,deepep,a2a \
    --batch-sizes 1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192 \
    --reps 20 --warmup 6 --timing graph --out results.csv
python3 plot_results.py --csv results.csv --out results.png          # -> plots/results.png

# DeepEP num_sms sweep -> results/results_sms.csv (each num_sms JIT-compiles once;
# NVLS/NCCL are num_sms-independent reference lines)
torchrun --nproc_per_node=4 run.py --model super --impl all \
    --batch-sizes 1,16,128 --deepep-num-sms 4,16,64,128 \
    --reps 10 --warmup 4 --timing graph --out results_sms.csv
python3 plot_results.py --csv results_sms.csv --x num_sms --out results_sms.png
```
Or submit `run.sbatch` (edit the SBATCH headers / `CONTAINER_IMAGE` for your cluster,
then `sbatch run.sbatch` from the repo directory; multi-node: `sbatch -N 4 --segment=4
run.sbatch`; pick the impls with `IMPL=nvls,deepep,a2a sbatch run.sbatch` and the model with
`MODEL=super sbatch run.sbatch`, default `ultra`; outputs are named
`results_<model>[_<N>n].csv`). For
one-task-per-GPU launches without torchrun, `launch_rank.sh` sets
`RANK`/`WORLD_SIZE`/`MASTER_ADDR` from Slurm and forwards its arguments to `run.py`.

**Outputs.** CSVs are written to `results/` and plots to `plots/` (bare `--out`/`--csv`
names are resolved there; override with `--results-dir` / `--plots-dir`, or pass a path).

`run.py` flags: `--model {super,ultra}` (default `ultra`), `--impl` (comma-separated list of
`deepep`, `nvls`, `nccl`, `a2a`, or the aliases `both` = deepep,nvls and `all` =
deepep,nvls,nccl; e.g. `--impl deepep,a2a`), `--batch-sizes`, `--num-layers` (default: the
model's MoE block count), `--deepep-num-sms` (comma list, even, clamped to device
SM count), `--timing {graph,eager}`, `--reps`, `--warmup`, `--out`, `--results-dir`,
`--validate` (correctness checks then exit — see above).

## Required patch (Triton 3.6): int64 pointer-widen
The vendored NVLS multimem kernels (`nvls/torch_symm_triton/variable_collectives.py`,
`nvls/metadata.py`) and the A2A-V kernels (`nvls/torch_symm_triton/all_to_all_v.py`) take
raw pointer *ints* and do `x.to(tl.pointer_type(...))`. Triton 3.6 specializes a scalar int
arg as **i32** when its value fits in 32 bits (a low GPU VA), but `tt.int_to_ptr` needs i64
→ compile error. Fix (tagged `# Required Triton-3.6 fix` / `Triton-3.6 fix`): widen each raw
pointer int to i64 at kernel entry — value-preserving, and a no-op on Triton versions that
already type it i64. Without it the NVLS and A2A-V paths will not compile.

## Files
- `run.py` — torchrun driver (times one full decode step as a CUDA graph).
- `models.py` — Nemotron-3 Super / Ultra presets (`--model`): layer pattern, MoE layer count, experts, top-k, `hidden_size`, `moe_latent_size` (torch-free; also used by the plotter).
- `common.py` — config (shape from the model preset), global→per-rank token split, routing/input gen, `time_region` (CUDA-event timing).
- `bench_deepep.py` / `bench_nvls.py` / `bench_nccl.py` / `bench_a2a.py` — the four benchers; each exposes `decode_step(num_layers)` + `validate()` + `functional_roundtrip()`.
- `nvls/` — vendored, isolated NVLS collectives (zero Megatron deps; only change vs upstream is the Triton-3.6 widen, the block cap defaulting to the device SM count instead of 128, and import isolation), plus `nvls/torch_symm_triton/all_to_all_v.py` (the A2A-V kernels).
- `plot_results.py` — plot `--x B` (default) or `--x num_sms`, in milliseconds; the title shows the CSV's model and MoE layers per step.
- `deepep_env.sh` — DeepEP build+runtime env (idempotent; relocatable).
- `install_deepep_ngc.sh` — one-shot DeepEP wheel install + build (called by `deepep_env.sh`).
- `launch_rank.sh` — per-task launcher (one Slurm task per GPU = one EP rank).
- `run.sbatch` — SLURM batch template.
- `results/`, `plots/` — benchmark CSVs and plots (created on demand).

## Caveats
- **Combine precision.** NVLS, DeepEP and A2A-V combine in **bf16**; NCCL combines in
  **fp32**. NVLS's ReduceScatter-V moves bf16 operands but **accumulates the cross-rank sum in
  fp32** (`multimem.ld_reduce.add.acc::f32.v4.bf16x2`), and A2A-V sums the pulled bf16 rows in
  fp32 registers — high precision at half the fp32 combine bytes. So NVLS, DeepEP and A2A-V
  match on combine precision (bf16), while NVLS-vs-NCCL differs in **both** transport and
  combine dtype; read that pair as a transport+precision contrast, not pure transport. (To
  restore a pure-transport NVLS-vs-NCCL comparison, switch NCCL's `reduce_scatter_tensor` to
  bf16 in `bench_nccl.py`.)
- **NCCL padding.** The graphable NCCL path requires equal per-rank token counts, so ranks
  are padded to the per-step max (`ceil(B/ep)`). Under the balanced global-B split counts
  differ by ≤1, so padding is negligible; the padded rows are gathered/reduced then
  truncated (they never pollute real-token outputs).
- **CTA cap is a static per-EP choice.** The cap (4× SMs at EP ≤ 8, SMs at EP ≥ 16) comes
  from the B=256–128K sweep above at EP=4/8/16; EP=32/64 follow the EP=16 rule untested.
  Under CUDA graphs the grid is frozen at capture, so a per-batch-size choice would need an
  on-device active-CTA count rather than a different launch grid.
- **A2A-V routing/probs multicast.** Every rank receives every token's 264-byte routing +
  probs row (`B × 264` bytes per layer), independent of EP. Hidden-row traffic shrinks as EP
  grows, so this becomes a larger share of dispatch traffic at large EP (~14% of incoming
  dispatch bytes at EP=16, ~30% at EP=64 for B=128K).
