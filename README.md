# MoE decode dispatch/combine benchmark — DeepEP, Megatron NVLS & NCCL, A2A-V, and Hybrid

Isolated micro-benchmark comparing five MoE expert-parallel **dispatch/combine**
schemes used for **inference decode** within a single NVLink domain, optionally with
Megatron's real expert GEMMs between dispatch and combine (`--expert vllm`):

- **DeepEP** (`EPBuffer`, DeepEP ≥ 2.5) — a true expert-parallel **all-to-all** over
  NVLink (direct-peer load/store, no `multimem`), set up like Megatron's `DeepEPDispatcher`:
  NVLink only (`EP_DISABLE_GIN=1`, `allow_hybrid_mode=False`).
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
- **Hybrid** (`nvls/torch_symm_triton/hybrid_v.py`) — one dispatch and one combine kernel that
  pick, **on device, once per decode step**, between the NVLS transport (multicast dispatch /
  `multimem.ld_reduce` combine) and the A2A-V transport (unicast dispatch / pull combine) from
  the global token count. The design and its rationale are in
  [`docs/hybrid_dispatcher_design.md`](docs/hybrid_dispatcher_design.md); see
  [Hybrid](#hybrid).

Both Megatron dispatchers are simulated standalone: the NVLS collectives are **vendored**
(`nvls/`, an isolated copy of Megatron's `torch_symm_triton` + `symmetric_memory.py` +
`metadata.py`; zero Megatron deps), and NCCL uses stock `torch.distributed`
AllGather/ReduceScatter. A2A-V lives next to the vendored NVLS kernels and reuses their
barrier and metadata. DeepEP is called at the `EPBuffer` level, with the arguments Megatron's
`DeepEPDispatcher` passes; it comes prebuilt in the container image (see Setup).

## What is measured

**One full decode step** = `--num-layers` MoE layers (default: the model's MoE block count,
**40** for Nemotron-3-Super and **48** for Nemotron-3-Ultra; see [Models](#models)), each
doing a **paired dispatch → expert → combine**, captured and replayed as a **single CUDA
graph**. The expert is the identity by default (communication only), or Megatron's
`vllm_fused_moe` with `--expert vllm` (see [Expert compute](#expert-compute---expert-vllm)).
Reported latency is **milliseconds per decode step** (all layers), taken as the **max across
ranks** (critical path). Per-layer = step / num_layers.

- **DeepEP** per layer: `EPBuffer.dispatch` (incl. the routing-dependent count exchange and
  copy epilogue) then `EPBuffer.combine` (incl. the reduce epilogue).
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
- **Hybrid** per step: `fused_metadata_update` with a policy **once** (also writes the two mode
  flags) then, per layer, `hybrid_dispatch_v` → `hybrid_combine_v`.

`--phases dispatch,combine` also times each collective alone: one metadata call, then
`--num-layers` dispatches (or combines), as one CUDA graph (NVLS, NCCL, A2A-V, Hybrid; DeepEP's
combine needs its dispatch's handle, so it has only `decode_step`). The CSV's `phase` column
tells the rows apart.

## Experimental setup (GB300 NVL72, 4 GPUs/node)

| | value |
|---|---|
| model | `--model ultra` (default) or `--model super`; sets the rows below (see [Models](#models)) |
| experts / top-k / dispatched row width | 512 / 22 / 1024 (Super) or 2048 (Ultra) — the `moe_latent_size`, since both are latent MoEs |
| parallelism | EP=4–64 (1 rank/GPU, 1–16 nodes of one NVL72 rack), TP=1, single NVLink domain |
| dtype | dispatch **bf16** (all four); combine **bf16** for NVLS, DeepEP & A2A-V (all accumulate the sum in fp32: NVLS via `acc::f32`, A2A-V in registers), **fp32** for NCCL |
| batch axis | **GLOBAL** B ∈ {1,…,1M} tokens across all EP ranks (balanced; B<EP leaves some ranks with 0 tokens; NCCL pads to the per-step max). `--imbalance ρ` gives rank 0 ρ× the mean instead |
| layers | 40 (Super) / 48 (Ultra) MoE layers per decode step; `--num-layers` overrides |
| routing | uniform: 22 distinct experts/token; identical tensors fed to all impls |
| expert | identity (default, communication only) or `--expert vllm`: Megatron's `vllm_fused_moe` (BF16, squared ReLU, random weights) |
| no host sync | DeepEP `do_cpu_sync=False`; NVLS / A2A-V on-device metadata; NCCL pad-count all-reduce done in setup (outside the graph) |
| comm SMs / blocks | DeepEP `num_sms` = DeepEP's `get_theoretical_num_sms` (`auto`, Megatron's default; 36 at EP=4) unless swept; **NVLS and A2A-V**: CTA cap = 4× SM count at EP ≤ 8, SM count at EP ≥ 16 (see below); NCCL uses NCCL-internal grid |

## Models

`--model` selects a preset in `models.py`, copied from each model's Hugging Face
`config.json` ([Super](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16/blob/main/config.json),
[Ultra](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16/blob/main/config.json)):

| | Super (120B-A12B) | Ultra (550B-A55B) |
|---|---|---|
| blocks (Mamba / MoE / attention) | 88 (40 / **40** / 8) | 108 (48 / **48** / 12) |
| `hidden_size` | 4096 | 8192 |
| `moe_latent_size` (dispatched / combined row width) | **1024** | **2048** |
| `moe_intermediate_size` (routed-expert FFN width, `--expert vllm`) | 2688 | 5120 |
| `n_routed_experts` / `num_experts_per_tok` | 512 / 22 | 512 / 22 |

Only MoE blocks dispatch and combine, so a decode step has one dispatch → combine per MoE
block, not per block of the hybrid stack. Both models project tokens from `hidden_size` to
`moe_latent_size` before routing to experts, so the collectives move `moe_latent_size`-wide
rows. Super's config gives the layout as `hybrid_override_pattern` (`M` = Mamba, `E` = MoE,
`*` = attention); Ultra's gives the same as `layers_block_type`, which `models.py` stores in
the same letter format. The MoE count is computed from the pattern (and asserted against the
config's block count). Each run records `model` and `num_layers` in its CSV, and
`plot_results.py` uses them for the plot title.

DeepEP flags, as in Megatron's `DeepEPDispatcher`: `EPBuffer(num_max_tokens_per_rank,
hidden, num_topk, allow_hybrid_mode=False)` with `EP_DISABLE_GIN=1` (NVLink only; the bencher
sets it unless already set) and `EPBuffer`'s defaults otherwise — bf16 dispatch,
`allow_multiple_reduction=True` (rank-layout combine: one reduced copy per destination rank),
`prefer_overlap_with_compute=True`, `deterministic=False`; `dispatch(do_expand=False,
do_cpu_sync=False)` (one copy per destination rank; the expert GEMM groups rows itself);
`combine(x, handle)` (unweighted — the expert already applied the routing probs). `num_sms`
defaults to `auto` (DeepEP's bandwidth model, MAX across ranks) and can be swept.

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

**Combine (`a2a_combine_v`).** A leading per-CTA-pair barrier (relaxed, as RSV-V's) waits
until every destination has finished its expert step: the partials are written by earlier
kernels. For each local token the CTA recomputes the
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

## Hybrid

`nvls/torch_symm_triton/hybrid_v.py` provides `hybrid_dispatch_v` and `hybrid_combine_v`;
`bench_hybrid.py` drives them. Full design: [`docs/hybrid_dispatcher_design.md`](docs/hybrid_dispatcher_design.md).

**Kernels.** The A2A-V kernels with one branch each, on a mode flag read on device:
- dispatch: multicast the hidden row (`multimem.st`, as AGV-V) or unicast it to the token's
  destination ranks (as A2A-V). Routing and probs are multicast either way.
- combine: `multimem.ld_reduce` over every rank (as RSV-V) or pull from the destination ranks
  (as A2A-V).

Layout, metadata, grid, early exit and barriers are the same in both modes, so the flag may
change on any step, inside one captured graph.

**Policy.** `fused_metadata_update(..., policy=)` evaluates, once per step and identically on
every rank, `valid_tokens - kappa * ep_max_tokens <= threshold` for dispatch and for combine.
It writes the results to `step_metadata[3]` (multicast dispatch) and `step_metadata[4]`
(multicast-reduce combine). The policy is an int32 device tensor, so it can change between
graph replays.
- Defaults (`bench_hybrid.DEFAULT_THRESHOLDS`, `kappa = 0`) are separate dispatch and
  combine crossovers. EP=4 and EP=64 are measured per phase on GB200; EP=16/32 are read off
  the `plots/comm` crossovers for now.
- `run.py` flags: `--hybrid-threshold D[,C]` and `--hybrid-kappa K[,K]`.
- `--hybrid-modes` times several modes from one bencher:
  - `auto`: the global threshold policy (κ=0 by default);
  - `kappa`: global with κ = u, which also multicasts every rank under heavy imbalance;
  - `rank`: per rank, where the heaviest senders multicast while their unicast fan-out
    `u·n_s` exceeds `q·N + (1−q)·S_s` (q = P(a rank is a token's destination), u = (EP−1)·q,
    S_s = n_s + the tokens of heavier ranks). Flags may then differ per rank.
  - forced modes: `mc_rs` (≈ NVLS), `uc_pull` (≈ A2A-V), `mc_pull`, `uc_rs`.
- Under token imbalance (`--imbalance`), `rank` was the best policy measured: up to 1.8×
  faster than `auto` at EP=4 and up to 1.45× at EP=64. At EP=64, `kappa` can be slower than
  `auto`. Details are in the design doc's §9.

**Expert contract.** The expert always writes a zero row for tokens with no local expert
(`write_unrouted_rows=True`): the multicast-reduce combine sums every rank's row, and the pull
combine ignores those rows.

**Numerics.** Dispatch copies rows bit-exactly in both modes. The pull combine is bit-exact
against `bf16(m·x)`. `multimem.ld_reduce` (as in RSV-V) is within 1 bf16 ulp of it, with
~87–88% of elements bit-exact in `--validate`. So the combine output can change by 1 ulp
when B crosses the combine threshold; pin `--hybrid-threshold D,-1` (combine always pull) for
batch-invariant runs.

## Expert compute (`--expert vllm`)

`--expert vllm` runs Megatron's inference expert kernel `vllm_fused_moe`
(`megatron/core/inference/moe/vllm_fused_moe.py`: BF16 grouped GEMMs, squared ReLU, routing
probs applied in the fp32 top-k reduction) between dispatch and combine, imported from the
Megatron-LM checkout given by `--megatron-path` (or `$MEGATRON_PATH`). Each rank holds random
BF16 weights for its `512 / EP` local experts (`fc1` `[E, moe_intermediate_size,
moe_latent_size]`, `fc2` the transpose shape), and the kernel's intermediates are allocated
once for the worst case (`per_rank_cap × EP` rows), so nothing allocates inside the graph.
Each impl calls it the way Megatron's MoE layer does for that dispatcher
(`experts.py: _vllm_forward`):

| impl | input rows | expert ids | valid rows | output written to |
|---|---|---|---|---|
| DeepEP | received rows (compact) | local, `-1` for other ranks (`local_expert_start=0`) | received count, on device (`handle.psum_num_recv_tokens_per_scaleup_rank[-1]`) | the bf16 combine input |
| NVLS | every gathered row | global | `valid_tokens` (step metadata) | the bf16 RSV buffer |
| NCCL | every gathered row (padding rows routed to `-1`) | global | EP × padded count | the fp32 ReduceScatter input (Megatron passes no buffer, and the kernel then returns fp32) |
| A2A-V | every row of the gathered layout | global | `valid_tokens` (step metadata) | the bf16 partial buffer; rows with no local expert are skipped (`write_unrouted_rows=False`) |
| Hybrid | every row of the gathered layout | global | `valid_tokens` (step metadata) | the bf16 partial buffer, zero rows for tokens with no local expert (`write_unrouted_rows=True`) |

`num_tokens_hint` (it only picks the kernel's tile sizes) is local tokens × EP, as in Megatron.
At small EP the experts dominate the step: at EP=4, Super's 128 local experts are ~1.4 GB of
weights per layer per GPU, and nearly all of them get a token even at B=64.
`expert_breakdown.py` splits one rank's expert step by kernel on a single GPU, for the gathered
(NVLS / A2A-V) and compact (DeepEP) layouts.

## Correctness check (`--validate`)

`torchrun --nproc_per_node=4 run.py --impl nvls,deepep,nccl,a2a --validate` runs known-value
checks per impl (random tensors, verified element-wise) then exits without timing. It
confirms NVLS AGV-V gathers each rank's tokens to the right global offset, NVLS RSV-V sums
across ranks and scatters to the right owner, NCCL's padded AllGather→ReduceScatter
round-trips correctly, and DeepEP's timed dispatch→combine layer round-trips to `m·x` (`m` =
#destination ranks). For A2A-V, tokens are routed to random subsets of ranks (plus an all-`-1` padding
token) and it checks that routing and probs arrive bit-exact on every rank, routed hidden
rows arrive bit-exact, rows not routed to a rank are left untouched (NaN sentinels), and
combine returns `bf16(m·partial)` bit-exact. For Hybrid, it checks:
- the policy flags against a host reference for random policies;
- every forced mode with the A2A-V checks made mode-aware: multicast dispatch delivers *every*
  row, and the multicast-reduce combine reads zero rows on non-destination ranks and is
  within 1 bf16 ulp;
- one captured graph replayed under seven policies, each replay following the mode its flags
  select.

With ≥2 impls built it also **cross-checks**
that all of them produce the same combine output on identical inputs (all compute `m·x`;
NCCL is exact in fp32, NVLS, DeepEP & A2A-V combine in bf16 and agree within bf16 rounding).
With `--expert vllm` it also runs one timed layer per impl (dispatch → `vllm_fused_moe` →
combine, exactly as `decode_step` runs it) on the benchmark's own inputs (uniform routing,
softmax probs) and checks each output against a plain-torch MoE reference
(`VllmExperts.reference`, which regenerates every rank's expert weights from their seeds) to 3%
of the output's max magnitude, and the impls against each other to 2%.
Prints `[PASS]/[FAIL]` per check and a verdict reduced across ranks (exit 0/1); tested at a
full batch (`B=2·ep`) and the 0-token-rank case (`B=1`).

## Setup

**Requirements.** GPUs in a **single NVLink domain** (one node, or several nodes of one
NVL72 rack; validated on 4× B200 / GB200 and GB300 NVL72) and an NGC-style PyTorch container
(validated: CUDA 13, torch 2.11/2.12, **Triton 3.6/3.7**, `torch.distributed._symmetric_memory`
+ multicast, NVRTC/ptxas). Multi-GPU launched with `torchrun` or one task per GPU
(`launch_rank.sh`). NVLS and A2A-V need Hopper+ (SM ≥ 9) with NVLink + symmetric memory, and
torch ≥ 2.10 (for `set_signal_pad_size`).

**1. Container image with DeepEP ≥ 2.5.** The DeepEP bencher uses `deep_ep.EPBuffer`
(DeepEP 2.5), which needs NCCL ≥ 2.32.3 and reuses PyTorch's NCCL communicator, so torch and
DeepEP must load the same `libnccl` (DeepEP checks this at import). `build_deepep_image.sh`
builds such an image (`mcore_inference+fa4+deepep.sqsh`) from the mcore-inference + FA4 image:
it replaces the system NCCL with 2.32.3, installs DeepEP at a pinned commit plus DeepJIT's
`libdw`, and writes a `.manifest.txt` of the versions next to the image. Run it inside a pyxis
container of the base image as remapped root (see the script's header). Validated with that
image: torch 2.11, Triton 3.6, CUDA 13.2, NCCL 2.32.3, `deep_ep` 2.5.0+f6aa38f. NVLS, NCCL and
A2A-V need only torch + Triton. DeepEP JIT-compiles its kernels on first use (DeepJIT), once
per `num_sms` value.

**2. Megatron-LM (only for `--expert vllm`).** A checkout that has
`megatron/core/inference/moe/vllm_fused_moe.py`; pass it as `--megatron-path` or set
`MEGATRON_PATH`. Checkouts from before `vllm_fused_moe`'s `write_unrouted_rows` option also
work (A2A-V's expert step then writes every row).

**3. Get an interactive allocation in the container** (adjust account/partition/image):
```bash
srun -p batch --account=<ACCT> --qos=interactive -t 2:00:00 --nodes=1 --exclusive \
  --gpus-per-node=4 --container-image <mcore_inference+fa4+deepep.sqsh> \
  --container-mounts "/home:/home,/lustre:/lustre" --pty /bin/bash
```

**4. Run:**
```bash
# correctness first (no timing) — see the Correctness section
torchrun --nproc_per_node=4 run.py --model super --impl nvls,deepep,a2a --validate

# quick smoke (--num-layers defaults to the model's MoE block count: 40 Super, 48 Ultra)
torchrun --nproc_per_node=4 run.py --model super --impl nvls,deepep,a2a --batch-sizes 4 \
    --reps 3 --warmup 2

# full batch-size sweep -> results/results.csv (DeepEP num_sms defaults to auto)
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

# with Megatron's experts: check the expert path, then dispatch -> vllm_fused_moe -> combine
torchrun --nproc_per_node=4 run.py --model super --impl nvls,deepep,nccl,a2a --expert vllm \
    --megatron-path /path/to/Megatron-LM --validate
torchrun --nproc_per_node=4 run.py --model super --impl nvls,deepep,a2a --expert vllm \
    --megatron-path /path/to/Megatron-LM --per-rank-cap 2048 \
    --batch-sizes 64,256,1024,4096,8192 --reps 20 --warmup 6 --out results_vllm.csv
```
Or submit `run.sbatch` (edit the SBATCH headers / `CONTAINER_IMAGE` for your cluster,
then `sbatch run.sbatch` from the repo directory; multi-node: `sbatch -N 4 --segment=4
run.sbatch`; pick the impls with `IMPL=nvls,deepep,a2a sbatch run.sbatch` and the model with
`MODEL=super sbatch run.sbatch`, default `ultra`; add Megatron's experts with
`EXPERT=vllm MEGATRON_PATH=/path/to/Megatron-LM`, which caps the default sweep at B=128K
(`BATCH_SIZES` overrides it); outputs are named `results_<model>[_<expert>][_<N>n].csv`). For
one-task-per-GPU launches without torchrun, `launch_rank.sh` sets
`RANK`/`WORLD_SIZE`/`MASTER_ADDR` from Slurm and forwards its arguments to `run.py`.
`run_ep64_current.sh` is the EP=64 (16-node) communication-only and with-experts pair at the e2e
buffer size, run on every task of such a job; `plot_ep64_vs_e2e.py` plots its output next to
the e2e decode benchmark.

**Outputs.** CSVs are written to `results/` and plots to `plots/` (bare `--out`/`--csv`
names are resolved there; override with `--results-dir` / `--plots-dir`, or pass a path).

`run.py` flags: `--model {super,ultra}` (default `ultra`), `--impl` (comma-separated list of
`deepep`, `nvls`, `nccl`, `a2a`, `hybrid`, or the aliases `both` = deepep,nvls and `all` =
deepep,nvls,nccl; e.g. `--impl deepep,a2a`), `--batch-sizes`, `--phases` (comma list of
`decode_step` (default), `dispatch`, `combine`), `--hybrid-modes` / `--hybrid-threshold` /
`--hybrid-kappa` (see [Hybrid](#hybrid)), `--imbalance` (rank 0 holds ρ× the mean token
count, the others split the rest; default 1 = balanced), `--cta-cap` (override the CTA-grid
cap of NVLS, A2A-V and Hybrid; default 4× SMs at EP ≤ 8, 1× above), `--num-layers` (default: the
model's MoE block count), `--expert {identity,vllm}` (default `identity`) with
`--megatron-path`, `--per-rank-cap` (per-rank token capacity of every buffer; default
`ceil(max B / EP)`, or rank 0's largest share with `--imbalance`; Megatron's e2e uses 2048), `--deepep-num-sms` (comma list of `auto` and/or
SM counts — even, clamped to the device SM count; default `auto`), `--timing {graph,eager}`,
`--reps`, `--warmup`, `--seed`, `--out`, `--results-dir`, `--validate` (correctness checks then
exit — see above). CSVs record `model`, `num_layers`, `expert` and `per_rank_cap` per row.

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
- `models.py` — Nemotron-3 Super / Ultra presets (`--model`): layer pattern, MoE layer count, experts, top-k, `hidden_size`, `moe_latent_size`, `moe_intermediate_size` (torch-free; also used by the plotter).
- `common.py` — config (shape from the model preset), global→per-rank token split, routing/input gen, `time_region` (CUDA-event timing), `VllmExperts` (the `--expert vllm` step and its torch reference).
- `bench_deepep.py` / `bench_nvls.py` / `bench_nccl.py` / `bench_a2a.py` / `bench_hybrid.py` — the benchers; each exposes `decode_step(num_layers)` (returns the last layer's output) + `validate()` + `functional_roundtrip()`, and runs `self.expert` between dispatch and combine when set.
- `nvls/` — vendored, isolated NVLS collectives (zero Megatron deps; changes vs upstream: the Triton-3.6 widen, the block cap defaulting to the device SM count instead of 128, import isolation, and the optional hybrid `policy` in `metadata.py`), plus `nvls/torch_symm_triton/all_to_all_v.py` (the A2A-V kernels) and `nvls/torch_symm_triton/hybrid_v.py` (the hybrid kernels).
- `docs/hybrid_dispatcher_design.md` — design of the hybrid dispatcher.
- `plot_results.py` — plot `--x B` (default) or `--x num_sms`, in milliseconds, for `--phase decode_step` (default), `dispatch` or `combine`; the title shows the CSV's model, MoE layers per step and expert mode.
- `expert_breakdown.py` — per-kernel time of one rank's `vllm_fused_moe` step on one GPU (no communication), per input layout.
- `plot_ep64_vs_e2e.py` — EP=64 microbenchmark (communication only and with experts) next to the e2e decode benchmark.
- `build_deepep_image.sh` — builds the container image with NCCL 2.32.3 and DeepEP 2.5 (see Setup).
- `launch_rank.sh` — per-task launcher (one Slurm task per GPU = one EP rank).
- `run.sbatch` — SLURM batch template.
- `run_ep64_current.sh` — the EP=64 communication-only + with-experts runs at the e2e buffer size.
- `results/`, `plots/` — benchmark CSVs and plots (created on demand).

## Caveats
- **Combine precision.** NVLS, DeepEP and A2A-V combine in **bf16**; NCCL combines in
  **fp32**, as Megatron's NCCL path does with `vllm_fused_moe` (which returns fp32 when given
  no output buffer). NVLS's ReduceScatter-V moves bf16 operands but **accumulates the cross-rank sum in
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
- **Random experts, uniform routing.** `--expert vllm` uses random weights and uniform routing,
  so every expert gets about the same load; real routing is skewed, which changes the GEMM
  work per rank. The weight values themselves do not affect the timing.
- **Sample plots.** `plots/comm/` and `plots/comm_expert/` hold the EP=4/16/32/64 sweeps of
  a2a, nvls and DeepEP (`EPBuffer`, at `auto` and at the device SM count) for Super and Ultra on
  GB200, communication only (B up to 1M) and with `--expert vllm` (B up to 128K). The older
  `plots/old_GB300/results_ultra_ep*.png` are GB300 runs from before the move to `EPBuffer`: their
  DeepEP lines are DeepEP v2 (`ElasticBuffer`) at `num_sms` = the device SM count, communication
  only. `plots/hybrid/` holds the hybrid (auto) next to AGV-V/RSV-V and A2A-V on GB200,
  communication only, per phase (`decode_step`, `dispatch`, `combine`) for Super and Ultra at
  EP=4 and EP=64 (see the design doc's Phase 0 results). `plots/hybrid_imbalance/` compares
  the hybrid's `auto`, `kappa` and `rank` policies under `--imbalance` ρ ∈ {1, 1.5, 2, 4} at
  EP=4 and ρ ∈ {1, 4, 16, 64} at EP=64.
- **A2A-V routing/probs multicast.** Every rank receives every token's 264-byte routing +
  probs row (`B × 264` bytes per layer), independent of EP. Hidden-row traffic shrinks as EP
  grows, so this becomes a larger share of dispatch traffic at large EP (~14% of incoming
  dispatch bytes at EP=16, ~30% at EP=64 for B=128K).
