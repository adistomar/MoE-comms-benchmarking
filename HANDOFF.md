# Session handoff — K3 fused MoE dispatch + NVLS variable-multicast idea

_Written 2026-07-15. This captures the conversation state so the work can resume on another
cluster (same filesystem, no session resume). Durable facts are also in the auto-memory at
`/home/psinghania/.claude/projects/-scratch-.../memory/` — esp. `k3-fused-kernel-debug-state.md`,
`hierarchical-dispatch-hypothesis.md`, `nvl72-single-rack-slurm.md`, `hier-build-env.md`._

---

## 0. TL;DR — where things stand RIGHT NOW

Two threads, in order of recency:

1. **DONE & validated:** built a **barrier-free, pipelined, fused hierarchical (AGv-inner / A2A-outer)
   MoE dispatch kernel**. It **beats DeepEP-v2 in the decode range** at EP=64 (B≤1024) at both
   k=10 and k=22. Wired into `run.py --hier-fused`, validated `m*x=0`.
2. **REFUTED:** the "middle-band" premise (grouped multicast beats full multicast at large B) — the
   AGv-inner hierarchy is dominated by NVLS *and* DeepEP for B≥4k. Reason: at realistic k, tokens
   route to most groups, so the "grouped" inner AllGather isn't sparse. **Dead end for large B.**
3. **CURRENT / LIVE:** a NEW, SEPARATE idea (collaborator's) — **per-token variable-size multicast
   dispatch**, a variant of the *pure NVLS multicast dispatch* (NOT the hierarchy). Pre-create nested
   aligned multicast groups (sizes 4,8,16,32,64), per token multicast only to the smallest group
   spanning its dest ranks. **Feasibility probe PASSED at EP=64 (all 31 groups valid) — next is to
   BUILD the prototype off `bench_nvls.py` (see §5).** This is where to resume.

---

## 1. Repo context

3-way MoE dispatch/combine micro-bench for inference decode in one NVLink domain. Read `CLAUDE.md`
and `README.md`. Impls: `bench_deepep.py` (A2A), `bench_nvls.py` (NVLS AGv/RSv multicast),
`bench_nccl.py`, and the new `bench_hier.py` (hierarchical). Driver `run.py`, shared `common.py`,
vendored NVLS in `nvls/`. Metric: per-layer decode latency (µs), CUDA-graph, max across ranks.

**Cluster/run essentials (see `nvl72-single-rack-slurm.md`, `hier-build-env.md`):**
- Container: `/lustre/fsw/portfolios/nemotron/projects/nemotron_sw_pre/users/psinghania/gtp_latest/gtp_latest.sqsh`
  (torch 2.12 / triton 3.6 / cuda 13.2). `/lustre` == `/scratch` (symlink).
- **NVLS needs all nodes in ONE NVL72 rack** → always `sbatch --segment=16` (this is the correct flag
  on this/the new cluster; the old `--switches=1@10:00` may not work — see `nvl72-single-rack-slurm.md`).
- Launchers: `test_hier.sbatch` (single node, 4 GPU, `TEST_CMD=...`), `test_multinode.sbatch`
  (multi-node, `TEST_CMD=...`, `SOURCE_DEEPEP=1` to enable DeepEP). Run `hier` WITHOUT DeepEP
  in-process (DeepEP's LD_PRELOAD'd NCCL deadlocks the symm-mem path).
- **Per-rank Triton cache is set in `run.py`/`test_hier.py` tops** (`TRITON_CACHE_DIR=/tmp/..._{RANK}`)
  — required, else 64 ranks race the shared /lustre Triton cache → `OSError: Stale file handle`.
- **Fast debug loop (if available):** the `slurm-broker` MCP holds a persistent 4-GPU container and
  execs `timeout`-wrapped `torchrun` in seconds. It may not exist on the new cluster; fall back to
  `sbatch` cycles. When using it: WAIT ~2 min for squashfs extraction before the first exec.

---

## 2. What was built (the fused hierarchical dispatch)

**Design:** ranks form a G×g grid, `rank = group_id*g + pos`. Rows (g ranks) = inner NVLS multicast
AGv group; columns (G ranks) = outer directed-P2P A2A group. Dispatch = outer scatter → inner AGv;
combine = inner RSv → outer gather. Group-major expert placement. `bench_hier.py` = `HierBencher`.

**The fused kernel** (`nvls/torch_symm_triton/directed_p2p.py`, `_fused_dispatch_kernel` /
`fused_dispatch`): ONE persistent 148-CTA kernel replacing the 2 global barriers of scatter+AGv with
**per-landed-row release/acquire flags** in a dedicated `col_cap`-sized flag region of the dispatch
buffer. Every CTA does phase-1 scatter (set flags) then phase-2 multicast (wait on flags), then one
row barrier — the "unified CTA" design (all SMs serve both phases; per-token flags overlap them).

**`masked_copy` kernel** (same file): in-kernel combine mask (`rsv[t]=ah[t]` iff token routes to a
local expert, else 0; rsv pre-zeroed) — replaced `local_mask`+`torch.where`+casts. Cut `mask_rsv`
18.9→6.7 µs.

**Integration:** `run.py --hier-fused [--hier-fused-max-n 48] --hier-g <g>`. `HierBencher.fused` +
`fused_max_n` (B-threshold: fused per-token dispatch for small per-rank token counts, bulk staged
above — fused blows up at large B). `_use_fused()` decides via `all_reduce(MAX n)` in setup_batch
(consistent across ranks, resolves at graph capture).

### ⚠ CRITICAL learnings (do not re-derive — cost ~15 cluster iterations)
- **The hang was a thread-divergent `if ftid < 1:` immediately before `__syncthreads()`** — lanes
  don't reconverge → barrier deadlock. NOT the atomic/visibility/signal-pad/livelock (all red herrings).
- **The working consumer wait is UNIFORM, bounded, early-exit** (all lanes same addr → uniform loop
  cond → no divergence):
  ```python
  v = 0; i = 0
  while (v != 1) & (i < wait_iters):
      v = tl.atomic_add(flag_u32 + r, 0, sem="acquire", scope="sys").to(tl.int32)
      i += 1
  ```
  Producer sets: `tl.atomic_xchg(base_u32 + FLAG_OFF//4 + row, 1, sem="release", scope="sys")`.
- Triton 3.6: NO `break` in for-loops; loop-carried var must keep ONE type (atomic→uint32 → `.to(int32)`).
- Flags live in a dedicated symm-BUFFER region (not the signal pad — too small at large B). Normal
  symm-buffer regions DO support cross-device atomics.
- Diagnostic `dbg`/`wait_iters` knobs + `NO_FLAGS`/`SKIP_WAIT` constexprs still in the kernel (cheap;
  strip for a clean final version).

---

## 3. Key results (EP=64, one NVL72 rack, graph, per-layer µs)

**k=22, best config (g=8, threshold):** hier beats DeepEP across the whole decode range:
| B | NVLS | DeepEP | hier_fused |
|---|---|---|---|
| 1 | 11.9 | 41.9 | **34.0** |
| 128 | 13.5 | 49.2 | **40.8** |
| 1024 | 20.3 | ~52 | **43.4** |
| 8192 | 66.5 | 64.8 | ~90 (staged via threshold) |

**k=10, g=8, threshold:** NVLS 11/14/20/39/65, DeepEP 37/44/45/~48/51, hier 33/40/43/64/90
(B=1/128/1024/4096/8192). hier beats DeepEP at B≤1024.

**Group size g (the NVLS↔DeepEP knob):** g=1 = pure A2A (DeepEP), g=P = pure AGv (NVLS). **g=8 is the
best decode default** (beats DeepEP through B=1024; g=4 only to B=128). At B=1, g barely matters
(gap to NVLS is combine overhead, not g). Larger g worse at large B.

**Large-B refutation (job 2511269):** NVLS vs hier(g=8): B=4096 39/68, 8192 65/100, 16384 115/164,
32768 213/296. **hier NEVER beats NVLS; gap grows.** Grouped multicast doesn't scale better than full
multicast — at k=10-22 tokens hit ~50-75% of groups so the inner AGv isn't sparse, and the hierarchy
adds an outer hop on top. **The AGv-inner hierarchy cannot win the middle/large band at realistic k.**

---

## 4. Honest conclusions

- **Fused hierarchy = a real small-B (decode) win vs DeepEP.** Genuine, validated.
- **Still ~3× NVLS at B=1** — combine/barrier overhead, not algorithm. To close it: fuse the combine
  (`gather_reduce` ~19µs still has a barrier + torch ops). This is the natural "increment 3".
- **Middle/large band is a dead end for this design.** Only a sparse A2A/A2A hierarchy could beat
  DeepEP there (abandons multicast), OR the variable-multicast idea in §5.

---

## 5. ⭐ CURRENT THREAD — per-token variable-size multicast dispatch (NEW, live)

**Idea (collaborator):** this is a **variant of the pure NVLS multicast dispatch** (`bench_nvls.py` +
`nvls/torch_symm_triton/variable_collectives.py`), NOT the hierarchy. Pre-create **nested aligned
multicast groups** — size 4 {0-3},{4-7},…; size 8 {0-7},…; … up to {0..63}. Per token, pick the
SMALLEST group spanning its destination ranks and multicast only to that group (instead of always
size-64). Average case < 64 → cheaper multicast. **Builds off the NVLS AGv/RSv path; uses per-GROUP
barriers (each group's own `symm_mem_sync`), like NVLS — NOT the hierarchy's per-token flags.**

**Feasibility gate = can a rank hold all ~5 nested multicast groups (31 total at P=64)?** We already
do N=2/rank (row+col in the hierarchy), so the mechanism is proven; the open question is the NVSwitch
multicast-group-count ceiling for 31 overlapping groups.

**Probe written:** `probe_nested_mcast.py` (standalone; creates every nested aligned group via
`dist.new_group`, rendezvouses a small symm buffer per group via `SymmetricMemoryBuffer`, asserts
every rank gets a valid `multicast_ptr` for each group it joins; reports `FIRST_FAIL` if a ceiling
is hit).

**Probe status: ✅ BOTH PASSED (feasibility gate CLEARED).**
- EP=4 (`hier-test-2656792.out`): PASSED — 3 groups, 2/rank, all valid multicast.
- **EP=64 (`hier-mnode-2656793.out`): PASSED — all 31 nested groups create valid `multicast_ptr`s;
  every rank joins its 5 (`s4@0:OK s8@0:OK s16@0:OK s32@0:OK s64@0:OK`).** So the hardware/torch
  supports the full nested set — **the idea is feasible on torch symm-mem; no need for NCCL device API.**
  (Resubmit form if ever needed: `TEST_CMD="probe_nested_mcast.py --min-size 4" sbatch --qos=short
  --nodes=16 --segment=16 --time=00:08:00 test_multinode.sbatch`.)

**Build plan (if probe passes) — extend `bench_nvls.py`, NOT the hierarchy:**
1. Create the 31 nested groups once (dist.new_group + `SymmetricMemoryManager.get_buffer` per group,
   small buffers). Store their `multicast_ptr`s in a device array.
2. In `setup_batch`: bucket each rank's tokens by `group_idx` = smallest aligned power-of-2 block
   spanning the token's dest ranks (device-side, graph-capturable — like the count-matrix code).
3. Multicast AGv **per group** (or one kernel that indexes the multicast VA by the token's group_idx —
   a small extension to `multimem_all_gather_v`'s single `multicast_ptr`); each group barriers among
   its own ranks. Mask + variable-group RSv for combine.
4. Compare vs flat NVLS at EP=64. NOTE: benefit requires routing LOCALITY (dest ranks clustered).
   Uniform random routing (the current bench) → max dest rank ~57 → almost always size-64 → ~0 benefit.
   `/tmp/.../mcast_groups.py` (a quick sim I ran) confirmed: uniform → avg block 63.9/64; only with
   locality (windowed routing) does avg block drop (W=16 → 17.5/64). So this pays off only under
   skewed/local routing or popularity-aware expert placement — worth stating up front to the user.

---

## 6. Immediate next steps (resume order)
1. **BUILD the variable-multicast NVLS dispatch prototype** — feasibility is confirmed (probe PASSED,
   all 31 groups valid). Follow the §5 build plan, extending `bench_nvls.py` (NOT the hierarchy).
   Start small: get the per-token group_idx bucketing + one variable-group AGv working & validated
   at EP=4, then EP=64. Remember: benefit needs routing LOCALITY (uniform routing → ~size-64 always,
   ~0 gain) — so also plan to test with a skewed/local routing or popularity-aware placement.
2. (Independent, if decode perf on the hierarchy matters) fuse the combine barriers to close the last
   gap to NVLS.
- Use `sbatch --segment=16` for all EP=64 runs (single NVL72 rack).

## 7. New/changed files this session
- `nvls/torch_symm_triton/directed_p2p.py` — `_fused_dispatch_kernel`/`fused_dispatch`, `masked_copy`.
- `bench_hier.py` — `_dispatch_fused`, `fused`/`fused_max_n`/`_use_fused`, `masked_copy` in `_mask_rsv`,
  `all_reduce(MAX n)` in setup, `rsv.zero_()`/`dflag.zero_()`.
- `run.py` — `--hier-fused`, `--hier-fused-max-n`, per-rank `TRITON_CACHE_DIR`.
- `test_hier.py` — milestones `k3fused`/`decompose`, per-rank `TRITON_CACHE_DIR`, diagnostic args.
- **`probe_nested_mcast.py`** — the live feasibility probe (§5).
- CSVs: `results_fused_ep64.csv`, `results_fused_k10_ep64.csv`, `results_g{8,16}_ep64.csv`,
  `results_bigB_g8.csv`, `results_triangle_ep64_k10.csv`.
