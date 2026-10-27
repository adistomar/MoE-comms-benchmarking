# Hybrid NVLS dispatcher: token-count-adaptive multicast / unicast

Design for one dispatch kernel and one combine kernel that choose, **on device, once per
decode step**, between the multicast path (AGV-V dispatch / RSV-V combine) and the unicast
path (A2A-V dispatch / pull combine). The choice uses the token counts aggregated across all
EP ranks, which `fused_metadata_update` already computes.

## TL;DR

- AGV-V/RSV-V and A2A-V already share the row layout (`rank_token_offset[s] + t` in
  `[global_max, *]` symmetric buffers), the per-step metadata, the `pid >= ep_max_tokens`
  early exit, the grid, and the per-CTA-pair barrier. Inside the per-token loop, each pair
  differs only in how a row is moved:
  - **dispatch:** one `multimem.st` of the hidden row, or D peer `st.global` (D = distinct
    destination ranks of the token);
  - **combine:** one `multimem.ld_reduce`, or D peer loads summed in fp32.
- **The hybrid** is therefore a branch inside each kernel on a mode flag. The metadata kernel
  computes the flags once per step from the token counts against a policy held in a device
  tensor: from `valid_tokens` (and optionally `ep_max_tokens`) under the global policies,
  from every rank's count under the per-rank policy. There are no extra launches, no host
  sync, and nothing to recapture.
- **Neither mode needs agreement between ranks.** Routing and probs are multicast in both
  modes, and the expert only reads rows routed to its rank, which arrive either way. An
  `ld_reduce` combine needs zero rows on ranks a token was not routed to, so the expert always
  writes them (`write_unrouted_rows=True`); the pull combine ignores them (§3.5, §3.6).
- **Result (Phase 0, §8):** the hybrid tracks the envelope of the two curves on GB200 (Super
  and Ultra, EP=4 and EP=64, communication only).
  - Small B: 26–34% faster than A2A-V at B=1.
  - Large B: 16–19% faster than AGV-V/RSV-V at EP=4, and 2.4–2.7× at EP=64.
  - Crossovers are separate per collective: B≈0.7–1.4k at EP=64, and ≈1.4–4k at EP=4.
- **Imbalanced token counts (§9):** a per-rank policy multicasts only the heaviest senders.
  - In every strongly imbalanced case measured it is the best policy: up to 1.8× faster than
    the global rule at EP=4 and 1.45× at EP=64.
  - In the mild range typical of decode-only instances (ρ ≤ 1.25 at EP=64) it is mixed or
    worse, so it needs a minimum-imbalance floor ρ* ≈ 1.3 (§9.1).
  - One GPU multicasting into a 64-GPU group reaches only about 45 GB/s, independent of the
    CTA count; most likely a fabric limit (§9.1).
- **Modeling (§10):** an α-β model with a per-port traffic β. The traffic part predicts the
  large-B ratio u/EP to within about 0.1; the α part explains the crossover trend.

## 1. The two collectives, side by side

| | AGV-V → RSV-V (`variable_collectives.py`) | A2A-V (`all_to_all_v.py`) |
|---|---|---|
| CTA ↔ token | one CTA per local token, grid-stride, grid `min(per_rank_cap, max_num_blocks)` | same |
| early exit | `pid >= ep_max_tokens` (same on every rank) | same |
| dispatch: hidden | `ld_128` local → **one** `multimem.st` to row `offset+t` | `ld_128` local → walk top-k (`_next_destination`, int64 "sent" mask) → **D** predicated `st.global` through `buffer_ptrs_dev[dst]` |
| dispatch: routing, probs | `multimem.st` (3-tensor kernel) | `multimem.st` (`_multicast_row`), same as AGV |
| dispatch barrier | trailing, release/acquire, hidden buffer's signal pad | same |
| expert input | every row present on every rank | only rows routed to this rank are present ("holes" elsewhere); the expert reads only routed rows anyway |
| expert output | **every** valid row (zero row if no local expert): `write_unrouted_rows=True` | routed rows only: `write_unrouted_rows=False` |
| combine barrier | leading, **relaxed** | leading, **relaxed** (since e86c598; the README now says so too) |
| combine | one `multimem.ld_reduce.add.acc::f32.v4.bf16x2` per pack → local `st` | walk top-k again → **D** predicated peer `ld.global.v4` → fp32 sum in fixed order → bf16 → local `st` |
| limits | 8/16-B rows | EP ≤ 64 (int64 mask), 16-B hidden rows, bf16 partials, int64 routing |

Both use the same five symmetric buffers: hidden, routing, probs, combine partial, and
metadata. Each symmetric-memory handle exposes both `multicast_ptr` and `buffer_ptrs_dev`.
So a single set of buffers supports both modes; no extra memory is needed.

## 2. Why each wins where it does

### 2.1 Bandwidth regime (large B): traffic per GPU per layer

Balanced load, B = `valid_tokens`, r = hidden row bytes. With uniform top-k routing:
- q = P(a given rank holds at least one of a token's experts) = 1 − C(E−E/EP, k)/C(E, k);
- D = EP·q, the expected number of distinct destination ranks;
- u = (EP−1)·q = D − q, the expected number of *remote* destinations. The sender is itself
  a destination with probability q, and that copy is a local write.

**Where q, D and u come from.**
- **q, exactly.** Fix a token and one rank r. Rank r holds m = E/EP experts, and receives the
  token if at least one of its k = 22 distinct experts is among them. It misses the token only
  if all k are drawn from the other E − m experts, so P(miss) = C(E−m, k)/C(E, k) and
  q = 1 − C(E−m, k)/C(E, k).
- **q, approximately.** A single expert pick misses rank r with probability 1 − 1/EP. Treating
  the k picks as independent gives q ≈ 1 − (1 − 1/EP)^k. At EP=64 that is 1 − (63/64)^22 ≈
  0.293; the exact value is 0.298, slightly higher because distinct picks cannot repeat an
  expert.

  | EP | per-pick miss 1 − 1/EP | approx q | exact q |
  |---|---|---|---|
  | 4 | 0.75 | 0.998 | 0.998 |
  | 16 | 0.9375 | 0.758 | 0.766 |
  | 32 | 0.969 | 0.502 | 0.510 |
  | 64 | 0.984 | 0.293 | 0.298 |

- **D and u via indicators.** Let X_r = 1 if rank r is one of the token's destinations, else 0.
  Then E[X_r] = q: a probability is also the expected value of a 0/1 count. The token's
  destinations number Σ_r X_r, so D = Σ_r E[X_r] = EP·q. Its remote destinations number
  Σ_r X_r − X_s, where s is the sender. By linearity of expectation (which holds although the
  X_r are correlated), u = D − E[X_s] = D − q = (EP−1)·q.
  - Subtracting q therefore subtracts an expected count: how many of the token's destinations
    are, on average, the sender itself. That copy is a local write.
  - "D − 1" would assume the sender is always a destination; at EP=64 it is one only 29.8% of
    the time, so u = 18.77 rather than 18.07.
- q is also the expected fraction of all tokens that one rank receives, so each GPU's unicast
  ingress is q·N. With skewed routing it becomes a per-rank q_r, measurable from router
  outputs as (tokens touching rank r) / (all tokens).

This ignores the 264 B/token routing+probs multicast, which every GPU receives in both modes.

| | MC dispatch | UC dispatch | RS combine | pull combine |
|---|---|---|---|---|
| egress | B/EP · r | B·u/EP · r | **B · r** (the switch reads every row from every GPU) | B·u/EP · r |
| ingress | **B · r** (every row, own rows included, through the switch) | B·u/EP · r | B/EP · r | B·u/EP · r |

At large B the unicast path wins by EP/u:

| EP | q | D = EP·q | u = (EP−1)·q | predicted UC/MC = u/EP | measured A2A/AGV at B=1M, Super | Ultra |
|---|---|---|---|---|---|---|
| 4 | 0.998 | 3.99 | 3.00 | 0.75 | 0.82 | 0.80 |
| 16 | 0.766 | 12.25 | 11.48 | 0.72 | 0.77 | 0.76 |
| 32 | 0.510 | 16.32 | 15.81 | 0.49 | 0.55 | 0.46 |
| 64 | 0.298 | 19.07 | 18.77 | 0.29 | 0.39 | 0.29 |

The measured ratios come from `plots/comm/*.png`, GB200, comm only. Super's residual is
larger because the routing+probs multicast is 13% of a Super row and 6% of an Ultra row. Even
at EP=4, where D≈EP, unicast wins: its own-rank copy never crosses NVLink, while the
multicast copy loops back through the switch.

**Multicast costs more in aggregate, less at the sender.** A multicast token adds one row to
*every* GPU's ingress (EP rows in total); a unicast token adds u < EP. With balanced load
every port carries the same share, so unicast always wins in the bandwidth regime. What
multicast saves is the sender's own egress (1 row instead of u). That only matters when one
sender's egress is the busiest port, i.e. with imbalanced token counts (§2.4, §9).

### 2.2 Latency regime (small B)

At B=1 a decode step costs (read from the plots, ms per step):

| | Super EP4 | EP16 | EP32 | EP64 | Ultra EP4 | EP16 | EP32 | EP64 |
|---|---|---|---|---|---|---|---|---|
| AGV/RSV | 0.42 | 0.43 | 0.44 | 0.45 | 0.56 | 0.59 | 0.51 | 0.52 |
| A2A-V | 0.58 | 0.61 | 0.61 | 0.65 | 0.73 | 0.81 | 0.80 | 0.86 |
| gap / layer | 4.0 µs | 4.5 | 4.3 | 5.0 | 3.5 | 4.6 | 6.0 | 7.1 |

AGV/RSV sits at about 11 µs/layer, about one NVLink round trip plus the barrier for each of
the two kernels.

**Where A2A-V's extra latency comes from.** Measured per phase in §8: unicast dispatch costs
2 µs/layer more than multicast at EP=4 and 4.5 µs at EP=64, and pull combine 1.6–1.9 µs more
than `ld_reduce`. Per token, in both kernels:
- a serial 22-step top-k walk: a scalar routing load, then a dependent `peer_table[dst]`
  load, then a store or load;
- D peer operations, where the multicast path issues one;
- in combine, D outstanding 128-bit loads per thread plus fp32 unpacking.

The gap grows with EP (more destinations) and with the row width.

### 2.3 Crossover

With t_MC ≈ α_MC + B·c_MC and t_UC ≈ α_UC + B·c_MC·u/EP:

  T ≈ Δα / (c_MC · (1 − u/EP))

This explains the trend: a small slope advantage at EP≤16 pushes T up, a large one at EP=64
pulls it down. Measured crossovers (global B, power-of-two grid, so coarse):

| | EP=4 | EP=16 | EP=32 | EP=64 |
|---|---|---|---|---|
| Super (2 KB rows) | ~4k | ~4k | ~2k | ~1–2k |
| Ultra (4 KB rows) | ~4k | ~2–4k | ~1–2k | ~0.5–1k |

The linear model underestimates T at small EP (Ultra EP4: it predicts about 1.4k against a
measured ~4k). Mid-B is neither purely latency- nor bandwidth-bound. So thresholds should be
**calibrated from sweeps** (§4), not computed from asymptotic slopes.

With `--expert vllm` the crossovers are about the same (e.g. Ultra EP64: AGV ahead up to 512,
equal at 1k, A2A ahead from 2k). Expert work is the same in both modes except for the
zero-row writes. The hybrid always does those (§3.5), so in pull mode it pays the under-1%
that A2A-V skips.

### 2.4 Token imbalance (measured in §9)

With imbalance, the all-unicast path is limited by the heaviest sender's egress,
`ep_max · u · r`. Multicast fan-out happens in the switch, so all-multicast stays at
`valid · r`. In the bandwidth regime, then:

  all-multicast beats all-unicast ⇔ valid − u · ep_max < 0

At EP=4 a heaviest rank holding 1.33× the mean already favours multicast at every B; at EP=64
the factor is 3.4×. That is the global κ = u policy (§3.2). The per-rank rule of §9
multicasts only the senders whose fan-out is the bottleneck.

## 3. Design

```
step start ── fused_metadata_update (once per step; 1 CTA) ──────────────────────────────┐
   multicast local count → barrier → sum / prefix / max → step_metadata[0..2]              │
   NEW: policy(valid, ep_max) → step_metadata[3] = dispatch_mc, [4] = combine_rs           │
                                                                                            │
per MoE layer:                                                                              │
   hybrid_dispatch_v   if dispatch_mc: multimem.st hidden   else: unicast to D peers        │
                       routing + probs: multimem.st (both)        trailing barrier (rel/acq)
   expert              reads routed rows only (both);  write_unrouted_rows=True (both)      │
   hybrid_combine_v    leading barrier (relaxed)                                            │
                       if combine_rs: multimem.ld_reduce   else: pull D peers, fp32 sum   ←─┘
```

### 3.1 Where the decision lives

- **Input:** `valid_tokens = Σ_r local_tokens[r]` and `ep_max_tokens`. The metadata kernel
  computes both from counts it has multicast to every rank, so they are bit-identical
  everywhere.
- **Where:** the metadata kernel, once per step, writing two int32 flags into
  `step_metadata[3:5]`. Dispatch and combine read the flags; they do not recompute them. The
  flags are one source of truth for the step and cost nothing, since the kernel already has
  every rank's count in registers. Every layer of a step uses the same modes. Under the global
  policies the flags are identical on every rank; under the per-rank policy (§9) they may
  differ, which is safe (§3.6).
- **Why not per layer:** a routing-aware per-layer choice would need a new collective per
  layer for the ranks to agree, which costs about as much as the gap being recovered.

### 3.2 Policy

The policy is a device int32 tensor `[n0_d, kappa_q8_d, n0_c, kappa_q8_c, per_rank,
q_q16]`, identical on all ranks. The global part is:

```
dispatch_mc = valid*256 - kappa_q8_d*ep_max <= n0_d*256     (int64 arithmetic)
combine_rs  = valid*256 - kappa_q8_c*ep_max <= n0_c*256
```

- `kappa = 0` gives the plain global-token threshold `valid <= n0`. This is the default and
  what the balanced benchmark calibrates.
- `kappa = u` (Q8 fixed point) adds the imbalance term from §2.4. `HybridBencher`'s `kappa`
  mode uses it, with `n0` scaled by (1 − u/EP) so a balanced batch switches at the same B.
- **Per-rank term** (§9): two more policy entries, `per_rank` and q (Q16). When enabled,
  rank s's flag is also set if u·n_s > q·N + (1−q)·S_s. Flags may then differ per rank.
- **Integer math** makes the comparison bit-identical on every rank.
- **Overrides:** `n0 = -1, kappa = 0` forces unicast/pull; `n0 = INT32_MAX` forces
  multicast/ld_reduce.
- **Why a device tensor rather than a kernel argument:** the policy can be changed between
  graph replays without recapture. That serves tests (flip modes inside one captured graph),
  online tuning, and debugging.
- Dispatch and combine get **separate thresholds**. They are separate kernels with their own
  crossovers, and any combination is legal (§3.6).

### 3.3 Hybrid dispatch kernel

This is the A2A-V dispatch kernel with one branch on the hidden store. The routing/probs
multicast, the barrier, the early exit and the indexing are unchanged.

```python
pid = tl.program_id(0)
if pid >= tl.load(ep_max_tokens_ptr): return                # same rule as both today
use_mc = tl.load(dispatch_mode_ptr) != 0                    # same line as ep_max; same on all ranks
...
for token_offset in range(pid, local_tokens, tl.num_programs(0)):
    global_row = (rank_token_offset + token_offset).to(tl.int64)
    for channel_offset in range(0, HIDDEN_PACKS, BLOCK_SIZE):
        cols = channel_offset + tid; mask = cols < HIDDEN_PACKS
        (x, y, z, w) = ld_128(hidden_u64 + (token_offset*HIDDEN_PACKS + cols)*2, mask, False)
        if use_mc:                                          # AGV-V
            st_128(hidden_mc + hidden_byte_offset//8 + (global_row*HIDDEN_PACKS + cols)*2,
                   x, y, z, w, mask=mask, multicast_op=True)
        else:                                               # A2A-V
            sent = tl.full((), 0, tl.int64)
            for k in tl.static_range(TOPK):
                dst, first, sent = _next_destination(routing_ids, token_offset, sent, k, TOPK, NLE)
                peer = tl.load(peer_table + dst).to(tl.pointer_type(tl.uint64))
                st_128(peer + ..., x, y, z, w, mask=mask & first, multicast_op=False)
    _multicast_row(routing ...); _multicast_row(probs ...)  # both modes
sync_threads(); symm_mem_sync(..., hasPreviousMemAccess=True, hasSubsequentMemAccess=True)
```

The branch is uniform across the grid, so there is no divergence. Its cost is one scalar load
from the cache line that already holds `ep_max_tokens`. The release in the trailing barrier
covers `multimem.st` (as in AGV-V today) and peer stores (as in A2A-V today).

### 3.4 Hybrid combine kernel

```python
ep_max_tokens = tl.load(ep_max_tokens_ptr)
use_rs = tl.load(combine_mode_ptr) != 0                     # plain load, next to ep_max
if pid >= ep_max_tokens: return
symm_mem_sync(..., relaxed, relaxed); sync_threads()        # as both do today
rank_token_offset = tl.load(rank_token_offset_ptr)
if use_rs:
    _reduce_rows(...)   # the RSV-V token loop: multimem.ld_reduce (acc::f32.bf16x2) -> st
else:
    _pull_rows(...)     # the A2A-V token loop: D predicated peer loads, fp32 sum -> st
```

The combine is bf16-only, as A2A-V is today; the RSV-V fp32 (`REDUCE_F32`) variant is not
carried over.

**Codegen lesson (Phase 0).** The branch must sit outside the token loop, and each side must
be the stand-alone kernel's loop *verbatim*. Two seemingly neutral variants compiled to a pull
loop 1–2 µs/layer (10–20%) slower than A2A-V's at 16+ active CTAs per rank, at the same
register count:
- the branch inside the loop with a shared store, or a per-row offset shared across the 22
  peer loads;
- the mode flag loaded `volatile`, to stop LLVM sinking it below the barrier's inline asm. It
  becomes an `LDG.E.STRONG.SYS`, which also cost the multicast-reduce path ~0.4 µs.

Probed with throwaway kernel variants on identical buffers. The dispatch kernel's
inside-the-loop branch showed no such cost.

### 3.5 Expert contract

- **Input:** identical in both modes. Routing and probs cover all rows `[0, valid)` on every
  rank, and the expert reads hidden rows only for its local experts. Those rows are present
  whether they arrived by multicast or unicast. Holes and stale rows from earlier steps are
  never read.
- **Output:** the RS combine sums row `g` over **all** ranks, so ranks without an expert for
  `g` must hold a zero row. The pull combine never reads those rows.
  - **Decision: the expert always writes zero rows for unrouted tokens**
    (`write_unrouted_rows=True`, Megatron's default) in both combine modes. This keeps the
    expert contract mode-independent and needs no Megatron change. It also opens the way to
    removing the option altogether (A2A-V is its only `False` caller).
  - **Cost:** local zero-row writes in pull mode, about (1−D/EP)·B·r of HBM writes. For
    Ultra EP64 at B=128k that is about 375 MB, roughly 50 µs against an 8 ms layer: under 1%.
  - **Not pursued:** a device-side `write_unrouted` flag in `_moe_sum_kernel`, driven by the
    combine mode (`WRITE_UNROUTED: tl.constexpr`, `vllm_fused_moe.py:580`, used at `:660`).
    It would save that under-1% cost at the price of a mode-dependent expert contract.
  - **Rejected alternatives:**
    - Multicasting a zero row into the partial buffer from dispatch: it doubles multicast
      dispatch bytes near the crossover.
    - Zero-filling inside the combine: the per-CTA-pair barrier only orders rows read by the
      same CTA index. The zero-fill would need the owner offset of every row, and it adds
      local work on the latency-critical path.

### 3.6 Why switching is safe (correctness invariants)

1. **Same layout.** Every buffer uses row `rank_token_offset[s] + t` in both modes, so either
   kernel's output is a valid input to the other mode's next stage.
2. **Same barrier pairing.** The grid, the active-CTA rule (`pid < ep_max_tokens`) and the
   signal slots (`block_id * EP + rank`) do not depend on mode. Barriers pair up identically,
   so mode can never cause a deadlock.
3. **Both modes may differ per rank.**
   - Dispatch: a rank that multicasts while a peer unicasts still delivers every routed row.
     The dispatch mode could even be chosen per token.
   - Combine: each owner chooses for its own tokens. An RS owner sums the row from every
     rank, which is valid because the expert always writes zero rows for tokens it did not
     process (§3.5). Had the expert skipped them, as A2A-V's does, the combine mode would
     have to agree across ranks.
   - The per-rank rule (§9) relies on this; `--validate` checks a step where rank 0
     multicasts and multicast-reduces while the other ranks unicast and pull.
4. **Buffer reuse holds in every mode, including across a mode flip between steps.**
   - Rank s's dispatch of layer L+1 starts only after all of its combine-L CTAs have finished.
     Combine-L CTA 0 waited on CTA 0 of **every** rank, and that rank's expert L had already
     finished in stream order. So overwriting any row on any rank (multicast writes all of
     them) is safe.
   - Partials read by s in combine L (one rank in pull mode, all ranks in RS mode) are
     overwritten only after the next dispatch barrier, which waits on s.
   - No buffer state carries across steps: pull ignores rows it did not request, and RS mode
     rewrites every valid row. Mode can therefore flip on any step, so no hysteresis is
     needed.
5. **Memory ordering.**
   - Dispatch keeps release/acquire, as both kernels do today.
   - Combine keeps the relaxed leading barrier that both RSV-V and A2A-V use. It relies on
     the expert kernel's writes reaching the home L2 at the kernel boundary before the
     relaxed signal. This assumption is inherited, not new; Phase 1 includes a
     repeated-replay stress validation.

### 3.7 Launch configuration and resources

- **Grid, block size, signal pad:** unchanged.
  - Grid: `min(per_rank_cap, default_num_blocks(EP))`, i.e. 4× SMs at EP≤8 and 1× SMs at
    EP≥16.
  - Block: next power of two of the hidden packs (128 threads for Super, 256 for Ultra).
  - Signal pad: `num_blocks · EP · 4` B via `ensure_signal_pad_size`.
  - The multicast path only runs when B ≤ T, where at most `T/EP` CTAs are active, so the
    grid cap is set by the unicast path's large-B needs, as it is today.
- **Registers (measured, GB200, Triton 3.6):**

  | kernel | RSV-V combine | A2A-V combine | hybrid combine | AGV-V 3-tensor | A2A-V dispatch | hybrid dispatch |
  |---|---|---|---|---|---|---|
  | registers | 18 | 110 | 110 | 26 | 27 | 32 |

  None spill. The fused combine carries the pull path's registers into RS mode. That costs
  nothing for Super (128 threads: 4 CTAs/SM fit either way). For Ultra (256 threads) only
  2 CTAs/SM fit where RSV-V fits 8, so the 608-CTA grid at EP≤8 runs in two waves once more
  than 304 tokens per rank are active. Measured: RS mode +0.4% at B=1k, +4% at B=4k and +0.9%
  at B=64k over RSV-V. Splitting the kernels (conditional graph nodes, §5) would remove it.
- **Code reuse:** import `_next_destination`, `_multicast_row`, `_ld_128_or_zero`,
  `_bf16_*` and `_f32_to_bf16x2` from `all_to_all_v.py`. Leave the existing AGV-V, RSV-V and
  A2A-V kernels untouched as baselines.

### 3.8 Numerics

- **Pull** sums bf16 partials in fp32 in a fixed order and rounds once to bf16
  (deterministic). It is bit-exact against `bf16(m·x)`.
- **`multimem.ld_reduce ... acc::f32 ... bf16x2` is *not* bit-identical to that.** Measured
  in Phase 0 (`--validate`, GB200, EP=4), summing m equal bf16 partials plus EP−m zeros:
  86.9% (Super) and 88.4% (Ultra) of elements are bit-exact, and the rest are off by exactly
  1 bf16 ulp. Single-token cases (B=1) were all exact. Upstream RSV-V's own check already
  allows `rtol=2e-2`, so this is the existing NVLS behaviour, not something new in the
  hybrid. The switch's accumulation or rounding is evidently not plain fp32 add +
  round-to-nearest-even of the exact sum.
- **Consequence:** the combine output can change by up to 1 bf16 ulp when B crosses `n0_c`.
- **Batch-invariant mode:** pin the combine mode there (`n0_c = -1` for pull, which is also
  the deterministic one). The dispatch mode never affects numerics, since both modes copy
  rows bit-exactly, so `n0_d` can stay adaptive.

## 4. Threshold calibration

1. **Calibrate with forced modes.** Sweep the four forced combinations
   (MC|UC dispatch × RS|pull combine) at B ∈ {256, 362, 512, …, 16k} (×√2 steps) for each
   `(GPU, model row width, EP)`. Take `n0_d` where the dispatch-only difference changes sign,
   and `n0_c` likewise for combine. The kernels are separate, so the two calibrate
   independently.
2. **Ship a defaults table** keyed by `(row_bytes, EP)` in the bencher, measured on GB200.
   For untested shapes, fall back to scaling T with `1/(1 − u/EP)` from the nearest
   measured EP.
3. **Optional online autotune.** Since the policy is a device tensor, `build()` can time both
   modes at a few B values on one captured graph and set `n0` without recapture.
4. **Imbalance.** `--imbalance ρ` and the κ / per-rank policies are measured in §9.

## 5. Alternatives considered

| option | verdict |
|---|---|
| Host reads `valid_tokens` and picks a kernel or graph | Rejected: needs a device-to-host sync per step, which the design exists to avoid. |
| Launch both kernels each layer, each exiting early unless it is the active mode | Rejected: an extra graph kernel node per collective per layer costs on the order of a µs of launch and drain latency, eating a large part of the 3.5–7 µs/layer small-B win. |
| CUDA-graph conditional IF/ELSE nodes set by the metadata kernel (`cudaGraphSetConditional`) | Deferred: would allow per-mode grids and registers, but needs a CUDA-C++ setter and PyTorch conditional-node capture. Worth it only if §3.7 register pressure shows up. |
| Per-rank multicast vs unicast | Implemented (§9): only the heaviest senders multicast, when their unicast fan-out is the bottleneck. |
| Per-token multicast vs unicast | Not pursued: within one sender, splitting its tokens between modes trades that sender's egress against everyone's ingress, which the per-rank rule already balances. |
| Decide on `ep_max_tokens` alone | Rejected: multicast cost scales with the total count; `ep_max` enters only as the optional imbalance term. |
| A single dispatch+combine mode bit | Rejected in favour of two: costs nothing, lets the two collectives calibrate separately, and lets batch-invariant mode pin only the combine. |

## 6. Implementation plan

### Files

- `nvls/torch_symm_triton/hybrid_v.py` (new): `_hybrid_dispatch_v_kernel` and
  `_hybrid_combine_v_kernel`, plus wrappers `hybrid_dispatch_v(..., dispatch_mode)` and
  `hybrid_combine_v(..., combine_mode)`. Wrapper assertions are the union of A2A-V's
  (EP ≤ 64, int64 routing, 16-B rows, bf16 partials).
- `nvls/metadata.py`: add an optional `policy` tensor to `fused_metadata_update`. A
  `HAS_POLICY: tl.constexpr` guard writes `step_metadata[3:5]`; existing callers are
  unchanged.
- `bench_hybrid.py` (new): `HybridBencher`, a subclass of `A2ABencher` (same five buffers
  under its own `hyb_*` keys, `step_metadata` of 5).
  - CLI (in `run.py`): `--hybrid-modes` (modes to time; one bencher, only the device policy
    changes): the policies `auto` (global κ=0), `kappa` (global κ=u) and `rank` (per rank),
    and the forced `mc_rs`, `uc_pull`, `mc_pull`, `uc_rs`. Also `--hybrid-threshold D[,C]`
    (global tokens) and `--hybrid-kappa K[,K]`.
  - `destination_probability()` computes q for the `kappa` and `rank` policies.
  - Defaults: `DEFAULT_THRESHOLDS`, per (model, EP) and per collective (§8).
  - `run_expert` always passes `write_unrouted_rows=True`.
- `run.py`: `hybrid` in `IMPLS`, the CTA/pad report and the validation set, plus
  `--phases decode_step,dispatch,combine`. Per-phase timing applies to every bencher with
  separate `dispatch` / `combine` methods (NVLS, NCCL, A2A-V, Hybrid; not DeepEP). It times
  one metadata call plus `num_layers` calls of that collective.
- `run.py` also gained `--imbalance ρ` (rank 0 holds ρ× the mean; buffers sized for it; an
  `imbalance` CSV column) and `--cta-cap N` (override the CTA-grid cap of NVLS, A2A-V and
  Hybrid). `common.py` implements the imbalanced split (`all_rank_counts`).
- `plot_results.py`: hybrid names and colours, `--phase` to plot a per-phase CSV, `--impls` to
  select lines, and the imbalance in the title.
- Megatron (separate change): a hybrid dispatcher class next to `NVLSAllGatherVDispatcher` /
  `NVLAllToAllVDispatcher`. No change to the expert kernel.

### Phases and gates

0. **Kernels with forced modes, per-phase timing.** *Done; results in §8.*
   - Validate all four combinations.
   - Time dispatch and combine separately on EP=4 (1 node) and EP=64.
   - **Gate:** if the RS combine is not faster than the pull combine at small B by a
     meaningful margin (> ~0.5 µs/layer), drop RS and make the hybrid dispatch-only.
     *Passed:* RS is 1.6–1.9 µs/layer faster at B=1 and ~1–2 µs at B=1k (EP=4), so it
     stays.
1. **Auto policy and validation.** *Mostly done in Phase 0 (`--validate`):*
   - Mode-aware A2A-V checks per forced mode, done:
     - multicast dispatch: all rows bit-exact, no NaN holes;
     - RS combine: zero rows instead of NaN on non-destination ranks, output within 1 bf16
       ulp of `bf16(m·partial)` (§3.8).
   - Policy flags against a host reference, done.
   - **In-graph switch** (one captured graph, seven policies replayed), done. This proves
     device-side switching without recapture.
   - Cross-impl check of auto (identity expert), done.
   - *Open:* auto at B = T and T+1 under the expert, and the stress validation (many
     replays) of the relaxed combine barrier.
2. **Calibration sweeps** (§4), then full sweeps. *EP=4 and EP=64 calibrated; EP=16/32 and
   `--expert vllm` open.*
   - **Target:** hybrid within about 2% of `min(AGV, A2A)` at every B, Super and Ultra,
     EP 4/16/32/64, comm only and `--expert vllm`.
   - **Report:** gain at B=1 vs A2A-V and at B=1M vs AGV-V.
3. **Imbalance (`--imbalance`)** and the κ / per-rank policies. *Done; results in §9 and
   §9.1, including the mild range and the CTA-cap sweep.*
4. **Next.**
   - Add the ρ* floor to the per-rank term (one more policy entry), validate it, and re-check
     the EP=64 mild-imbalance points.
   - Measure the single-sender multicast rate at EP=16 and EP=32, to see whether it scales as
     1/group size and whether the cost c can approach u.
   - Fit α and β per primitive and EP from the per-phase CSVs, and draft the §10 figures.
   - Then Megatron integration.

## 7. Open questions and risks

- **Calibration drift.** T depends on GPU, NVLink topology (multi-node NVL72 vs one node),
  row width and EP, and on real routing skew (which changes D). The device-tensor policy
  keeps retuning cheap, but defaults are only as good as the sweep.
- **Register pressure** of the fused combine (§3.7). It must not regress A2A-V at EP≤8 for
  Ultra; if it does, consider conditional graph nodes or splitting the pull path's live
  registers.
- **Single-sender multicast throughput at large EP** (§9.1). One GPU multicasting into a
  64-GPU group moves only about 45 GB/s, against about 670 GB/s for its unicast, whatever
  the CTA count. The per-rank rule needs a minimum imbalance ρ* so it doesn't fire on mild
  imbalance; the CTA cap stays as is. How the rate scales with group size (EP=16/32) is
  unmeasured.
- **The ρ* floor is not implemented yet.** It was evaluated offline against the measured
  points only (§9.1).
- **Serving stacks that pad DP ranks to equal counts** hide imbalance from the policy (§9.2).
- **Lowering A2A-V's small-B latency** (vectorized destination mask instead of the serial
  top-k walk; reusing dispatch's mask in combine) is orthogonal. It would shift T down and
  shrink the hybrid's small-B advantage, but not remove it: the multicast path still issues
  one operation per pack where unicast issues D.

## 8. Phase 0 results (GB200, communication only)

**Data.** CSVs are `results/phase0_{super,ultra}_ep{4,64}.csv` (gitignored; regenerate with
`run.py --impl nvls,a2a,hybrid --hybrid-modes auto,mc_rs,uc_pull,mc_pull,uc_rs --phases
decode_step,dispatch,combine`). Plots are in `plots/hybrid/{model}_ep{EP}_{phase}.png`.
Settings: 20 graph replays per point, B up to 128k (EP=4) / 64k (EP=64), `auto` with the
calibrated `DEFAULT_THRESHOLDS`.

**Validation.** `--validate` passes for Super and Ultra at EP=4 and for Super at EP=64
(64 ranks). It covers the policy flags, all four forced modes, the in-graph mode switch and
the cross-impl check.

**Headline (decode step, all MoE layers):**

| | B=1: hybrid vs A2A-V | largest B: hybrid vs AGV-V/RSV-V | hybrid / min(AGV, A2A) over all B |
|---|---|---|---|
| Super EP=4 | −26% (10.9 vs 14.7 µs/layer) | −16% (B=128k) | 0.97–1.04 (one point at 1.09, B=724) |
| Ultra EP=4 | −27% (10.8 vs 14.7) | −19% (B=128k) | 0.88–1.05 |
| Super EP=64 | −33% (11.6 vs 17.2) | −58% (B=64k; 2.4× faster) | 0.90–1.01 |
| Ultra EP=64 | −34% (12.0 vs 18.2) | −62% (B=64k; 2.7× faster) | 0.98–1.04 |

**Per-phase crossovers** (largest B where the multicast side of that collective is faster;
these are the `DEFAULT_THRESHOLDS`):

| | dispatch | combine |
|---|---|---|
| Super EP=4 | 2896 | 4096 |
| Ultra EP=4 | 1448 | 2896 (tie at 4096) |
| Super EP=64 | 1448 | 1448 |
| Ultra EP=64 | 724 | 724 (tie; 512 by a 1% margin) |

- At EP=4 the combine crossover is 1.4–2× the dispatch crossover. In between, the mixed mode
  (unicast dispatch + multicast-reduce combine) is what `auto` runs. At Super B=4096 it is
  2.4% under the better baseline (A2A-V); for Ultra the gain is within noise (≤1.5%). This
  is what the separate thresholds buy.
- At EP=64 both crossovers coincide.

**Where the small-B win comes from.** Per layer at B=1:
- multicast dispatch is 2 µs faster than unicast at EP=4 (5.8 vs 7.8 µs Super, 5.6 vs
  7.7 µs Ultra) and 4.5 µs faster at EP=64 (5.9 vs 10.4 µs Super, 6.2 vs 10.7 µs Ultra);
- the multicast-reduce combine is 1.6–1.9 µs faster than pull (e.g. 5.3 vs 6.9 µs Super
  EP=4, 6.1 vs 8.0 µs Ultra EP=64).

Both grow with EP, as the per-token D-destination work in A2A-V predicts (§2.2).

**Overhead of the fused kernels vs the stand-alone ones** (same buffers, same process,
§3.4):
- dispatch: within noise in both modes;
- combine, pull: within noise, except +0.1–0.25 µs at B=1 (the extra arguments);
- combine, RS: within noise for Super; for Ultra +4% at B=4k and +1% at large B, from the
  pull path's registers (§3.7).

**Measurement caveat (EP=64).** Multi-node numbers vary between allocations of the same
rack size:
- Two of five EP=64 runs showed the hybrid's dispatch 5–20% slower than the baselines for
  the whole second half of the run.
- Back-to-back reruns, an isolation test (hybrid alone vs with the other benchers, same
  allocation) and a same-buffer A/B probe all showed parity instead, as did the final sweeps
  above.
- A drift check (baselines re-timed at the end of the run) agreed with the start to ±2%,
  apart from single-point outliers of up to 2×.

So treat single EP=64 points as ±5% and occasionally worse. Repeated runs (min/median) are
advisable for final numbers.

**Running the multi-node sweeps.**
- The broker's jobs do not get whole nodes, so a fixed rendezvous port can already be taken.
  Rank 0 then fails with "address already in use" while the other ranks wait, and every later
  run in the job falls out of sync. This happened twice.
- The sweep scripts now have rank 0 ask the OS for a free port before each run and publish it
  in a file on the shared filesystem; the other ranks wait for it. Each run is also wrapped in
  `timeout`.
- A `grep` that matches nothing exits 1, which makes Slurm kill the whole step. Earlier
  "teardown failures" were this; write per-rank logs instead.
- Jobs use `qos=short` with 10–20 minute limits, with long sweeps split into several jobs.

**Not yet done.**
- Per-phase calibration at EP=16/32.
- `--expert vllm` sweeps, where the hybrid also pays the zero-row writes in pull mode
  (§3.5).
- ~~`--imbalance` and κ~~ (done, §9).
- The ρ* floor for the per-rank term (§9.1).
- α-β fits and the modeling figures (§10).
- The relaxed-barrier stress test.
- Megatron integration.

## 9. Imbalanced token counts: global vs per-rank policy (GB200, communication only)

**Benchmark.** `run.py --imbalance ρ` makes rank 0 hold ceil(ρ·B/EP) tokens, i.e. ρ× the mean;
the other ranks split the rest evenly. ρ = EP puts every token on rank 0. Buffers are sized
for the heavy rank.

**Policies compared.** All use the calibrated thresholds T of §8, separately for dispatch and
combine:

| `--hybrid-modes` | rule | flags |
|---|---|---|
| `auto` (global, κ=0) | every rank multicasts iff N ≤ T | same on all ranks |
| `kappa` (global, κ=u) | every rank multicasts iff N − u·n_max ≤ T·(1 − u/EP); the scaling keeps a balanced batch switching at T | same on all ranks |
| `rank` (per rank) | rank s multicasts iff N ≤ T, or u·n_s > q·N + (1−q)·S_s | may differ |

Here S_s is n_s plus the tokens of all heavier ranks (ties broken by rank index).

**The model behind the per-rank rule** (bandwidth regime, rows per GPU port):
- a unicast sender's egress is u·n_s, and each other GPU receives q·n_s from it;
- a multicast sender's egress is n_s, and every GPU receives n_s from it;
- with A = tokens of multicasting senders, the step ≈ max(u·max_unicast n_s, q·N + (1−q)·A);
- multicasting a sender lowers its egress from u·n_s to n_s but raises every GPU's ingress
  by (1−q)·n_s. So multicast the heaviest first, while u·n_s > q·N + (1−q)·S_s.

The condition is monotone in rank order, so each rank decides alone, from the count vector the
metadata kernel already has. With balanced load nobody qualifies, and `rank` equals `auto`.

**Implementation.**
- The policy tensor grows to 6 int32: `[n0_d, κ_d, n0_c, κ_c, per_rank, q (Q16)]`.
- The metadata kernel computes S_s with vector ops over the count vector, in int64.
- The dispatch and combine kernels are unchanged.
- Per-rank flags are safe because routing is always multicast and the expert always writes
  zero rows (§3.6).

**Validation** (`--validate`; passes at EP=4 and EP=64, Super and Ultra):
- flags against the host reference on balanced, imbalanced and all-on-one-rank count vectors
  (random policies, including per-rank);
- a mixed step checked row by row: rank 0 multicasts and multicast-reduces while every other
  rank unicasts and pulls.

**Results** (decode step, B above the latency threshold, where the policies can differ; ratio
of per-layer times):

| | ρ | per-rank ÷ κ=0 | per-rank ÷ κ=u | κ=u ÷ κ=0 |
|---|---|---|---|---|
| Super EP=4 | 1.5 | 0.81–0.89 | 0.97–1.02 | 0.83–0.90 |
| | 2 | 0.66–0.80 | 0.94–1.00 | 0.69–0.85 |
| | 4 | 0.56–0.70 | 1.00 | 0.56–0.71 |
| Ultra EP=4 | 1.5 | 0.84–0.91 | 0.99–1.02 | 0.83–0.90 |
| | 2 | 0.69–0.83 | 0.98–1.02 | 0.70–0.84 |
| | 4 | 0.55–0.72 | 1.00 | 0.55–0.72 |
| Super EP=64 | 4 | 0.73–0.78 | 0.68–0.92 | **0.85–1.14** |
| | 16 | 0.71–0.77 | 0.90–1.00 | 0.76–0.81 |
| | 64 | 0.73 | 1.00 | 0.73 |
| Ultra EP=64 | 4 | 0.69–0.85 | 0.63–1.00 | **0.85–1.16** |
| | 16 | 0.72–0.75 | 0.88–1.02 | 0.74–0.82 |
| | 64 | 0.73–0.74 | 1.00 | 0.73–0.74 |

With balanced load (ρ=1) all three policies set identical flags. Their differences are noise:
up to ±7% at small B at EP=64, where each mode was timed at a different moment.

Plots: `plots/hybrid_imbalance/{model}_ep{EP}_rho{ρ}.png`. CSVs:
`results/imb_{model}_ep{EP}_rho{ρ}.csv`.

**Findings.**
1. **Per-rank is the best or tied-best policy in every imbalanced configuration measured:**
   up to 1.8× faster than κ=0 at EP=4 (ρ=4) and up to 1.45× at EP=64.
2. **At EP=4, κ=u ≈ per-rank.** With q ≈ 1, multicast and unicast cost each receiver about
   the same, so multicasting the light ranks as well costs almost nothing.
3. **At EP=64, κ=u can be worse than doing nothing.** At ρ=4 it is up to 16% slower than
   κ=0, because forcing every rank to multicast makes every GPU ingest every row. Per-rank
   avoids that by multicasting only the heavy rank, and is 1.1–1.6× faster than κ=u there
   (B ≥ 2048).
4. **The bandwidth model gets the direction right but overstates the gains.**
   - EP=4, ρ ≤ 2: close (0.89 and 0.67 predicted vs 0.81–0.91 and 0.66–0.83 measured).
   - EP=4, ρ=4: optimistic (0.33 vs 0.55).
   - EP=64: far too optimistic (0.05–0.29 predicted vs 0.69–0.78 measured).
5. **Why: a single heavy sender's multicast is slow at EP=64.** Ultra,
   every token on one rank, EP=64, B=4096:
   - unicast dispatch moves u·n = 76,882 rows in 467 µs (about 670 GB/s of egress,
     link-bound as modelled);
   - multicast moves its 4,096 rows in 346 µs (about 49 GB/s);
   - at EP=4 (B=65536, ρ=4) the same single sender multicasts at 419 GB/s.

   So one multicast row from one sender costs about 14 unicast rows at EP=64 and about 1.6 at
   EP=4, where the model assumes 1. The `ld_reduce` combine shows the same factor (about 13).
   I first suspected too little concurrency (152 CTAs, one token in flight each). §9.1 rules
   that out: 8× more CTAs leave the rate unchanged. The most likely cause is per-source
   multicast replication in the fabric.
6. **Consequences.**
   - At EP=64, multicast still wins for the heavy rank because u = 18.8 > ~14, but the
     margin is thin; at EP=32 (u = 15.8) it may disappear.
   - §9.1 tests whether more concurrency fixes this: it does not.
   - Routing here is uniform. Skewed routing makes q differ per receiving rank.

### 9.1 Mild imbalance and the multicast cost (EP=64)

**Mild imbalance (decode-only range).** `--imbalance` ρ ∈ {1.1, 1.25, 1.5, 2}. Above the
threshold only rank 0 multicasts under `rank`. Per-rank ÷ global κ=0, B ≥ 2k:

| | ρ=1.1 | ρ=1.25 | ρ=1.5 | ρ=2 |
|---|---|---|---|---|
| Ultra | **1.08–1.29 (slower)** | 0.88–1.14 | 0.83–1.0 | 0.69–0.79 |
| Super | 0.92–1.04 | 0.89–1.11 | 0.84–0.91 | 0.82–0.93 |

- The rule's trigger at EP=64 is ρ > D/(D−1) = 1.06, but the measured break-even is about
  ρ ≈ 1.25–1.5.
- Most of the Ultra ρ=1.1 loss is in the combine: one owner's `ld_reduce` is up to 26% slower
  than everyone pulling.
- Global κ=u behaves like κ=0 in this range, as designed.

**What the multicast cost is.** Time the heavy rank when it holds every token (ρ=64, Ultra,
B=4096):
- unicast dispatch: 76,882 rows in 464 µs, about 670 GB/s of egress (link-bound);
- multicast dispatch: 4,096 rows in 346 µs, about 49 GB/s;
- so one multicast row from a single source costs about 14–15 unicast rows at EP=64
  (about 1.55 at EP=4, where the same sender reaches 419 GB/s).

**It is not a concurrency (CTA) limit.** `run.py --cta-cap` sweep, Ultra EP=64:

| cap (CTAs/SM) | ρ=64, B=4096: multicast / unicast dispatch (µs) | `ld_reduce` / pull combine (µs) | heavy rank's multicast rate | ρ=1, B=65536: unicast dispatch (µs) |
|---|---|---|---|---|
| 152 (1) | 346 / 464 | 310 / 428 | 49 GB/s | 164 |
| 304 (2) | 351 / 462 | 309 / 419 | 48 GB/s | 169 |
| 608 (4) | 354 / 458 | 310 / 422 | 47 GB/s | 168 |
| 1216 (8) | 358 / 460 | 313 / 429 | 47 GB/s | 186 |

- 8× more CTAs, and 8× more rows in flight (tokens per CTA from 27 down to 3.4), change
  neither the single sender's multicast nor its unicast rate.
- With balanced load, more CTAs only add barrier traffic: 13% slower unicast dispatch at
  cap 1216.

So the limit is not SM-side concurrency. The most likely explanation (not verified) is the
fabric: the NVSwitches replicating one source's multicast at a fixed total rate, about
45 GB/s × 64 receivers ≈ 2.9 TB/s delivered.
In the balanced case every GPU multicasts at once, so this per-source limit never shows.

**Recommendations.**
1. **Keep the CTA heuristic as it is** (4× SMs at EP ≤ 8, 1× SMs at EP ≥ 16). Imbalance
   doesn't change what the grid can deliver.
2. **Give the per-rank bandwidth term a minimum imbalance ρ*.** A rank multicasts by that
   term only if n_s ≥ ρ*·N/EP and the model condition holds.
   - Checked offline against all 96 measured imbalanced points, choosing between
     all-unicast and heavy-multicast. At EP=64, ρ* = 1.25–1.5 cuts the >5%-slower picks from
     11 to 7–9 of 69 and the worst case from 1.29× to 1.18× (the rest are within EP=64 noise).
   - At EP=4 the model's own trigger (1.33) already has no misses, and any ρ* ≤ 1.5 leaves it
     unchanged.
   - So a single ρ* ≈ 1.3 suits both EPs measured. It is one more policy entry.
3. **Weighting the multicast side by the measured cost** (u·n_s > max(c·n_s, …)) changes
   nothing at EP=64, because u = 18.8 > c ≈ 14.5, so the ingress term decides.
   Keep c as a guard for EPs where it could exceed u. Measuring the single-sender rate at
   EP=16/32 would tell; if it scales as 1/group size, c may approach u at EP=32.
4. **Deployment.** Decode-only (P/D) instances seldom exceed ρ ≈ 1.25, so with the floor the
   per-rank term rarely fires there. Colocated prefill+decode steps (ρ of 10+) get the full
   benefit.

### 9.2 When imbalance occurs in serving

The per-rank rule reacts only to how many tokens each EP rank owns before dispatch, i.e. the
output of each rank's attention / DP part for the step. With balanced counts it makes the same
choices as the global rule (§9: identical flags at ρ=1). It does not model expert-load skew
(popular experts concentrating work on certain receiving ranks); that would enter as a
per-rank q_r.

- **Colocated prefill + decode (no disaggregation): the main use case.** Every EP rank joins
  every MoE layer's collectives, so a rank running a prefill chunk shares the step with ranks
  running decode. Example: EP=64, one rank with a 2048-token prefill chunk and 63 ranks each
  decoding 32 tokens is ρ ≈ 32, between the measured ρ=16 and ρ=64 points, where per-rank was
  1.37–1.41× faster than global κ=0. The same mix at EP=4 is ρ ≈ 3.7 (up to 1.8× measured).
- **P/D disaggregation, decode instances: mild.** Imbalance comes from uneven active sequences
  per DP rank (arrivals, completions, KV-cache capacity with long contexts) and from B < EP
  leaving ranks idle. Much of decode is below the threshold, where every rank multicasts
  anyway. Without a floor the rule fires early, at ρ > D/(D−1); the heaviest rank then
  qualifies once u·n_s > q·N + (1−q)·n_s:

  | EP | 4 | 16 | 32 | 64 |
  |---|---|---|---|---|
  | trigger ρ = D/(D−1) | 1.33 | 1.09 | 1.07 | 1.06 |

  At EP=64 that is well below the measured break-even of ρ ≈ 1.25–1.5 (§9.1). Use the ρ*
  floor, or stay with global κ=0, on decode-only instances.
- **P/D disaggregation, prefill instances:** bandwidth-bound and possibly imbalanced when
  prompt lengths aren't packed evenly across DP ranks. The rule could matter there, but this
  hybrid is built and tuned for decode (one CTA per token, capped grid, CUDA graphs), and
  prefill usually runs a different dispatcher.
- **Padding:** if the serving stack pads every DP rank to the same token count (common for
  CUDA-graph batch sizes), the counts the collective sees are balanced by construction. The
  rule then never fires, but the padded rows still cost bandwidth.

## 10. Modeling for the paper: α-β vs per-port traffic model

**Recommendation: include both, as one model rather than as competitors.** The "bytes" model
used throughout this document (§2.1, §9) is the β half of an α-β model, with β computed from
per-link byte counts (per GPU port, egress and ingress separately) instead of a single
scalar. α-β explains **when** to switch; the per-port traffic model explains **why** unicast
wins at large batch sizes and **how** imbalance changes the choice. Each answers a question the
other can't.

### 10.1 What each model is

**α-β (Hockney).** A communication step costs T = α + β·n:
- α is the fixed per-step cost: latency, synchronization, launch, per-message overhead;
- β is the time per byte (1 / bandwidth);
- an algorithm reduces to (number of sequential steps k, bytes n on the critical path),
  giving T = k·α + n·β.

It is the standard way to compare collective algorithms, and reviewers will expect it.

**Per-port traffic (bottleneck-link) model.** From the traffic matrix, count the bytes through
every GPU port, egress and ingress separately. Time is the maximum over ports of bytes ÷ link
bandwidth. It answers "which link is the bottleneck, and how many bytes cross it", which a
single β hides.

### 10.2 What each shows in our results

**α-β: the two regimes and the crossover.**
- Multicast has the lower α: one operation per 16-B chunk, with the switch doing the fan-out
  or reduction. That costs about 6 µs per collective per layer at B=1 (§8).
- Unicast has the lower β: it moves only u/EP as many bytes per port.
- The crossover is B* = Δα / (β·r·(1 − u/EP)), where r is the row size in bytes (§2.3).
- It explains the trend with EP. The latency gap Δα grows with EP: for the dispatch, about
  2 µs at EP=4 and about 4.5 µs at EP=64, because unicast does about D peer operations per
  token (the combine gap is 1.6–1.9 µs).
  The slope advantage grows too: 1 − u/EP goes from 0.25 to 0.71. Together these predict the
  crossover moving down, from about 4k at EP=4 to about 1k at EP=64, as measured.
- **Limit:** the linear form underestimates the crossover at small EP (about 1.4k predicted
  vs 3–4k measured, Ultra EP=4). Mid-range B is neither latency- nor bandwidth-bound. That
  is why the thresholds are calibrated from measurements (§4).
- **What it can't do alone:** say why β differs between the algorithms, or say anything about
  how load is spread across ranks.

**Per-port traffic: deriving β, and imbalance.**
- It derives β for each primitive in closed form from the routing statistics (q, D, u), with
  nothing fitted (§2.1).
- It predicts the large-B unicast/multicast ratio u/EP: 0.75 / 0.72 / 0.49 / 0.29 at
  EP 4 / 16 / 32 / 64, against measured 0.80 / 0.76 / 0.46–0.55 / 0.29–0.39. That is a strong,
  falsifiable result for a paper.
- It makes the core trade explicit: multicast saves the sender's egress (1 row instead of u)
  but loads every GPU's ingress (n instead of q·n). With balanced load, unicast therefore
  always wins at large B. Under imbalance, the sender's egress becomes the bottleneck, which
  is where the per-rank rule comes from (§9).
- **Limit:** it assumes link-bound transfers, uniform routing and a non-blocking switch. It
  misses the per-source multicast rate limit (about 45 GB/s with 64 receivers, §9.1), so it
  overstates the imbalance gains at EP=64: about 19× predicted vs 1.4× measured.

### 10.3 Suggested structure for the paper's modeling section

1. **Routing statistics:** the formulas and a small table for q, D and u (§2.1).
2. **Traffic model:** per-port bytes for the four primitives (multicast dispatch, unicast
   dispatch, `ld_reduce` combine, pull combine). This derives β.
3. **Cost model:** T = α + β_port · (bytes on the busiest port). One equation that joins the
   two models. Fit α from the small-B per-phase data; take β from the link bandwidth, or fit it
   from the large-B slope.
4. **Validation:** predicted vs measured u/EP, and fitted α per algorithm vs EP.
5. **Crossover:** the B* formula, predicted vs measured for each EP and for dispatch and
   combine separately, with a short note on the mid-B gap.
6. **Imbalance:** the per-rank rule derived from per-port loads, predicted vs measured
   speedups, and the per-source multicast limit as an extra term β_mc,src(EP). Mention that
   it motivates the ρ* floor.

Useful figures:
- measured latency vs B for both transports, with the α-β fits and the crossover marked;
- predicted vs measured large-B ratio as a function of EP;
- predicted vs measured imbalance speedup as a function of ρ.

### 10.4 Things to be careful about

- **Call α an effective, fitted fixed cost** and say what it contains: launch inside a CUDA
  graph, the barrier round trip, the NVLink round trip, and for unicast the per-token top-k
  walk, which grows with D. If a reviewer asks why α depends on EP, that per-destination
  overhead is what LogGP's o and g terms model.
- **Optional refinement for mid-range B:** with one CTA per token, the middle regime has a
  "waves" term, T ≈ α + ceil(n_max / G)·t_iter, where G is the number of CTAs and t_iter the
  time per loop iteration. Using max(α + waves·t_iter, bytes/BW) fits mid-B better than
  α + n·β.
- **Report where each model fails** (the mid-B crossover at small EP, the imbalance gains at
  EP=64). Those failures justify the empirical thresholds and the floor.
- **State the uniform-routing assumption.** With skewed routing, q becomes a per-rank value.
