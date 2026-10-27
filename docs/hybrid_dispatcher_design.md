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
  computes the flags once per step from the token counts, with a mode-selection rule whose
  parameters (`mode_params`: thresholds and weights) sit in a device tensor. The global rules
  look at `valid_tokens` (and optionally `ep_max_tokens`); the local rule looks at every
  rank's count. There are no extra launches, no host sync, and nothing to recapture.
- **Neither mode needs agreement between ranks.** Routing and probs are multicast in both
  modes, and the expert only reads rows routed to its rank, which arrive either way. An
  `ld_reduce` combine needs zero rows on ranks a token was not routed to, so the expert always
  writes them (`write_unrouted_rows=True`); the pull combine ignores them (§3.5, §3.6).
- **Result (Phase 0, §8):** the hybrid tracks the envelope of the two curves on GB200 (Super
  and Ultra, EP=4 and EP=64, communication only).
  - Small B: 26–34% faster than A2A-V at B=1.
  - Large B: 16–19% faster than AGV-V/RSV-V at EP=4, and 2.4–2.7× at EP=64.
  - Crossovers are separate per collective: B≈0.7–1.4k at EP=64, and ≈1.4–4k at EP=4. At
    EP=4 the combine crosses over later than the dispatch (§2.3).
- **Imbalanced token counts (§9):** a local rule multicasts only the heaviest senders.
  - In every strongly imbalanced case measured it is the best rule: up to 1.8× faster than
    the global rule at EP=4 and 1.45× at EP=64.
  - In the mild range typical of decode-only instances (ρ ≤ 1.25 at EP=64) it is mixed or
    worse, so it needs a minimum-imbalance floor ρ* ≈ 1.3 (§9.1).
  - One GPU's multicast rate falls roughly as 1/(group size) from EP=16 up: 49 GB/s into 64
    GPUs, against 680 GB/s for its unicast. It does not depend on the CTA count or on other
    senders multicasting at the same time (§9.1).
- **Per-token split (§11):** a rank can multicast some tokens and unicast others.
  - With balanced load, the per-port model shows that no mix beats all-unicast in the
    bandwidth regime. Measured at EP=4, a split lands between the two pure modes there, and
    is at most 3–7% under the better one near the crossover.
  - For a heavy sender at EP=64, a split speeds up the combine by 19% but never the dispatch:
    multicast and unicast stores from one GPU share a bottleneck, while `ld_reduce` and pulls
    don't.
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

**Why the combine crosses over later than the dispatch at EP=4.** The per-phase crossovers
(§8) are 2896 (dispatch) vs 4096 (combine) for Super and 1448 vs 2896 for Ultra. At EP=64 they
coincide (1448 and 724). Three plausible causes can be ruled out:
- **The routing ids and probs.** Dispatch also moves 264 B/token of routing ids and probs,
  but it multicasts them in both modes. They add the same cost to both curves and cancel at
  the crossover; they only change the large-B ratio (§2.1).
- **A wider combine row.** The partials are bf16 in both combine modes. fp32 is only the
  accumulation precision, in registers (pull) or in the switch (`acc::f32`), so a combine row
  carries the same bytes as a dispatch row. Wider rows would *lower* the crossover in tokens
  (B* ∝ 1/r; Ultra's 4 KB rows cross at about half of Super's 2 KB rows), so an fp32 combine
  (RSV-V's `REDUCE_F32`, not carried over) would cross over earlier, not later.
- **The latency gap.** The combine's gap at B=1 is slightly *smaller* (1.6–1.9 µs vs
  2.0–2.1 µs per layer at EP=4), which on its own would put its crossover earlier.

What differs is the per-token cost in mid-B. Slopes between B=512 and B=4096, in ns per global
token per layer (Ultra / Super, EP=4, `results/phase0_*_ep4.csv`):

| | multicast | unicast | unicast's advantage |
|---|---|---|---|
| dispatch (`multimem.st` vs unicast stores) | 7.62 / 4.59 | 5.40 / 3.30 | 2.22 / 1.29 |
| combine (`ld_reduce` vs pull) | 6.32 / 2.91 | 5.76 / 2.62 | **0.56 / 0.29** |

- The pull combine's per-token advantage is about 4× smaller, so the combine's latency gap
  closes about 4× more slowly. Gap at B=512 divided by the slope advantage gives a crossover
  of about 2.1k (dispatch) and 4.3k (combine) for Ultra, and 2.9k and 6k for Super. Measured:
  1.4–2k and ~4k for Ultra, 2.9k and 4–5.8k for Super.
- The asymmetry is mostly on the multicast side. In mid-B, `ld_reduce` costs 1.3 (Ultra) and
  1.7 (Super) ns/token less than `multimem.st`, while pull differs from unicast dispatch by
  +0.4 and −0.7 ns/token. At large B the four primitives line up with the byte model again: unicast ÷ multicast
  = 0.82 (dispatch) and 0.80 (combine) at Ultra B=128k. Why `ld_reduce` scales better than
  `multimem.st` in mid-B has not been isolated.
- At EP=4 the two curves are nearly parallel, since unicast's large-B advantage is only
  1 − u/EP = 25%. Second-order efficiency differences between primitives therefore move the
  crossover by 2×. At EP=64 the advantage is 71%; it dominates, and the two crossovers
  coincide.

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
the factor is 3.4×. That is the global rule with `ep_max_weight = u` (`global_imbalance`,
§3.2). The local rule of §9
multicasts only the senders whose fan-out is the bottleneck.

## 3. Design

```
step start ── fused_metadata_update (once per step; 1 CTA) ──────────────────────────────┐
   multicast local count → barrier → sum / prefix / max → step_metadata[0..2]              │
   NEW: rule(counts, mode_params) → step_metadata[3] = dispatch_mc, [4] = combine_rs       │
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
  rules the flags are identical on every rank; under the local rule (§9) they may differ,
  which is safe (§3.6).
- **Why not per layer:** a routing-aware per-layer choice would need a new collective per
  layer for the ranks to agree, which costs about as much as the gap being recovered.

**Why the metadata kernel, and not an `if` in the dispatch and combine kernels.** The
communication kernels could decide for themselves. `valid_tokens` sits next to `ep_max_tokens`
in `step_metadata`, so `use_mc = valid_tokens <= THRESHOLD` would cost one more scalar load and
a compare, and every CTA on every rank would get the same answer. For the plain global
threshold, that design is equally correct and equally fast. Deciding in the metadata kernel
wins on four other points:
- **Where the threshold lives.** Inside the communication kernel, the threshold would be a
  `constexpr` (one compiled kernel per value) or a scalar argument. CUDA graph capture freezes
  scalar arguments, so changing the threshold would mean recapturing the graph, or patching
  96 kernel nodes per step (two per MoE layer). To change it without recapture it must be read
  from device memory, i.e. a parameter tensor anyway. The metadata kernel reads that tensor
  once per step instead of once per CTA per layer.
- **The local rule.** It needs every rank's count and an O(EP) reduction (the counts of all
  heavier ranks). The metadata kernel already holds the count vector in registers. In the
  communication kernels, every CTA of every layer would reload the vector and redo the
  reduction before its first store: 152 CTAs × 96 kernels per step on the latency-critical path.
- **Fragile codegen.** Small changes to the combine kernel moved its pull loop by 10–20%
  (§3.4); even making the flag load `volatile` cost 0.4 µs/layer. A single plain flag load is
  the least we can add. Rule logic that keeps growing (imbalance term, local rule, ρ* floor,
  split fractions) is safer in a kernel that runs once per step and is not timing-critical.
- **One source of truth and simple testing.** Every layer's dispatch and combine read the flags
  of the same step, the host reference (`hybrid_mode_flags`) checks one function, and forcing a
  mode in tests means writing a flag value.

The costs of this design: two extra `int32` words in `step_metadata`, a six-word parameter
tensor, and an indirection. You can only tell which mode a kernel will run by reading what the
metadata kernel wrote.

### 3.2 Mode-selection rule and `mode_params`

The kernels provide the two ways of moving a row; a small rule decides which one each step
uses. The rule's parameters are a device `int32[6]` tensor, `mode_params`, identical on all
ranks (`nvls/metadata.py`):

| index | name | meaning |
|---|---|---|
| 0 | `dispatch_threshold` | global tokens; dispatch multicasts while the (adjusted) global count is at most this |
| 1 | `dispatch_ep_max_weight` | weight of `ep_max_tokens` subtracted from the global count (Q8 fixed point) |
| 2 | `combine_threshold` | as 0, for the combine |
| 3 | `combine_ep_max_weight` | as 1, for the combine |
| 4 | `local_rule` | 0/1: also apply the local (per-rank) bandwidth rule (§9) |
| 5 | `dest_prob` | q, the probability that a given rank is one of a token's destinations (Q16) |

The global part is (int64 arithmetic):

```
dispatch_mc = valid*256 - dispatch_ep_max_weight_q8*ep_max <= dispatch_threshold*256
combine_rs  = valid*256 - combine_ep_max_weight_q8*ep_max  <= combine_threshold*256
```

- `ep_max_weight = 0` gives the plain global-token threshold `valid <= threshold`. This is
  the default (`global`) and what the balanced benchmark calibrates.
- `ep_max_weight = u` adds the imbalance term from §2.4. `HybridBencher`'s
  `global_imbalance` mode uses it, with the threshold scaled by (1 − u/EP) so a balanced batch
  switches at the same B.
- **Local term** (§9): with `local_rule` set, rank s's flags are also set if
  u·n_s > q·N + (1−q)·S_s, where u = (EP−1)·q. Flags may then differ per rank (`local`
  mode).
- **Integer math** makes the comparison bit-identical on every rank.
- **Overrides:** `threshold = ALWAYS_UNICAST (-1)` with weight 0 forces unicast/pull;
  `threshold = ALWAYS_MULTICAST (INT32_MAX)` forces multicast/`ld_reduce`.
- **Why a device tensor rather than a kernel argument:** the parameters can change between
  graph replays without recapture (§3.1). That serves tests (flip modes inside one captured
  graph), online tuning, and debugging.
- Dispatch and combine get **separate thresholds**. They are separate kernels with their own
  crossovers, and any combination is legal (§3.6).
- **Naming.** Earlier drafts called this tensor the "policy" (from the systems split between
  mechanism, here the two transports, and policy, the rule choosing between them), with
  parameters `n0` and κ. The code now uses the literal names above: `mode_params`,
  `COMPUTE_MODES` (the metadata kernel's flag that it should also write the mode flags),
  `hybrid_mode_params` / `hybrid_mode_flags` (host builder and reference), and the bencher
  modes `global`, `global_imbalance` and `local`. `global` and `local` were earlier called
  `auto` and `per_rank` (older CSVs label them `hybrid` and `hybrid-per_rank`; `plot_results.py`
  maps both). The symbols that remain (q, u, D, n_s, S_s, ρ) are the model's math, defined in
  §2.1 and §9.

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

The branch sits inside the token loop because both modes share everything around the hidden
store: the local `ld_128` of the row, the row index, and the routing/probs multicast. Measured
against a variant with the branch outside the loop (two complete token loops,
`profiles/dispatch_variants.py`), both forms time within 1% of each other and of AGV-V and A2A-V,
from B=1 to 16k, at EP=4 and EP=64.

With the `split<f>` prototype (`HAS_SPLIT`, §11), the flag also depends on the CTA:
`use_mc &= pid < mc_split`, so the first m CTAs of a rank multicast and the others unicast.

### 3.4 Hybrid combine kernel

```python
ep_max_tokens = tl.load(ep_max_tokens_ptr)
use_rs = tl.load(combine_mode_ptr) != 0                     # plain load, next to ep_max
if pid >= ep_max_tokens: return
symm_mem_sync(..., relaxed, relaxed); sync_threads()        # as both do today
rank_token_offset = tl.load(rank_token_offset_ptr)
for token_offset in range(pid, local_tokens, tl.num_programs(0)):
    global_row = (rank_token_offset + token_offset).to(tl.int64)
    if use_rs:
        _reduce_row(...)  # RSV-V's per-token body: multimem.ld_reduce (acc::f32.bf16x2) -> st
    else:
        _pull_row(...)    # A2A-V's per-token body, verbatim: D predicated peer loads, fp32 sum -> st
```

As in the dispatch, the branch is inside the token loop. Each helper is one token's column
loop, copied verbatim from the stand-alone kernel, with every peer-load address written out in
full (see the codegen lesson below).

The combine is bf16-only, as A2A-V is today; the RSV-V fp32 (`REDUCE_F32`) variant is not
carried over.

**Codegen lesson: write each peer-load address out in full.** The first fused combine
compiled to a pull loop 10–20% slower than A2A-V's at 64+ tokens per rank. Rewriting the kernel
as two token loops copied verbatim from the stand-alone kernels fixed it. That rewrite changed several things at once, so a later probe
(`profiles/combine_branch_probe.py`, EP=4, same buffers, each point timed twice in opposite
orders) varied them one at a time:

| variant | branch | peer-load address | flag load | Super pull, B = 64 / 1024 / 4096 (µs) | Ultra pull, B = 64 / 1024 / 4096 (µs) |
|---|---|---|---|---|---|
| A2A-V (reference) | — | written out | — | 7.74 / 9.90 / 17.70 | 8.17 / 12.76 / 30.50 |
| previous shipped (two token loops) | outside the token loop | written out | early | 7.75 / 9.95 / 17.74 | 8.20 / 12.77 / 30.65 |
| branch inside the token loop | per token | written out | early | 7.72 / 9.94 / 17.72 | 8.18 / 12.74 / 30.60 |
| branch inside the channel loop | per 16-B chunk | written out | early | 7.72 / 9.94 / 17.76 | 8.17 / 12.86 / 30.58 |
| shipped, flag loaded late | outside | written out | late | 7.75 / 9.93 / 17.75 | 8.20 / 12.79 / 30.55 |
| shipped, shared row offset | outside | **shared** | early | **9.03 / 11.57 / 19.33** | **9.48 / 14.05 / 32.02** |
| fix attempt 1 | outside | **shared** | late | **9.04 / 11.59 / 19.30** | **9.48 / 14.07 / 32.05** |
| first version | per 16-B chunk, one shared store | **shared** | late | **9.13 / 11.51 / 20.76** | **9.34 / 13.89 / 32.03** |

- **The only factor that matters is how the 22 peer-load addresses are written.**
  - A token's pull reads the same row from up to 22 ranks' partial buffers, at
    `peer_base + offset` with the same `offset` on every rank.
  - The slow versions computed that offset once (`row_u64 = byte_offset // 8 +
    (global_row * HIDDEN_PACKS + cols) * 2`) and loaded from `peer + row_u64`.
  - A2A-V writes the whole expression in each load: `peer + byte_offset // 8 +
    (global_row * HIDDEN_PACKS + cols) * 2`. The two are mathematically identical, but the
    shared form costs 9–17% (Super) and 4–16% (Ultra) at B = 64 and 1024–4096, and 1–2% at
    B=16k. Oddly, it costs nothing at B=256.
- **Where the branch sits makes no difference.** Outside the token loop, inside it, or
  inside the channel loop, all are within 0.5% of A2A-V in pull mode. The same holds for
  where the flag is loaded.
- **The multicast-reduce mode is unaffected** by any of these choices.
- **Mechanism (from the SASS).** In A2A-V's compiled pull loop, all 44 small loads (22
  routing ids, 22 peer base addresses) are issued first, then all 22 row loads back to back,
  then the fp32 adds. In the shared-offset version, 5 of the base-address loads are scheduled
  between the row loads. Each row load needs its base address, so the warp stalls partway
  through the gather with fewer row loads in flight. Why sharing the offset changes the
  compiler's schedule is not known; a long-lived 64-bit offset vector changing the
  scheduler's register-pressure decisions is a plausible guess.
- **The shipped combine now branches inside the token loop**, as the dispatch does (§3.3),
  with the per-token helpers `_reduce_row` / `_pull_row` holding the stand-alone kernels'
  token-loop bodies verbatim. Rechecked after the change (`profiles/combine_inloop_check.sh`):
  `--validate` passes for Super and Ultra at EP=4 and EP=64. Against the previous two-loop
  shape, A2A-V and RSV-V, every point from B=1 to 16k (EP=4) and 64k (EP=64) agrees within
  1%, except a small-B overhead that both shapes share: pull mode is 2–5% slower than A2A-V
  at B=1 and B=64 at EP=64 (about 0.15–0.4 µs/layer). Registers: 96 (Super) and 102 (Ultra),
  against 96 and 106 before.
- **Earlier side experiment:** loading the mode flag `volatile` (to stop LLVM moving it below
  the barrier's inline asm) turned it into an `LDG.E.STRONG.SYS` and cost the
  multicast-reduce path ~0.4 µs. A plain load is enough; its position is irrelevant (above).

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
     The dispatch mode can even be chosen per token (the `split<f>` prototype, §11).
   - Combine: each owner chooses for its own tokens. An RS owner sums the row from every
     rank, which is valid because the expert always writes zero rows for tokens it did not
     process (§3.5). Had the expert skipped them, as A2A-V's does, the combine mode would
     have to agree across ranks.
   - The local rule (§9) and the per-token split (§11) rely on this. `--validate` checks a
     step where rank 0 multicasts and multicast-reduces while the other ranks unicast and
     pull, and a step where half of every rank's tokens multicast and the rest unicast.
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
  - Under the global rules the multicast path only runs when B ≤ T, where at most `T/EP` CTAs
    are active, so the grid cap is set by the unicast path's large-B needs, as it is today.
    Under the local rule a heavy rank also multicasts at large B, with the full grid; the
    CTA-cap sweep (§9.1) found the single-sender multicast rate independent of the grid, so
    the cap stays.
- **Registers (measured, GB200, Triton 3.6):**

  | kernel | RSV-V combine | A2A-V combine | hybrid combine | AGV-V 3-tensor | A2A-V dispatch | hybrid dispatch |
  |---|---|---|---|---|---|---|
  | registers | 18 | 110 | 110 | 26 | 27 | 32 |

  None spill. The fused combine carries the pull path's registers into RS mode. That costs
  nothing for Super (128 threads: 4 CTAs/SM fit either way). For Ultra (256 threads) only
  2 CTAs/SM fit where RSV-V fits 8, so the 608-CTA grid at EP≤8 runs in two waves once more
  than 304 tokens per rank are active. Measured (Ultra EP=4, `combine_branch_probe.py`): RS mode
  +0.3% at B=1k, +12% at B=2k, +6% at B=4k and +2% at B=16k over RSV-V, the same for every
  branch placement. The sweeps show it too (Ultra B=1448–2048, +8–12%). Splitting the kernels
  (conditional graph nodes, §5) would remove it.
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
- **Consequence:** the combine output can change by up to 1 bf16 ulp when B crosses
  `combine_threshold`.
- **Batch-invariant mode:** pin the combine mode there (`combine_threshold = ALWAYS_UNICAST`
  for pull, which is also the deterministic one). The dispatch mode never affects numerics,
  since both modes copy rows bit-exactly, so `dispatch_threshold` can stay adaptive.

## 4. Threshold calibration

1. **Calibrate with forced modes.** Sweep the four forced combinations
   (MC|UC dispatch × RS|pull combine) at B ∈ {256, 362, 512, …, 16k} (×√2 steps) for each
   `(GPU, model row width, EP)`. Take `dispatch_threshold` where the dispatch-only difference
   changes sign, and `combine_threshold` likewise for combine. The kernels are separate, so
   the two calibrate independently.
2. **Ship a defaults table** keyed by `(row_bytes, EP)` in the bencher, measured on GB200.
   For untested shapes, fall back to scaling T with `1/(1 − u/EP)` from the nearest
   measured EP.
3. **Optional online autotune.** Since `mode_params` is a device tensor, `build()` can time
   both modes at a few B values on one captured graph and set the thresholds without
   recapture.
4. **Imbalance.** `--imbalance ρ` and the `global_imbalance` / `local` rules are measured
   in §9.

## 5. Alternatives considered

| option | verdict |
|---|---|
| Host reads `valid_tokens` and picks a kernel or graph | Rejected: needs a device-to-host sync per step, which the design exists to avoid. |
| Launch both kernels each layer, each exiting early unless it is the active mode | Rejected: an extra graph kernel node per collective per layer costs on the order of a µs of launch and drain latency, eating a large part of the 3.5–7 µs/layer small-B win. |
| CUDA-graph conditional IF/ELSE nodes set by the metadata kernel (`cudaGraphSetConditional`) | Deferred: would allow per-mode grids and registers, but needs a CUDA-C++ setter and PyTorch conditional-node capture. Worth it only if §3.7 register pressure shows up. |
| Per-rank multicast vs unicast | Implemented as the `local` rule (§9): only the heaviest senders multicast, when their unicast fan-out is the bottleneck. |
| Decide inside the dispatch / combine kernels (`if valid_tokens <= T`) | Equivalent for the plain global threshold, but a retunable threshold needs a device tensor anyway, the local rule would be recomputed by every CTA of every layer, and the combine's codegen is fragile. Kept in the metadata kernel (§3.1). |
| Per-token multicast vs unicast | Prototyped (`split<f>`, §11). With balanced load no mix beats all-unicast in the bandwidth regime. For a heavy sender, splitting helps the combine (−19% at EP=64, ρ=64) but not the dispatch. |
| Decide on `ep_max_tokens` alone | Rejected: multicast cost scales with the total count; `ep_max` enters only as the optional imbalance term. |
| A single dispatch+combine mode bit | Rejected in favour of two: costs nothing, lets the two collectives calibrate separately, and lets batch-invariant mode pin only the combine. |

## 6. Implementation plan

### Files

- `nvls/torch_symm_triton/hybrid_v.py` (new): `_hybrid_dispatch_v_kernel` and
  `_hybrid_combine_v_kernel`, plus wrappers `hybrid_dispatch_v(..., dispatch_mode)` and
  `hybrid_combine_v(..., combine_mode)`. Wrapper assertions are the union of A2A-V's
  (EP ≤ 64, int64 routing, 16-B rows, bf16 partials).
- `nvls/metadata.py`: add an optional `mode_params` tensor to `fused_metadata_update`. A
  `COMPUTE_MODES: tl.constexpr` guard writes `step_metadata[3:5]`; existing callers are
  unchanged. `hybrid_mode_params` builds the tensor's values and `hybrid_mode_flags` is the
  host reference.
- `bench_hybrid.py` (new): `HybridBencher`, a subclass of `A2ABencher` (same five buffers
  under its own `hyb_*` keys, `step_metadata` of 5).
  - CLI (in `run.py`): `--hybrid-modes` (modes to time; one bencher, only the device
    `mode_params` change): the rules `global` (one threshold on the global token count),
    `global_imbalance` (global,
    `ep_max_weight = u`) and `local`; the forced `mc_rs`, `uc_pull`, `mc_pull`, `uc_rs`;
    and the prototype `split<f>` (§11). Also `--hybrid-threshold D[,C]` (global tokens) and
    `--hybrid-ep-max-weight W[,W]`.
  - `destination_probability()` computes q for the `global_imbalance` and `local` rules.
  - Defaults: `DEFAULT_THRESHOLDS`, per (model, EP) and per collective (§8).
  - `run_expert` always passes `write_unrouted_rows=True`.
- `run.py`: `hybrid` in `IMPLS`, the CTA/pad report and the validation set, plus
  `--phases decode_step,dispatch,combine`. Per-phase timing applies to every bencher with
  separate `dispatch` / `combine` methods (NVLS, NCCL, A2A-V, Hybrid; not DeepEP). It times
  one metadata call plus `num_layers` calls of that collective.
- `run.py` also gained `--imbalance ρ` (rank 0 holds ρ× the mean; buffers sized for it; an
  `imbalance` CSV column), `--heavy-ranks H` (ranks 0..H−1 each hold ρ× the mean) and
  `--cta-cap N` (override the CTA-grid cap of NVLS, A2A-V and Hybrid). `common.py`
  implements the imbalanced split (`all_rank_counts`).
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
1. **Global rule and validation.** *Mostly done in Phase 0 (`--validate`):*
   - Mode-aware A2A-V checks per forced mode, done:
     - multicast dispatch: all rows bit-exact, no NaN holes;
     - RS combine: zero rows instead of NaN on non-destination ranks, output within 1 bf16
       ulp of `bf16(m·partial)` (§3.8).
   - Mode flags against a host reference, done.
   - **In-graph switch** (one captured graph, seven `mode_params` settings replayed), done.
     This proves device-side switching without recapture.
   - Cross-impl check of `global` (identity expert), done.
   - *Open:* `global` at B = T and T+1 under the expert, and the stress validation (many
     replays) of the relaxed combine barrier.
2. **Calibration sweeps** (§4), then full sweeps. *EP=4 and EP=64 calibrated; EP=16/32 and
   `--expert vllm` open.*
   - **Target:** hybrid within about 2% of `min(AGV, A2A)` at every B, Super and Ultra,
     EP 4/16/32/64, comm only and `--expert vllm`.
   - **Report:** gain at B=1 vs A2A-V and at B=1M vs AGV-V.
3. **Imbalance (`--imbalance`)** and the `global_imbalance` / `local` rules. *Done;
   results in §9 and §9.1, including the mild range, the CTA-cap sweep and the
   single-sender rate at EP 4–64.*
4. **Per-token split (`split<f>`).** *Prototype done; results in §11.*
5. **Next.**
   - Add the ρ* floor to the local term (one more `mode_params` entry), validate it, and
     re-check the EP=64 mild-imbalance points.
   - Give heavy ranks a combine split (§11.4): one more output of the local rule.
   - Fit α and β per primitive and EP from the per-phase CSVs, and draft the §10 figures.
   - Then Megatron integration.

## 7. Open questions and risks

- **Calibration drift.** T depends on GPU, NVLink topology (multi-node NVL72 vs one node),
  row width and EP, and on real routing skew (which changes D). The device-tensor
  `mode_params` keep retuning cheap, but defaults are only as good as the sweep.
- **Register pressure** of the fused combine (§3.7). It must not regress A2A-V at EP≤8 for
  Ultra; if it does, consider conditional graph nodes or splitting the pull path's live
  registers.
- **Single-sender multicast throughput at large EP** (§9.1). One GPU's multicast rate falls
  roughly as 1/(group size) from EP=16 up: 49 GB/s into 64 GPUs, against 680 GB/s for its
  unicast, whatever the CTA count. So the local rule needs a minimum imbalance ρ*, so that
  it doesn't fire on mild imbalance, and the CTA cap stays as is. The physical cause is not
  verified.
- **The ρ* floor is not implemented yet.** It was evaluated offline against the measured
  points only (§9.1).
- **Serving stacks that pad DP ranks to equal counts** hide imbalance from the rule (§9.2).
- **Lowering A2A-V's small-B latency** (vectorized destination mask instead of the serial
  top-k walk; reusing dispatch's mask in combine) is orthogonal. It would shift T down and
  shrink the hybrid's small-B advantage, but not remove it: the multicast path still issues
  one operation per pack where unicast issues D.

## 8. Phase 0 results (GB200, communication only)

**Data.** CSVs are `results/phase0_{super,ultra}_ep{4,64}.csv` (gitignored; regenerate with
`run.py --impl nvls,a2a,hybrid --hybrid-modes global,mc_rs,uc_pull,mc_pull,uc_rs --phases
decode_step,dispatch,combine`). Plots are in `plots/hybrid/{model}_ep{EP}_{phase}.png`.
Settings: 20 graph replays per point, B up to 128k (EP=4) / 64k (EP=64), `global` with the
calibrated `DEFAULT_THRESHOLDS`.

**Validation.** `--validate` passes for Super and Ultra at EP=4 and for Super at EP=64
(64 ranks). It covers the mode flags, all four forced modes, the in-graph mode switch and
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
  (unicast dispatch + multicast-reduce combine) is what `global` runs. At Super B=4096 it is
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
- ~~`--imbalance` and the imbalance rules~~ (done, §9).
- The ρ* floor for the local term (§9.1).
- A combine split for heavy ranks (§11.4).
- α-β fits and the modeling figures (§10).
- The relaxed-barrier stress test.
- Megatron integration.

## 9. Imbalanced token counts: global vs local rule (GB200, communication only)

**Benchmark.** `run.py --imbalance ρ` makes rank 0 hold ceil(ρ·B/EP) tokens, i.e. ρ× the mean;
the other ranks split the rest evenly. ρ = EP puts every token on rank 0. Buffers are sized
for the heavy rank. `--heavy-ranks H` makes ranks 0..H−1 each that heavy.

**Rules compared.** All use the calibrated thresholds T of §8, separately for dispatch and
combine:

| `--hybrid-modes` | rule | flags |
|---|---|---|
| `global` (`ep_max_weight = 0`) | every rank multicasts iff N ≤ T | same on all ranks |
| `global_imbalance` (global, `ep_max_weight = u`) | every rank multicasts iff N − u·n_max ≤ T·(1 − u/EP); the scaling keeps a balanced batch switching at T | same on all ranks |
| `local` | rank s multicasts iff N ≤ T, or u·n_s > q·N + (1−q)·S_s | may differ |

Here S_s is n_s plus the tokens of all heavier ranks (ties broken by rank index).

**The model behind the local rule** (bandwidth regime, rows per GPU port):
- a unicast sender's egress is u·n_s, and each other GPU receives q·n_s from it;
- a multicast sender's egress is n_s, and every GPU receives n_s from it;
- with A = tokens of multicasting senders, the step ≈ max(u·max_unicast n_s, q·N + (1−q)·A);
- multicasting a sender lowers its egress from u·n_s to n_s but raises every GPU's ingress
  by (1−q)·n_s. So multicast the heaviest first, while u·n_s > q·N + (1−q)·S_s.

The condition is monotone in rank order, so each rank decides alone, from the count vector the
metadata kernel already has. With balanced load nobody qualifies, and `local` equals
`global`.

**Implementation.**
- `mode_params` grows to 6 int32 with `local_rule` and `dest_prob` (q, Q16) (§3.2).
- The metadata kernel computes S_s with vector ops over the count vector, in int64.
- The dispatch and combine kernels are unchanged.
- Per-rank flags are safe because routing is always multicast and the expert always writes
  zero rows (§3.6).

**Validation** (`--validate`; passes at EP=4 and EP=64, Super and Ultra):
- flags against the host reference on balanced, imbalanced and all-on-one-rank count vectors
  (random `mode_params`, including the local rule);
- a mixed step checked row by row: rank 0 multicasts and multicast-reduces while every other
  rank unicasts and pulls.

**Results** (decode step, B above the latency threshold, where the rules can differ; ratio
of per-layer times):

| | ρ | `local` ÷ `global` | `local` ÷ `global_imbalance` | `global_imbalance` ÷ `global` |
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

With balanced load (ρ=1) all three rules set identical flags. Their differences are noise:
up to ±7% at small B at EP=64, where each mode was timed at a different moment.

Plots: `plots/hybrid_imbalance/{model}_ep{EP}_rho{ρ}.png`. CSVs:
`results/imb_{model}_ep{EP}_rho{ρ}.csv`.

**Findings.**
1. **`local` is the best or tied-best rule in every imbalanced configuration measured:**
   up to 1.8× faster than `global` at EP=4 (ρ=4) and up to 1.45× at EP=64.
2. **At EP=4, `global_imbalance` ≈ `local`.** With q ≈ 1, multicast and unicast cost each
   receiver about the same, so multicasting the light ranks as well costs almost nothing.
3. **At EP=64, `global_imbalance` can be worse than doing nothing.** At ρ=4 it is up to 16%
   slower than `global`, because forcing every rank to multicast makes every GPU ingest every
   row. `local` avoids that by multicasting only the heavy rank, and is 1.1–1.6× faster
   than `global_imbalance` there (B ≥ 2048).
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
   that out: 8× more CTAs leave the rate unchanged. The rate falls roughly as 1/(group size)
   and is a per-source limit (§9.1).
6. **Consequences.**
   - At EP=64, multicast still wins for the heavy rank because u = 18.8 > c ≈ 14, but the
     margin is thin. At EP=16 and 32 it is wider (§9.1: c = 4.0 and 7.2 against u = 11.5 and
     15.8).
   - Routing here is uniform. Skewed routing makes q differ per receiving rank.

### 9.1 Mild imbalance and the multicast cost (EP=64)

**Mild imbalance (decode-only range).** `--imbalance` ρ ∈ {1.1, 1.25, 1.5, 2}. Above the
threshold only rank 0 multicasts under `local`. `local` ÷ `global`, B ≥ 2k:

| | ρ=1.1 | ρ=1.25 | ρ=1.5 | ρ=2 |
|---|---|---|---|---|
| Ultra | **1.08–1.29 (slower)** | 0.88–1.14 | 0.83–1.0 | 0.69–0.79 |
| Super | 0.92–1.04 | 0.89–1.11 | 0.84–0.91 | 0.82–0.93 |

- The rule's trigger at EP=64 is ρ > D/(D−1) = 1.06, but the measured break-even is about
  ρ ≈ 1.25–1.5.
- Most of the Ultra ρ=1.1 loss is in the combine: one owner's `ld_reduce` is up to 26% slower
  than everyone pulling.
- `global_imbalance` behaves like `global` in this range, as designed.

**What the multicast cost is: a per-source rate that falls with group size.** Every token on
rank 0, Ultra, B=4096, per layer (`results/single_ultra_ep{8,16,32}.csv`,
`imb_ultra_ep4_rho4.csv`, `cta_ultra_ep64_rho64_cap152.csv`):

| EP (group size G) | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|
| multicast dispatch (µs) | 47.6 | 58.6 | 98.0 | 179.4 | 345.5 |
| multicast rate (GB/s) | 353 | 286 | 171 | 94 | 49 |
| rate × G (TB/s of copies delivered) | 1.4 | 2.3 | 2.7 | 3.0 | 3.1 |
| unicast dispatch egress (GB/s) | 578 | 655 | 678 | 674 | 679 |
| c = one multicast row in unicast-row times | 1.6 | 2.3 | 4.0 | 7.2 | 14.0 |
| u (remote destinations per token) | 3.00 | 6.65 | 11.48 | 15.81 | 18.77 |
| `ld_reduce` combine rate (GB/s) | 363 | 323 | 186 | 103 | 54 |

- From G=16 up the rate is about 3 TB/s ÷ G: the copies delivered per second stay constant,
  so a multicast row costs about 0.22·G unicast rows.
- c < u at every EP measured, so multicasting still wins for a sender that holds every token,
  but by a shrinking factor: u/c = 1.9, 2.9, 2.9, 2.2, 1.3.
- The `ld_reduce` combine follows the same curve.

**It is not a concurrency (CTA) limit.** `run.py --cta-cap` sweep, Ultra EP=64:

| cap (CTAs/SM) | ρ=64, B=4096: multicast / unicast dispatch (µs) | `ld_reduce` / pull combine (µs) | heavy rank's multicast rate | ρ=1, B=65536: unicast dispatch (µs) |
|---|---|---|---|---|
| 38 (¼) | 344 / 495 | 343 / 464 | 49 GB/s | — |
| 76 (½) | 345 / 482 | 317 / 425 | 49 GB/s | — |
| 114 (¾) | 345 / 474 | 314 / 431 | 49 GB/s | — |
| 152 (1) | 346 / 464 | 310 / 428 | 49 GB/s | 164 |
| 304 (2) | 351 / 462 | 309 / 419 | 48 GB/s | 169 |
| 608 (4) | 354 / 458 | 310 / 422 | 47 GB/s | 168 |
| 1216 (8) | 358 / 460 | 313 / 429 | 47 GB/s | 186 |

- From 38 to 1216 CTAs (tokens per CTA from 108 down to 3.4), the single sender's multicast
  dispatch time does not move. Its unicast improves 7% from 38 to 152 CTAs and is flat
  above.
- With balanced load, more CTAs only add barrier traffic: 13% slower unicast dispatch at
  cap 1216.

**It is per source.** Two heavy ranks, each holding 4096 tokens (B=8192, `--imbalance 32
--heavy-ranks 2`, `results/two_senders_ultra_ep64.csv`): multicast dispatch 346.1 µs against
345.5 µs with one heavy rank, `ld_reduce` 310.6 against 309.3, unicast 462.2 against 462.2.
Each sender keeps its own rate, so the limit is not a capacity the switches share between
sources.

**It shares a resource with the same GPU's unicast stores, but not with its pull loads**
(§11.2). Splitting a heavy sender's tokens between `multimem.st` and unicast stores costs
the time-weighted average of the two. Splitting its combine between `ld_reduce` and pulls
overlaps the two almost fully.

**Interpretation (consistent with all of the above, not verified).** The switch replicates or
reduces one source's multicast traffic at a fixed number of copies per second, about 3 TB/s
in total. For stores, the waiting packets back up into the source's own NVLink injection,
which its unicast stores also use. For `ld_reduce`, the requests are small, so the source's
pull loads still get through. In the balanced case every GPU multicasts at once, so each
needs only 1/EP of that rate and the limit never shows.

**Recommendations.**
1. **Keep the CTA heuristic as it is** (4× SMs at EP ≤ 8, 1× SMs at EP ≥ 16). Imbalance
   doesn't change what the grid can deliver.
2. **Give the local bandwidth term a minimum imbalance ρ*.** A rank multicasts by that
   term only if n_s ≥ ρ*·N/EP and the model condition holds.
   - Checked offline against all 96 measured imbalanced points, choosing between
     all-unicast and heavy-multicast. At EP=64, ρ* = 1.25–1.5 cuts the >5%-slower picks from
     11 to 7–9 of 69 and the worst case from 1.29× to 1.18× (the rest are within EP=64 noise).
   - At EP=4 the model's own trigger (1.33) already has no misses, and any ρ* ≤ 1.5 leaves it
     unchanged.
   - So a single ρ* ≈ 1.3 suits both EPs measured. It is one more `mode_params` entry.
3. **Weighting the multicast side by the measured cost** (u·n_s > max(c·n_s, …)) changes
   nothing at the EPs measured: c < u everywhere (table above), so the ingress term decides.
   c approaches u at EP=64 (14.0 vs 18.8), so keep c as a guard for larger groups or other
   fabrics.
4. **Deployment.** Decode-only (P/D) instances seldom exceed ρ ≈ 1.25, so with the floor the
   local term rarely fires there. Colocated prefill+decode steps (ρ of 10+) get the full
   benefit.

### 9.1a All methods under imbalance (rerun with the renamed rules)

`profiles/all_methods_sweep.sh` reran the imbalance sweep with all five methods in one process
per point: AGV-V/RSV-V, A2A-V, DeepEP (`EPBuffer`, `--deepep-num-sms` = the device's 152 SMs),
and the hybrid's `global` and `local` rules. These runs used the in-loop combine (§3.4).
Decode step, communication only, B ≥ 2048 (above the thresholds, where the rules can differ).
CSVs: `results/imball_{model}_ep{EP}_rho{ρ}.csv`; plots: `plots/hybrid_imbalance/`.

| | ρ | `local` ÷ `global` | `local` ÷ best of the three baselines |
|---|---|---|---|
| Super EP=4 | 1 / 1.5 / 2 / 4 | 1.00 / 0.82–1.00 / 0.65–0.98 / 0.55–1.00 | 0.98–1.00 / 0.98–1.02 / 0.94–1.02 / 1.01–1.05 |
| Ultra EP=4 | 1 / 1.5 / 2 / 4 | 1.00 / 0.84–0.90 / 0.70–0.84 / 0.54–0.72 | 0.99–1.03 / 1.01–1.07 / 0.95–1.02 / 1.00–1.08 |
| Super EP=64 | 1 / 1.1 / 1.25 / 1.5 | 1.00 / 1.00–1.06 / 0.95–1.00 / 0.88–1.00 | 0.93–0.99 / 0.99–1.04 / 0.91–1.03 / 0.86–1.01 |
| | 2 / 4 / 16 / 64 | 0.80–0.99 / 0.70–0.77 / 0.74–0.87 / 0.75 | 0.78–1.00 / 0.71–0.81 / 0.90–1.06 / 0.99 |
| Ultra EP=64 | 1 / 1.1 / 1.25 / 1.5 | 1.00 / 1.00–**1.29** / 0.90–**1.15** / 0.85–1.01 | 0.99–1.02 / 0.98–1.27 / 0.87–1.16 / 0.79–1.04 |
| | 2 / 4 / 16 / 64 | 0.70–0.83 / 0.67–0.74 / 0.71–0.72 / 0.72–0.73 | 0.71–0.82 / 0.70–0.78 / 0.88–0.98 / 0.99–1.00 |

- **Balanced (ρ=1): `local` = `global`**, as designed (identical flags, times within 1%).
- **Strong imbalance: `local` is the fastest method at EP=64.** At ρ = 2–4 it takes
  0.67–0.83× the time of `global`, and up to 1.4× less than the best baseline (0.70–0.82 for
  Ultra at ρ = 2–4 and Super at ρ=4). At ρ = 16–64 the best baseline is AGV-V/RSV-V
  (all-multicast); `local` is within −12% to +6% of it.
- **At EP=4, `local` tracks the best baseline** (usually AGV-V/RSV-V under imbalance) within
  −6% to +8%: with q ≈ 1, multicasting everyone costs little, so there is little left to gain.
- **Mild imbalance at EP=64 is still the weak spot**, as in §9.1: Ultra ρ=1.1 at B=16k–64k is
  1.29× slower than `global`, and ρ=1.25 at 64k 1.15×. The ρ* floor would remove these.
- **DeepEP at 152 SMs** is the best baseline at B=64k for ρ = 1.25–4 at EP=64, and the slowest
  method at small B (2–3 ms per step at B=256).

### 9.2 When imbalance occurs in serving

The local rule reacts only to how many tokens each EP rank owns before dispatch, i.e. the
output of each rank's attention / DP part for the step. With balanced counts it makes the same
choices as the global rule (§9: identical flags at ρ=1). It does not model expert-load skew
(popular experts concentrating work on certain receiving ranks); that would enter as a
per-rank q_r.

- **Colocated prefill + decode (no disaggregation): the main use case.** Every EP rank joins
  every MoE layer's collectives, so a rank running a prefill chunk shares the step with ranks
  running decode. Example: EP=64, one rank with a 2048-token prefill chunk and 63 ranks each
  decoding 32 tokens is ρ ≈ 32, between the measured ρ=16 and ρ=64 points, where `local` was
  1.37–1.41× faster than `global`. The same mix at EP=4 is ρ ≈ 3.7 (up to 1.8× measured).
- **P/D disaggregation, decode instances: mild.** Imbalance comes from uneven active sequences
  per DP rank (arrivals, completions, KV-cache capacity with long contexts) and from B < EP
  leaving ranks idle. Much of decode is below the threshold, where every rank multicasts
  anyway. Without a floor the rule fires early, at ρ > D/(D−1); the heaviest rank then
  qualifies once u·n_s > q·N + (1−q)·n_s:

  | EP | 4 | 16 | 32 | 64 |
  |---|---|---|---|---|
  | trigger ρ = D/(D−1) | 1.33 | 1.09 | 1.07 | 1.06 |

  At EP=64 that is well below the measured break-even of ρ ≈ 1.25–1.5 (§9.1). Use the ρ*
  floor, or stay with `global`, on decode-only instances.
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
  always wins at large B, and no per-token mix of the two can beat it (§11.1). Under
  imbalance, the sender's egress becomes the bottleneck, which is where the local rule
  comes from (§9).
- **Limit:** it assumes link-bound transfers, uniform routing and a non-blocking switch. It
  misses the per-source multicast rate limit (one multicast row costs c ≈ 0.22·G unicast rows
  from G=16 up, §9.1), so it overstates the imbalance gains at EP=64: about 19× predicted vs
  1.4× measured. Adding c as the sender-side cost of a multicast row fixes the direction of
  the error, not its size (§11.2).

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
6. **Imbalance:** the local rule derived from per-port loads, predicted vs measured
   speedups, and the per-source multicast limit as an extra sender-side cost c(G) per
   multicast row (measured at G = 4–64, §9.1). Mention that it motivates the ρ* floor.
7. **Per-token mixing:** the balanced-load result that all-unicast is optimal in the
   bandwidth regime (§11.1), and the shared vs independent resources of the heavy-sender
   split (§11.2).

Useful figures:
- measured latency vs B for both transports, with the α-β fits and the crossover marked;
- predicted vs measured large-B ratio as a function of EP;
- predicted vs measured imbalance speedup as a function of ρ;
- single-sender multicast rate vs group size, with the 1/G line (§9.1);
- the split sweep against the "shared" and "independent" predictions (§11.2).

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

## 11. Per-token multicast / unicast split

The global rules pick one mode per step and the local rule one mode per rank. Any mixture is
legal (§3.6), so a rank can also send some of its tokens by multicast and the rest by unicast.
This section asks when that helps.

### 11.1 Model

Per-port traffic model (§2.1, §9), in rows. Token t on sender s has d_t remote destination
ranks (E[d_t] = u). If x_t = 1 it is multicast, if x_t = 0 unicast:
- **sender egress:** x_t·1 + (1 − x_t)·d_t;
- **ingress:** multicast adds one row to every GPU, EP in total, the sender's own loopback
  included; unicast adds one row to each of the d_t remote destinations, and the sender's own
  copy (if any) is a local HBM write.

The step takes as long as the busiest port, egress or ingress (each direction has its own
bandwidth). Multicasting token t moves d_t − 1 rows off its sender's egress and adds
EP − d_t rows of ingress, spread over all GPUs.

**Balanced load: all-unicast is optimal in the bandwidth regime.** Summed over all GPUs:
- total ingress = Σ_t [x_t·EP + (1 − x_t)·d_t];
- total egress = Σ_t [x_t + (1 − x_t)·d_t].

Since d_t < EP, total ingress ≥ total egress, term by term. The busiest port therefore carries
at least total ingress / EP ≥ Σ_t d_t / EP, with equality only if every x_t = 0. With balanced
counts and uniform routing, all-unicast reaches this bound: every GPU's egress and ingress
both equal Σ_t d_t / EP. Every multicast token raises total ingress by EP − d_t ≥ 1, so it
raises the bound. The egress it saves was not the binding side, because under balanced
unicast egress equals ingress.

Put differently: multicast does not remove copies. It moves the copying into the switch and
makes more of them (EP instead of d_t), and every copy still has to enter some GPU. When
every GPU is equally loaded as sender and receiver, there is no idle egress for multicast to
use. Trading egress for ingress only pays when some sender's egress is busier than the
average ingress, i.e. under imbalance (§9). This argument covers the bandwidth regime only;
§11.3 measures the balanced case near the crossover, where the ports are not saturated.

**Choosing tokens by their destination count.** When a sender's egress is the bottleneck,
picking which of its tokens to multicast is a knapsack problem. A token's egress saving
(d_t − 1) and its ingress price (EP − d_t) both favour a large d_t, so multicast the tokens
with the most remote destinations first. With the per-source multicast cost c of §9.1, and
since the dispatch's multicast and unicast stores share the sender's resource (§11.2), the
sender-side saving is d_t − c rather than d_t − 1: multicasting a token only pays if
d_t > c.

How much room does the spread of d leave? With uniform routing (top-22 of 512 experts,
simulated, 200k tokens):

| EP | u = mean d | std of d | 1st–99th percentile of d | c (§9.1) |
|---|---|---|---|---|
| 4 | 3.00 | 0.07 | 3–3 | 1.6 |
| 8 | 6.65 | 0.54 | 5–7 | 2.3 |
| 16 | 11.48 | 1.25 | 9–14 | 4.0 |
| 32 | 15.81 | 1.57 | 12–19 | 7.2 |
| 64 | 18.77 | 1.44 | 15–22 | 14.0 |

- At every EP, even the 1st-percentile token has d > c. For a sender whose egress is the
  bottleneck, ordering by d therefore selects "multicast everything", the same answer as the
  local rule.
- d varies by only about ±8% around u, so ordering by d buys little over a per-rank decision.
- Real routers may differ. For example, group-limited routing puts a token's experts on few
  ranks and lowers d. The d distribution is cheap to measure from router outputs.

### 11.2 Heavy sender: the `split<f>` sweep (EP=64, ρ=64)

**Prototype.** Mode `split<f>` (`--hybrid-modes split0.5`, etc.) forces both flags on. Only CTAs
with pid < m = round(f · active CTAs) multicast or `ld_reduce`; the other CTAs' tokens are
unicast or pulled (`mc_split`, `HAS_SPLIT` in `hybrid_v.py`). Tokens go to CTAs round-robin, so
about a fraction f of each rank's tokens are multicast. `--validate` checks a split0.5 step row
by row.

**Results.** Every token on rank 0, Ultra, B=4096, per layer
(`results/split_ultra_ep64_rho64.csv`):

| f | dispatch (µs) | combine (µs) |
|---|---|---|
| 0 (all unicast / pull) | 462.2 | 430.4 |
| 0.25 | 425.3 | 347.1 |
| 0.40 | 408.5 | 291.8 |
| 0.50 | 400.1 | 259.4 |
| 0.57 | 391.7 | **250.4** |
| 0.65 | 387.1 | 257.7 |
| 0.80 | 377.7 | 277.7 |
| 1 (all multicast / `ld_reduce`) | **345.5** | 309.3 |

Two predictions bracket the result:
- **shared:** both halves consume one resource, so T(f) = f·T_MC + (1 − f)·T_UC;
- **independent:** the halves run in parallel, so T(f) = max(f·T_MC, (1 − f)·T_UC), each
  measured at its own CTA count (the reduced-cap controls of §9.1). This is lowest at
  f* = T_UC / (T_MC + T_UC).

**Findings.**
- **The dispatch follows "shared"**, within 2.4% at every f (f=0.5: 400.1 µs measured,
  403.9 shared, 241.0 independent), and never beats all-multicast. One sender's
  `multimem.st` and unicast stores compete for the same resource.
- **The combine comes close to "independent".** Its best point, f=0.57, takes 250.4 µs: 19%
  under all-`ld_reduce` (309.3) and 42% under all-pull (430.4). Predicted
  f* = 430.4 / (309.3 + 430.4) = 0.58. The independent bound near f* is 187–213 µs, so the
  overlap is good but not perfect.
- In per-row costs (pull: u rows at link rate; `ld_reduce`: c), the optimum is
  f* = u / (u + c) = 18.77 / 32.77 = 0.57.
- So `ld_reduce` and pull are limited by different resources, while `multimem.st` and unicast
  stores share one (interpretation in §9.1).

### 11.3 Balanced load near the crossover (EP=4)

Balanced load, `split0.25/0.5/0.75` against the two pure modes, per layer
(`results/bsplit_{ultra,super}_ep4.csv`, 20 replays per point):

| | B | f=0 (unicast / pull) | 0.25 | 0.5 | 0.75 | f=1 (multicast / `ld_reduce`) | best split ÷ better pure mode |
|---|---|---|---|---|---|---|---|
| Ultra dispatch | 1024 | 17.61 | 17.09 | 16.65 | 17.25 | 16.09 | 1.035 |
| | 2048 | 23.18 | **21.19** | 22.13 | 22.68 | 22.78 | **0.930** |
| | 4096 | 32.49 | 32.55 | 34.63 | 36.50 | 36.25 | 1.002 |
| | 16384 | 92.90 | 97.82 | 104.32 | 112.66 | 114.78 | 1.053 |
| Ultra combine | 1024 | 12.80 | 11.56 | 11.25 | 11.34 | 11.29 | 0.997 |
| | 2048 | 21.30 | 19.30 | 19.94 | 20.07 | 18.88 | 1.023 |
| | 4096 | 30.51 | 29.57 | 30.88 | 30.83 | 30.52 | 0.969 |
| | 16384 | 88.83 | 91.30 | 96.24 | 100.82 | 102.98 | 1.028 |
| Super dispatch | 2048 | 17.27 | 16.70 | 16.59 | 16.79 | 16.01 | 1.036 |
| | 4096 | 22.54 | 23.10 | 24.00 | 24.17 | 24.34 | 1.025 |
| | 8192 | 34.26 | 35.46 | 36.57 | 37.65 | 38.70 | 1.035 |
| | 32768 | 104.59 | 111.78 | 118.60 | 127.68 | 124.08 | 1.069 |
| Super combine | 2048 | 12.90 | 11.27 | 11.41 | 11.24 | 11.10 | 1.012 |
| | 4096 | 17.87 | 17.71 | 17.58 | 17.30 | 17.02 | 1.016 |
| | 8192 | 28.44 | 28.61 | **27.38** | 29.45 | 30.43 | **0.963** |
| | 32768 | 86.61 | 90.97 | 95.01 | 100.36 | 104.22 | 1.050 |

- **Bandwidth regime (Ultra 16k, Super 32k): as §11.1 predicts.** The split lies on or
  slightly above the straight line between the pure modes (Ultra dispatch f=0.5: 104.3 µs
  measured, 103.8 for the average of the two). It is never below all-unicast.
- **Near the crossover:** 3 of 12 points are 3–7% under the better pure mode, the largest
  being Ultra dispatch at B=2048 with f=0.25 (−7%). The other points are at or above it. Since
  single-node points move by about ±2% between runs, only the −7% point is clearly outside
  noise, and it needs a repeat before it counts.
- **Conclusion for balanced load:** no gain in the bandwidth regime, and at most a few
  percent in a narrow band around the crossover. That is not worth a third threshold until a
  repeat confirms it.

### 11.4 What to build

- **Dispatch:** keep the all-or-nothing decision per rank. For a heavy sender a split never
  beat all-multicast. With balanced load, the one point where it helped (−7% near the
  crossover, §11.3) is unconfirmed.
- **Combine:** for a rank that the local rule multicasts, split its combine with
  f* = u / (u + c(EP)). This needs one more output from the metadata kernel (the rank's
  multicast-reduce CTA count, e.g. `step_metadata[5]`) and c per EP, either in `mode_params`
  or in a table. Expected gain: about 19% on the heavy rank's combine at EP=64, ρ=64. Milder
  imbalance is unmeasured; there the other ranks' traffic also loads the owner's ports, so
  f* will be lower.
