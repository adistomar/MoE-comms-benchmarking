# Copyright (c) 2026. Hierarchical MoE dispatch — 2D grid, communicators, placement.
"""
Shared scaffolding for the hierarchical (AGv-inner / A2A-outer) MoE dispatcher.

Ranks form a G x g grid, rank = group_id*g + pos, P = G*g = world_size:
  - Row / inner group = fixed group_id (g contiguous ranks) -> NVLS multicast AGv/RSv.
  - Column / outer group = fixed pos (G strided ranks)      -> directed P2P all-to-all.

Group-major expert placement makes the two-hop routing decompose with NO permute
table: expert e is owned by rank `e // epr`; its destination group is `e // (E//G)`
and its position-within-group is `(e % (E//G)) // epr`. The invariant
    owning_rank(e) == dest_group(e) * g + dest_pos(e)
is what lets the outer hop route to the group and the inner hop finish to the rank.

Grid / placement math is plain Python (torch-free) so it is unit-testable without a
GPU (`python3 hier_common.py`). torch/dist are imported lazily inside the helpers
that need them.
"""

import os
from dataclasses import dataclass

# ---- 2D grid -----------------------------------------------------------------
@dataclass(frozen=True)
class HierGrid:
    """The G x g rank grid, from this rank's point of view."""
    world: int
    g: int
    rank: int

    def __post_init__(self):
        assert self.world % self.g == 0, f"g={self.g} must divide world={self.world}"
        assert 1 <= self.g <= self.world

    @property
    def G(self) -> int:
        return self.world // self.g

    @property
    def group_id(self) -> int:
        return self.rank // self.g

    @property
    def pos(self) -> int:
        return self.rank % self.g

    def row_ranks(self) -> "list[int]":
        """The g ranks in this rank's inner group (same group_id)."""
        return [self.group_id * self.g + p for p in range(self.g)]

    def col_ranks(self) -> "list[int]":
        """The G ranks in this rank's outer group (same pos)."""
        return [gi * self.g + self.pos for gi in range(self.G)]

    def all_row_ranks(self) -> "list[list[int]]":
        return [[gi * self.g + p for p in range(self.g)] for gi in range(self.G)]

    def all_col_ranks(self) -> "list[list[int]]":
        return [[gi * self.g + p for gi in range(self.G)] for p in range(self.g)]


# ---- Group-major expert placement --------------------------------------------
@dataclass(frozen=True)
class HierPlacement:
    """Group-major expert -> (rank, group, pos) mapping."""
    num_experts: int
    world: int
    g: int

    def __post_init__(self):
        assert self.num_experts % self.world == 0, "E must be divisible by P"
        assert self.world % self.g == 0, "P must be divisible by g (=> P % G == 0)"
        # A group's expert block must split evenly across its g ranks.
        assert self.experts_per_group % self.epr == 0

    @property
    def G(self) -> int:
        return self.world // self.g

    @property
    def epr(self) -> int:
        """Experts per rank."""
        return self.num_experts // self.world

    @property
    def experts_per_group(self) -> int:
        return self.num_experts // self.G  # == g * epr

    def owning_rank(self, e: int) -> int:
        return e // self.epr

    def dest_group(self, e: int) -> int:
        return e // self.experts_per_group

    def dest_pos(self, e: int) -> int:
        return (e % self.experts_per_group) // self.epr


# ---- Communicator construction (torch) ---------------------------------------
def make_groups(grid: HierGrid):
    """Create all G row subgroups and all g column subgroups (every rank must enter
    every new_group — it is collective), returning THIS rank's (row_group, col_group).
    """
    import torch.distributed as dist
    # Order matters and must be identical on every rank: all rows first, then all cols.
    row_groups = [dist.new_group(rr) for rr in grid.all_row_ranks()]
    col_groups = [dist.new_group(cr) for cr in grid.all_col_ranks()]
    return row_groups[grid.group_id], col_groups[grid.pos]


# ---- Routing decomposition (torch) -------------------------------------------
def dest_group_mask(topk_idx, placement: HierPlacement):
    """[n, K] expert ids -> [n, G] bool: which destination groups each token routes to."""
    import torch
    import torch.nn.functional as F
    groups = topk_idx // placement.experts_per_group           # [n, K] in [0, G)
    if topk_idx.numel() == 0:
        return torch.zeros(topk_idx.shape[0], placement.G,
                           dtype=torch.bool, device=topk_idx.device)
    return F.one_hot(groups, num_classes=placement.G).any(dim=1)  # [n, G] bool


def local_mask(gathered_routing, rank: int, epr: int):
    """[valid, K] gathered expert ids -> [valid] bool: token routed >=1 expert local to `rank`.
    Same predicate as bench_nvls.functional_roundtrip / bench_nccl, but at a global rank."""
    return ((gathered_routing >= rank * epr) & (gathered_routing < (rank + 1) * epr)).any(dim=1)


# ---- Per-peer data-pointer acquisition (torch; reused by K1/K2) ---------------
def peer_pointer_array(hdl, world: int, device, dtype, shape):
    """Return (peer_va, keepalive, source_desc): device VA of a [world] array of
    per-rank symmetric-buffer base pointers for directed (non-multicast) P2P stores.
    Mirrors probe_symm_mem.py's resolution order (validated by the K-1 probe):
      buffer_ptrs_dev -> get_buffer(r).data_ptr() -> (None => fallback B needed).
    Keep `keepalive` alive for the kernel's lifetime (it backs the pointer array)."""
    import torch
    bpd = getattr(hdl, "buffer_ptrs_dev", None)
    if bpd is not None:
        try:
            return int(bpd), None, "buffer_ptrs_dev"
        except (TypeError, ValueError):
            pass
    if hasattr(hdl, "get_buffer"):
        for call in (lambda r: hdl.get_buffer(r, shape, dtype, 0),
                     lambda r: hdl.get_buffer(r, shape, dtype),
                     lambda r: hdl.get_buffer(r, list(shape), dtype)):
            try:
                ptrs = [call(r).data_ptr() for r in range(world)]
                arr = torch.tensor(ptrs, dtype=torch.int64, device=device)
                return arr.data_ptr(), arr, "get_buffer().data_ptr()"
            except (TypeError, RuntimeError):
                continue
    return None, None, None


# ---- Outer-hop (column) directed-a2a contiguous landing offsets (K1b) ---------
def outer_send_plan(token_dest_groups, M, my_group):
    """Contiguous landing rows for this rank's outbound tokens in the column-wise
    G->G directed all-to-all (K1b scheme).

    Within a column (fixed pos), the G ranks are one-per-group. Source group `gs`
    sends each of its tokens to every destination group the token routes to. On the
    destination rank, source `gs`'s tokens land CONTIGUOUSLY after all lower-indexed
    sources' tokens for that destination -> no compaction kernel needed downstream.

    Args:
      token_dest_groups: list (len n) of this rank's tokens; each an iterable of the
        distinct destination group ids (in [0, G)) that token routes to.
      M: G x G int matrix, M[s][d] = # tokens source group s (in THIS column) sends
         to destination d. Row `my_group` must equal this rank's own per-dest counts;
         other rows come from the column count-matrix exchange.
      my_group: this rank's source group id (gs).

    Returns:
      sends: list of (dest_group, token_idx, dest_row).
      recv_counts: list (len G), recv_counts[d] = total tokens landing on dest d
        (this column) = the `local_tokens` the inner AGv sees at (d, this_pos).
    """
    G = len(M)
    cursor = [sum(M[s][d] for s in range(my_group)) for d in range(G)]  # start of my block per d
    sends = []
    for t, groups in enumerate(token_dest_groups):
        for d in groups:
            sends.append((d, t, cursor[d]))
            cursor[d] += 1
    recv_counts = [sum(M[s][d] for s in range(G)) for d in range(G)]
    return sends, recv_counts


def device_send_plan(mask, M, my_group):
    """Graph-capturable, fixed-size (no host sync) equivalent of outer_send_plan.

    Produces the SAME (dest, token, row) assignment as outer_send_plan but as flat
    [n*G] int32 device tensors, with INACTIVE slots marked dest=-1 (the scatter kernel
    skips dest<0). Avoids torch.nonzero / .tolist() so the whole dispatch can be CUDA-
    graph captured.

    Args:
      mask: [n, G] bool — dest_group_mask (which groups each token routes to).
      M:    [G, G] int64 device tensor — column count matrix, M[s, d].
      my_group: int, this rank's source group.
    Returns:
      send_dest, send_tok, send_row: each [n*G] int32. dest=-1 => inactive.
    """
    import torch
    n, G = mask.shape
    dev = mask.device
    base = M[:my_group].sum(dim=0)                              # [G] exclusive prefix over sources
    mi = mask.to(torch.int64)
    within = mi.cumsum(dim=0) - mi                              # [n,G] # of my earlier tokens -> d
    rows = base.view(1, G) + within                            # [n,G] contiguous dest row (if active)
    tok = torch.arange(n, device=dev).view(n, 1).expand(n, G)
    grp = torch.arange(G, device=dev).view(1, G).expand(n, G)
    dest = torch.where(mask, grp, torch.full_like(grp, -1))
    return (dest.reshape(-1).to(torch.int32),
            tok.reshape(-1).to(torch.int32),
            rows.reshape(-1).to(torch.int32))


# ---- Pure-python self-test (no torch; runs anywhere) -------------------------
def _selftest():
    def check_grid(world, g):
        # Rows and columns must each PARTITION the ranks exactly.
        grids = [HierGrid(world, g, r) for r in range(world)]
        rows = {tuple(gr.row_ranks()) for gr in grids}
        cols = {tuple(gr.col_ranks()) for gr in grids}
        assert sum(len(r) for r in rows) == world and set().union(*rows) == set(range(world)), rows
        assert sum(len(c) for c in cols) == world and set().union(*cols) == set(range(world)), cols
        assert len(rows) == grids[0].G and len(cols) == g
        # all_row_ranks / all_col_ranks agree with the per-rank views.
        assert set(rows) == {tuple(x) for x in grids[0].all_row_ranks()}
        assert set(cols) == {tuple(x) for x in grids[0].all_col_ranks()}
        # A rank sits at the row/col intersection.
        for gr in grids:
            assert gr.rank in gr.row_ranks() and gr.rank in gr.col_ranks()

    def check_placement(E, world, g):
        p = HierPlacement(E, world, g)
        for e in range(E):
            r = p.owning_rank(e)
            # THE invariant: outer hop (dest_group) + inner hop (dest_pos) == owning rank.
            assert p.dest_group(e) * g + p.dest_pos(e) == r, (e, r)
            assert 0 <= p.dest_group(e) < p.G and 0 <= p.dest_pos(e) < g
        # Every rank owns exactly epr experts.
        from collections import Counter
        c = Counter(p.owning_rank(e) for e in range(E))
        assert set(c.values()) == {p.epr} and len(c) == world

    def check_send_plan(G, seed):
        # Simulate a whole column: G source groups, each with random tokens routing
        # to random subsets of dest groups. Build M, compute each source's landing
        # rows, and assert the union per dest is EXACTLY [0, R[d]) with no collision.
        import random
        rng = random.Random(seed)
        per_source = []          # token_dest_groups for each source group
        for _ in range(G):
            n = rng.randint(0, 5)
            per_source.append([sorted(rng.sample(range(G), rng.randint(1, G))) for _ in range(n)])
        # M[s][d] = # tokens on source s routing to d.
        M = [[sum(1 for groups in per_source[s] if d in groups) for d in range(G)] for s in range(G)]
        landed = {d: [] for d in range(G)}   # dest -> list of rows written
        for gs in range(G):
            sends, recv = outer_send_plan(per_source[gs], M, gs)
            # recv_counts must be identical regardless of which source computes it.
            assert recv == [sum(M[s][d] for s in range(G)) for d in range(G)]
            # device_send_plan (pure-python mirror) must yield the SAME (dest,token,row) set.
            base = [sum(M[s][d] for s in range(gs)) for d in range(G)]
            running, dev_sends = [0] * G, []
            for t, groups in enumerate(per_source[gs]):
                gset = set(groups)
                for d in range(G):
                    if d in gset:
                        dev_sends.append((d, t, base[d] + running[d]))
                        running[d] += 1
            assert set(sends) == set(dev_sends), (gs, sorted(sends), sorted(dev_sends))
            for d, t, row in sends:
                landed[d].append(row)
        for d in range(G):
            R = sum(M[s][d] for s in range(G))
            assert sorted(landed[d]) == list(range(R)), (d, sorted(landed[d]), R)  # contiguous, no gaps/dups

    for world, g in [(4, 2), (4, 1), (4, 4), (64, 4), (64, 2), (64, 8), (36, 6)]:
        check_grid(world, g)
    for E, world, g in [(512, 4, 2), (512, 64, 4), (512, 64, 2), (512, 64, 8), (512, 32, 4)]:
        check_placement(E, world, g)
    for G in [2, 3, 4, 8, 16]:
        for seed in range(20):
            check_send_plan(G, seed)
    print("hier_common self-test: PASS")


if __name__ == "__main__":
    _selftest()
