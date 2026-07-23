# Copyright (c) 2026. Directed peer-to-peer all-to-all — hierarchical MoE outer hop.
"""
K1: the OUTER hop of the hierarchical dispatcher — a directed (non-multicast) P2P
all-to-all within a COLUMN of G ranks (one rank per group). Confirmed viable by
probe_symm_mem.py: the torch _SymmetricMemory handle exposes `buffer_ptrs_dev`
(device array of per-peer base pointers), and a directed `st_128(multicast_op=False)`
to a peer base, fenced by `symm_mem_sync`, round-trips correctly.

`directed_a2a_scatter` (dispatch): each rank sends each of its local tokens to the
column-peer ranks that own the groups the token routes to, landing at precomputed
CONTIGUOUS rows (K1b; rows come from hier_common.outer_send_plan). The per-token copy
mirrors the addressing of the vendored AGv-V kernel (variable_collectives.py) but
writes to a PEER buffer base (from buffer_ptrs_dev) instead of the multicast VA.

Fixed grid = MAX_NUM_BLOCKS on every rank so the end-of-kernel `symm_mem_sync` (which
synchronizes blocks with matching block_id across the column) always has matched
participants regardless of the per-rank send count.
"""

from unittest.mock import MagicMock

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except ImportError:
    triton = MagicMock()
    tl = MagicMock()
    HAVE_TRITON = False

from ._compat import null_decorator
if not HAVE_TRITON:
    triton.jit = null_decorator

from .multimem_asm import ld_128, st_128, st_32
from .barrier import symm_mem_sync, _send_signal, _wait_signal
from .utils import sync_threads, get_flat_tid

# One CTA per send; fixed grid so the column barrier participants match across ranks.
MAX_NUM_BLOCKS = 148


@triton.jit
def _masked_copy_kernel(ah_ptr, ar_ptr, rsv_ptr, valid, LO, HI,
                        K: tl.constexpr, NPT_H: tl.constexpr, BLOCK: tl.constexpr):
    """Combine mask, in-kernel: rsv[t] = ah[t] if token t routes to >=1 expert in [LO,HI)
    (this rank's experts), else 0. Replaces local_mask + torch.where + f32/bf16 casts (~5 torch
    launches) with one kernel. rsv is pre-zeroed once in setup, so UNOWNED tokens are simply
    skipped (they stay 0) — no zero-store needed. `owned` is uniform per token, so the branch
    around the copy has no __syncthreads and is safe."""
    pid = tl.program_id(0)
    tid = tl.arange(0, BLOCK)
    ah_u64 = ah_ptr.to(tl.pointer_type(tl.uint64))
    rsv_u64 = rsv_ptr.to(tl.pointer_type(tl.uint64))
    for t in range(pid, valid, tl.num_programs(0)):
        owned = 0
        for k in range(K):
            e = tl.load(ar_ptr + t * K + k)                  # int64 expert id
            owned = owned | ((e >= LO) & (e < HI)).to(tl.int32)
        if owned == 1:
            for co in range(0, NPT_H, BLOCK):
                c = co + tid
                m = c < NPT_H
                off = (t * NPT_H + c) * 2                     # 128-bit pack -> 2 uint64 units
                x, y, z, w = ld_128(ah_u64 + off, mask=m, multicast_op=False)
                st_128(rsv_u64 + off, x, y, z, w, mask=m, multicast_op=False)


def masked_copy(ah, ar, rsv, valid, lo, hi, topk, hidden_size, max_blocks=MAX_NUM_BLOCKS):
    """Launch the in-kernel combine mask (replaces the torch mask in _mask_rsv). rsv must be
    pre-zeroed (setup_batch). Fixed grid (grid-strided over valid tokens)."""
    assert HAVE_TRITON, "Triton required for masked_copy."
    assert hidden_size % 8 == 0
    npt_h = hidden_size // 8
    block = min(triton.next_power_of_2(npt_h), 1024)
    _masked_copy_kernel[(max_blocks, 1, 1)](
        ah, ar, rsv, int(valid), lo, hi,
        K=topk, NPT_H=npt_h, BLOCK=block,
        num_warps=min(max(block // 32, 1), 8))


@triton.jit
def _directed_a2a_scatter_kernel(
    peer_ptrs,            # buffer_ptrs_dev of the COLUMN symm buffer (VA of [GCOL] base ptrs)
    signal_pad_ptrs,      # column signal_pad_ptrs_dev (for the barrier)
    hidden_ptr,           # local [n, H] bf16 (this rank's tokens)
    routing_ptr,          # local [n, K] int64
    srctok_ptr,           # local [n] int32 (carried return index)
    send_dest_ptr,        # [S] int32 destination group (column index) per send
    send_tok_ptr,         # [S] int32 local token index per send
    send_row_ptr,         # [S] int32 destination row per send (contiguous, from outer_send_plan)
    num_sends,            # S (runtime scalar)
    H_OFF, R_OFF, S_OFF,  # byte offsets of hidden / routing / srctok regions in the column buffer
    NPT_H: tl.constexpr,  # H // 8   (128-bit packs of hidden per token; bf16 => 8/pack)
    NPT_R: tl.constexpr,  # (K*8)//16 (128-bit packs of routing per token; int64)
    BLOCK: tl.constexpr,
    MY_GROUP: tl.constexpr,   # this rank's column index (group_id)
    GCOL: tl.constexpr,       # column size G
):
    tid = tl.arange(0, BLOCK)
    peer_ptrs = peer_ptrs.to(tl.int64).to(tl.pointer_type(tl.uint64))  # Triton-3.6 widen
    hidden_u64 = hidden_ptr.to(tl.pointer_type(tl.uint64))
    routing_u64 = routing_ptr.to(tl.pointer_type(tl.uint64))

    for s in range(tl.program_id(0), num_sends, tl.num_programs(0)):
        d = tl.load(send_dest_ptr + s)
        if d >= 0:                                            # skip padded/inactive send slots
            t = tl.load(send_tok_ptr + s)
            row = tl.load(send_row_ptr + s)
            base_i = tl.load(peer_ptrs + d)                   # peer d's buffer base VA (uint64)
            base_u64 = base_i.to(tl.pointer_type(tl.uint64))
            base_u32 = base_i.to(tl.pointer_type(tl.uint32))

            # hidden: NPT_H 128-bit packs, local read -> directed peer write.
            for co in range(0, NPT_H, BLOCK):
                c = co + tid
                m = c < NPT_H
                src = hidden_u64 + (t * NPT_H + c) * 2
                dst = base_u64 + H_OFF // 8 + (row * NPT_H + c) * 2
                x, y, z, w = ld_128(src, mask=m, multicast_op=False)
                st_128(dst, x, y, z, w, mask=m, multicast_op=False)

            # routing: NPT_R 128-bit packs.
            for co in range(0, NPT_R, BLOCK):
                c = co + tid
                m = c < NPT_R
                src = routing_u64 + (t * NPT_R + c) * 2
                dst = base_u64 + R_OFF // 8 + (row * NPT_R + c) * 2
                x, y, z, w = ld_128(src, mask=m, multicast_op=False)
                st_128(dst, x, y, z, w, mask=m, multicast_op=False)

            # srctok: single int32 (redundant same store across the CTA — idempotent, cf. metadata.py).
            st_32(base_u32 + S_OFF // 4 + row, t.to(tl.uint32),
                  tl.full([], 1, tl.int1), multicast_op=False)

    # Publish this rank's directed stores and acquire peers' — over the COLUMN group.
    sync_threads()
    symm_mem_sync(signal_pad_ptrs, None, MY_GROUP, GCOL,
                  hasPreviousMemAccess=True, hasSubsequentMemAccess=True)


def directed_a2a_scatter(col_buffer_ptrs, col_signal_pad_ptrs,
                         hidden, routing, srctok,
                         send_dest, send_tok, send_row,
                         h_off, r_off, s_off,
                         my_group, gcol, hidden_size, topk, max_blocks=MAX_NUM_BLOCKS):
    """Launch the directed-P2P scatter (dispatch outer hop). All device tensors must be
    contiguous. `col_buffer_ptrs`/`col_signal_pad_ptrs` are the COLUMN handle's
    buffer_ptrs_dev / signal_pad_ptrs_dev. Grid is fixed at MAX_NUM_BLOCKS so the
    end-of-kernel column barrier always matches across ranks."""
    assert HAVE_TRITON, "Triton required for directed_a2a_scatter."
    assert hidden_size % 8 == 0, "H must be a multiple of 8 (bf16 128-bit packing)."
    assert (topk * 8) % 16 == 0, "K*8 must be 16-byte aligned for 128-bit routing packs."
    npt_h = hidden_size // 8
    npt_r = (topk * 8) // 16
    block = min(triton.next_power_of_2(npt_h), 1024)
    num_sends = int(send_dest.numel())
    _directed_a2a_scatter_kernel[(max_blocks, 1, 1)](
        col_buffer_ptrs, col_signal_pad_ptrs,
        hidden, routing, srctok,
        send_dest, send_tok, send_row, num_sends,
        h_off, r_off, s_off,
        NPT_H=npt_h, NPT_R=npt_r, BLOCK=block,
        MY_GROUP=my_group, GCOL=gcol,
        num_warps=min(max(block // 32, 1), 8),
    )


@triton.jit
def _directed_a2a_gather_kernel(
    peer_ptrs, signal_pad_ptrs,
    partial_ptr,          # local [R, H] bf16 — this rank's per-landed-token partials (inner RSv out)
    send_dest_ptr,        # [R] int32 source group (column index) to return each row to
    send_row_ptr,         # [R] int32 destination slot in the source's combine buffer
    num_sends,            # R (runtime scalar)
    OUT_OFF,              # byte offset of the combine-recv region in the source's column buffer
    NPT_H: tl.constexpr,  # H // 8
    BLOCK: tl.constexpr,
    MY_GROUP: tl.constexpr,   # this rank's column index (= destination-group of the tokens it holds)
    GCOL: tl.constexpr,
):
    tid = tl.arange(0, BLOCK)
    peer_ptrs = peer_ptrs.to(tl.int64).to(tl.pointer_type(tl.uint64))  # Triton-3.6 widen
    partial_u64 = partial_ptr.to(tl.pointer_type(tl.uint64))
    # Send in landed-row order: send `s` returns local partial row `s`.
    for s in range(tl.program_id(0), num_sends, tl.num_programs(0)):
        d = tl.load(send_dest_ptr + s)                      # source group gs
        row_out = tl.load(send_row_ptr + s)                 # slot in gs's combine buffer
        base_u64 = tl.load(peer_ptrs + d).to(tl.pointer_type(tl.uint64))
        for co in range(0, NPT_H, BLOCK):
            c = co + tid
            m = c < NPT_H
            src = partial_u64 + (s * NPT_H + c) * 2
            dst = base_u64 + OUT_OFF // 8 + (row_out * NPT_H + c) * 2
            x, y, z, w = ld_128(src, mask=m, multicast_op=False)
            st_128(dst, x, y, z, w, mask=m, multicast_op=False)
    sync_threads()
    symm_mem_sync(signal_pad_ptrs, None, MY_GROUP, GCOL,
                  hasPreviousMemAccess=True, hasSubsequentMemAccess=True)


def directed_a2a_gather(col_buffer_ptrs, col_signal_pad_ptrs, partial,
                        send_dest, send_row, out_off, my_group, gcol, hidden_size,
                        max_blocks=MAX_NUM_BLOCKS):
    """Combine outer hop: directed-P2P return of per-group partials to source ranks
    (one per landed token, in landed order), landing at DISTINCT per-dest-group slots
    of the source's combine buffer. The source then locally fp32-reduces over the G
    slots (done in torch at K2). Mirror of directed_a2a_scatter."""
    assert HAVE_TRITON, "Triton required for directed_a2a_gather."
    assert hidden_size % 8 == 0
    npt_h = hidden_size // 8
    block = min(triton.next_power_of_2(npt_h), 1024)
    num_sends = int(send_dest.numel())
    _directed_a2a_gather_kernel[(max_blocks, 1, 1)](
        col_buffer_ptrs, col_signal_pad_ptrs, partial,
        send_dest, send_row, num_sends, out_off,
        NPT_H=npt_h, BLOCK=block, MY_GROUP=my_group, GCOL=gcol,
        num_warps=min(max(block // 32, 1), 8),
    )


# ============================================================================
# K3 increment 1: FUSED, BARRIER-FREE pipelined DISPATCH (outer scatter + inner AGv).
# The two hops run concurrently in ONE persistent grid, coupled by per-landed-row
# release/acquire flags instead of the two global symm_mem_sync barriers:
#
#   producer CTAs (pid <  SCATTER_CTAS): directed-P2P store each local token into its
#     dest-column peer's dispatch buffer, then `_send_signal(release)` the peer's
#     per-row flag[landed_row].  NO end-of-scatter column barrier.
#   consumer CTAs (pid >= SCATTER_CTAS): for each landed row r, `_wait_signal(acquire)`
#     on the LOCAL flag[r] (set by whichever column-peer scattered it), then multicast
#     (multimem.st) that token's hidden+routing across the g-rank ROW group.
#
# So landed row r's inner multicast starts as soon as its outer store lands — token t's
# inner hop overlaps token t+1's outer hop, with the fabric hiding the P2P latency. One
# ROW barrier is retained at the very end so downstream (mask/RSv) sees the full gather;
# increment 2 folds that away too by fusing combine. Flags self-reset (CAS 1->0 on
# acquire) so the whole thing is CUDA-graph replayable, exactly like symm_mem_sync.
# ============================================================================
@triton.jit
def _fused_dispatch_kernel(
    # --- scatter (column / directed-P2P to peers) ---
    peer_ptrs,            # col dispatch buffer_ptrs_dev (VA of [GCOL] peer bases, bytes)
    hidden_ptr, routing_ptr, srctok_ptr,   # local [n,H] bf16 / [n,K] i64 / [n] i32
    send_dest_ptr, send_tok_ptr, send_row_ptr, num_sends,
    H_OFF, R_OFF, S_OFF, FLAG_OFF,         # byte offsets of regions in the dispatch buffer
    # --- AGv (row / multicast broadcast of landed tokens) ---
    disp_h_local, disp_r_local, flag_local,  # THIS rank's landed hidden/routing + own flag region
    agv_mc_ptr, agv_signal_pad_ptrs,       # row AGv multicast VA + row signal pad
    rank_token_offset_ptr, num_landed,     # AGv global write offset + R (runtime)
    AH_OFF, AR_OFF,                        # byte offsets of hidden/routing in the AGv buffer
    dbg_ptr, wait_iters,                   # DIAGNOSTIC [col_cap] i32 + fixed poll count (runtime)
    NPT_H: tl.constexpr, NPT_R: tl.constexpr, BLOCK: tl.constexpr,
    MY_GROUP: tl.constexpr, GCOL: tl.constexpr,
    ROW_RANK: tl.constexpr, ROW_WORLD: tl.constexpr,
    SCATTER_CTAS: tl.constexpr, NO_FLAGS: tl.constexpr = False,
    SKIP_WAIT: tl.constexpr = False,
):
    pid = tl.program_id(0)
    tid = tl.arange(0, BLOCK)
    ftid = get_flat_tid()
    nprog = tl.num_programs(0)
    peer_ptrs = peer_ptrs.to(tl.int64).to(tl.pointer_type(tl.uint64))
    hidden_u64 = hidden_ptr.to(tl.pointer_type(tl.uint64))
    routing_u64 = routing_ptr.to(tl.pointer_type(tl.uint64))

    # ==== Phase 1: directed-P2P scatter + per-row release flag. EVERY CTA participates
    # (grid-strided) so scatter uses all SMs. No column barrier — the per-row flags in the
    # peer's dispatch flag region hand each token off to its inner-multicast consumer. ====
    for s in range(pid, num_sends, nprog):
        d = tl.load(send_dest_ptr + s)
        if d >= 0:
            t = tl.load(send_tok_ptr + s)
            row = tl.load(send_row_ptr + s)
            base_i = tl.load(peer_ptrs + d)                   # peer d base (bytes, uint64)
            base_u64 = base_i.to(tl.pointer_type(tl.uint64))
            base_u32 = base_i.to(tl.pointer_type(tl.uint32))
            for co in range(0, NPT_H, BLOCK):
                c = co + tid; m = c < NPT_H
                src = hidden_u64 + (t * NPT_H + c) * 2
                dst = base_u64 + H_OFF // 8 + (row * NPT_H + c) * 2
                x, y, z, w = ld_128(src, mask=m, multicast_op=False)
                st_128(dst, x, y, z, w, mask=m, multicast_op=False)
            for co in range(0, NPT_R, BLOCK):
                c = co + tid; m = c < NPT_R
                src = routing_u64 + (t * NPT_R + c) * 2
                dst = base_u64 + R_OFF // 8 + (row * NPT_R + c) * 2
                x, y, z, w = ld_128(src, mask=m, multicast_op=False)
                st_128(dst, x, y, z, w, mask=m, multicast_op=False)
            st_32(base_u32 + S_OFF // 4 + row, t.to(tl.uint32),
                  tl.full([], 1, tl.int1), multicast_op=False)
            # Publish this row's data, THEN flip the peer's per-row flag. ftid<1 guard is safe:
            # no __syncthreads follows in the scatter path.
            sync_threads()
            if ftid < 1 and not NO_FLAGS:
                tl.atomic_xchg(base_u32 + FLAG_OFF // 4 + row, 1, sem="release", scope="sys")

    # ==== Phase 2: per-row acquire-wait + inner row multicast. EVERY CTA participates
    # (grid-strided over landed rows), so the multicast also uses all SMs. A CTA's phase-2
    # work overlaps OTHER CTAs' still-running phase-1 scatter (the flags, not a barrier,
    # gate each token) — so all SMs serve both hops without the old static split. ====
    rank_token_offset = tl.load(rank_token_offset_ptr)
    disp_h_u64 = disp_h_local.to(tl.pointer_type(tl.uint64))
    disp_r_u64 = disp_r_local.to(tl.pointer_type(tl.uint64))
    agv_u64 = agv_mc_ptr.to(tl.int64).to(tl.pointer_type(tl.uint64))
    flag_u32 = flag_local.to(tl.pointer_type(tl.uint32))
    for r in range(pid, num_landed, nprog):
        if not NO_FLAGS and not SKIP_WAIT:
            # UNIFORM bounded spin-wait (all lanes read the same addr/value -> uniform loop
            # condition -> no ftid divergence, no __syncthreads deadlock). atomic_add(,0) =
            # non-hoisted acquire read. wait_iters is a safety ceiling.
            v = 0
            i = 0
            while (v != 1) & (i < wait_iters):
                v = tl.atomic_add(flag_u32 + r, 0, sem="acquire", scope="sys").to(tl.int32)
                i += 1
            tl.store(dbg_ptr + r, v)
            tl.store(flag_u32 + r, 0)   # reset flag for the next layer (uniform, no divergence)
        sync_threads()                                        # all threads see the acquire
        gr = rank_token_offset + r
        for co in range(0, NPT_H, BLOCK):
            c = co + tid; m = c < NPT_H
            src = disp_h_u64 + (r * NPT_H + c) * 2
            dst = agv_u64 + AH_OFF // 8 + (gr * NPT_H + c) * 2
            x, y, z, w = ld_128(src, mask=m, multicast_op=False)
            st_128(dst, x, y, z, w, mask=m, multicast_op=True)
        for co in range(0, NPT_R, BLOCK):
            c = co + tid; m = c < NPT_R
            src = disp_r_u64 + (r * NPT_R + c) * 2
            dst = agv_u64 + AR_OFF // 8 + (gr * NPT_R + c) * 2
            x, y, z, w = ld_128(src, mask=m, multicast_op=False)
            st_128(dst, x, y, z, w, mask=m, multicast_op=True)

    # ==== One retained ROW barrier (ALL CTAs) so downstream sees the full gather. ====
    sync_threads()
    symm_mem_sync(agv_signal_pad_ptrs, pid, ROW_RANK, ROW_WORLD,
                  hasPreviousMemAccess=True, hasSubsequentMemAccess=True)


def fused_dispatch(peer_ptrs, hidden, routing, srctok,
                   send_dest, send_tok, send_row, h_off, r_off, s_off, flag_off,
                   disp_h_local, disp_r_local, flag_local,
                   agv_mc_ptr, agv_signal_pad_ptrs, rank_token_offset, num_landed,
                   ah_off, ar_off, dbg, my_group, gcol, row_rank, row_world,
                   hidden_size, topk, scatter_ctas=96, max_blocks=MAX_NUM_BLOCKS,
                   no_flags=False, skip_wait=False, wait_iters=50000):
    """Fused barrier-free dispatch: outer directed-P2P scatter (producer CTAs) piped into the
    inner row multicast AGv (consumer CTAs) via per-landed-row release/acquire flags in the
    dispatch buffer's dedicated flag region (col_cap-sized). Replaces directed_a2a_scatter +
    multimem_all_gatherv (hidden+routing) with a single launch and no scatter barrier. The flag
    region must be zero on entry (bench zeroes it in setup_batch)."""
    assert HAVE_TRITON, "Triton required for fused_dispatch."
    assert hidden_size % 8 == 0 and (topk * 8) % 16 == 0
    npt_h = hidden_size // 8
    npt_r = (topk * 8) // 16
    block = min(triton.next_power_of_2(npt_h), 1024)
    _fused_dispatch_kernel[(max_blocks, 1, 1)](
        peer_ptrs, hidden, routing, srctok,
        send_dest, send_tok, send_row, int(send_dest.numel()),
        h_off, r_off, s_off, flag_off,
        disp_h_local, disp_r_local, flag_local,
        agv_mc_ptr, agv_signal_pad_ptrs, rank_token_offset, int(num_landed),
        ah_off, ar_off, dbg, int(wait_iters),
        NPT_H=npt_h, NPT_R=npt_r, BLOCK=block, MY_GROUP=my_group, GCOL=gcol,
        ROW_RANK=row_rank, ROW_WORLD=row_world, SCATTER_CTAS=scatter_ctas,
        NO_FLAGS=no_flags, SKIP_WAIT=skip_wait,
        num_warps=min(max(block // 32, 1), 8),
    )
