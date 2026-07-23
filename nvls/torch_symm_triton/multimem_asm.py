# Copyright (c) Meta Platforms, Inc. and affiliates.
# pylint: disable=line-too-long

# Adapted from https://github.com/yifuwang/symm-mem-recipes.git


from unittest.mock import MagicMock

from ._compat import null_decorator

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = MagicMock()
    tl = MagicMock()
    triton.jit = null_decorator


@triton.jit
def ld_128(ptr, mask, multicast_op: tl.constexpr, reduce_f32: tl.constexpr = False):
    """
    Loads 128 bits from memory into registers.

    This function abstracts two distinct hardware behaviors based on `multicast_op`:

    1.  **Standard Load (`multicast_op=False`)**:
        -   **Semantics:** Local Global Memory Load.
        -   **Action:** Reads 128 bits from `ptr` in global memory into the local register file.

    2.  **Multicast Reduce-Load (`multicast_op=True`)**:
        -   **Semantics:** "Pull" Reduction over NVLink.
        -   **Action:** Simultaneously reads 128 bits from the *same* address across all peer GPUs
            in the multicast group, sums them, and loads the result into the local register file.
        -   **Hardware:** Uses `multimem.ld_reduce` (Hopper+).
        -   When `reduce_f32=False` (default): bf16x2 addition with f32 accumulation
            (128 bits = 8 x bf16, 2 per register).
        -   When `reduce_f32=True`: native f32 addition
            (128 bits = 4 x fp32, 1 per register).

    Args:
        ptr: Memory pointer to the source buffer.
        mask: Boolean predicate. If False, the operation is skipped (no-op).
        multicast_op (tl.constexpr): Toggles between standard load (False)
            and multicast-reduce (True).
        reduce_f32 (tl.constexpr): When True and multicast_op=True, uses f32 reduction
            instead of bf16x2 reduction. Default False.

    Returns:
        Four 32-bit registers (tl.uint32), representing 128 bits of loaded data.
    """
    if multicast_op:
        if reduce_f32:
            # fp32 reduction: multimem.ld_reduce.add.v4.f32
            # Each 128-bit load reduces 4 x fp32 values across peers.
            return tl.inline_asm_elementwise(
                """
                {
                    .reg .pred %p0;
                    setp.ne.s32 %p0, $5, 1;
                    @%p0 bra end;
                    multimem.ld_reduce.relaxed.sys.global.add.v4.f32 {$0, $1, $2, $3}, [$4];
                    end:
                }
                """,
                "=r,=r,=r,=r,l,r",
                args=[ptr, mask.to(tl.int32)],
                dtype=(tl.uint32, tl.uint32, tl.uint32, tl.uint32),
                is_pure=True,
                pack=1,
            )
        else:
            # bf16x2 reduction with f32 accumulation: multimem.ld_reduce.add.acc::f32.v4.bf16x2
            # Each 128-bit load reduces 8 x bf16 values (packed as 4 x bf16x2) across peers.
            return tl.inline_asm_elementwise(
                """
                {
                    .reg .pred %p0;
                    setp.ne.s32 %p0, $5, 1;
                    @%p0 bra end;
                    multimem.ld_reduce.relaxed.sys.global.add.acc::f32.v4.bf16x2 {$0, $1, $2, $3}, [$4]; 
                    end:
                }
                """,
                "=r,=r,=r,=r,l,r",
                args=[ptr, mask.to(tl.int32)],
                dtype=(tl.uint32, tl.uint32, tl.uint32, tl.uint32),
                is_pure=True,
                pack=1,
            )
    else:
        return tl.inline_asm_elementwise(
            """
        {
            .reg .pred %p0;
            setp.ne.s32 %p0, $5, 1;
            @%p0 bra end;
            ld.global.v4.u32 {$0, $1, $2, $3}, [$4];
            end:
        }
        """,
            "=r,=r,=r,=r,l,r",
            args=[ptr, mask.to(tl.int32)],
            dtype=(tl.uint32, tl.uint32, tl.uint32, tl.uint32),
            is_pure=True,
            pack=1,
        )


@triton.jit
def st_128(ptr, x, y, z, w, mask, multicast_op):
    """
    Stores 128 bits (8 x bf16) from registers to memory.

    This function abstracts two distinct hardware behaviors based on `multicast_op`:

    1.  **Standard Store (`multicast_op=False`)**:
        -   **Semantics:** Local Global Memory Store.
        -   **Action:** Writes 128 bits from local registers to `ptr` in global memory.

    2.  **Multicast Store (`multicast_op=True`)**:
        -   **Semantics:** "Push" Broadcast over NVLink.
        -   **Action:** Writes 128 bits from local registers to the `ptr` address in
            the global memory of **all** peer GPUs in the multicast group simultaneously.
        -   **Hardware:** Uses `multimem.st` (Hopper+).
        -   **Use Case:** The "Broadcast" or "All-Gather" step in collective operations.

    Args:
        ptr: Memory pointer to the destination buffer.
        x, y, z, w: Four 32-bit registers containing the data to store.
        mask: Boolean predicate. If False, the store is skipped.
        multicast_op (tl.constexpr): Toggles between standard store (False)
        and multicast broadcast (True).
    """
    # PTX Assembly Logic:
    # 1. @$6: Predication. Only execute if argument 6 (mask) is True.
    # 2. Opcode Selection:
    #    - 'multimem.st...v4.f32': Broadcasts data to all peers.
    #      (Note: .f32 type used for bit-movement, equivalent to .u32 for storage).
    #    - 'st.global...v4.u32': Standard 128-bit memory write.
    # 3. Operands:
    #    - [$1]: Destination memory address.
    #    - {$2, $3, $4, $5}: Source registers containing data.
    if multicast_op:
        return tl.inline_asm_elementwise(
            """
            {
                .reg .pred %p0;
                setp.ne.s32 %p0, $6, 1;
                @%p0 bra end;
                multimem.st.relaxed.sys.global.v4.f32 [$1], {$2, $3, $4, $5};
                end:
            }
            """,
            "=r,l,r,r,r,r,r",
            args=[ptr, x, y, z, w, mask.to(tl.int32)],
            dtype=(tl.uint32),
            is_pure=False,
            pack=1,
        )
    else:
        return tl.inline_asm_elementwise(
            """
        {
            .reg .pred %p0;
            setp.ne.s32 %p0, $6, 1;
            @%p0 bra end;
            st.global.v4.f32 [$1], {$2, $3, $4, $5};
            end:
        }
        """,
            "=r,l,r,r,r,r,r",
            args=[ptr, x, y, z, w, mask.to(tl.int32)],
            dtype=(tl.uint32),
            is_pure=False,
            pack=1,
        )


@triton.jit
def add_v8_bf16_from_u32(
    a0,
    a1,
    a2,
    a3,  # First vector of 8 bf16s, packed in 4 uint32s
    b0,
    b1,
    b2,
    b3,  # Second vector of 8 bf16s, packed in 4 uint32s
):
    """
    Adds two vectors of 8 bfloat16 numbers.
    Each vector is passed as four tl.uint32 tensors.
    Returns the result as a tuple of four tl.uint32 tensors.
    """
    return tl.inline_asm_elementwise(
        """
        {
            add.bf16x2 $0, $4, $8;
            add.bf16x2 $1, $5, $9;
            add.bf16x2 $2, $6, $10;
            add.bf16x2 $3, $7, $11;
        }
        """,
        # 8 outputs (=r), 8 inputs (r)
        "=r,=r,=r,=r,r,r,r,r,r,r,r,r",
        args=[a0, a1, a2, a3, b0, b1, b2, b3],
        dtype=(tl.uint32, tl.uint32, tl.uint32, tl.uint32),
        is_pure=True,
        pack=1,
    )


@triton.jit
def ld_64(ptr, mask):
    """
    Loads 64 bits from local global memory into two 32-bit registers.

    Uses `ld.global.v2.u32`. Mirrors the non-multicast path of ld_128.

    Args:
        ptr: source pointer typed as uint64 (8-byte aligned).
        mask: boolean predicate — if False, the load is skipped.

    Returns:
        (x, y): two tl.uint32 registers containing 64 bits of loaded data.
    """
    return tl.inline_asm_elementwise(
        """
        {
            .reg .pred %p0;
            setp.ne.s32 %p0, $3, 1;
            @%p0 bra end;
            ld.global.v2.u32 {$0, $1}, [$2];
            end:
        }
        """,
        "=r,=r,l,r",
        args=[ptr, mask.to(tl.int32)],
        dtype=(tl.uint32, tl.uint32),
        is_pure=True,
        pack=1,
    )


@triton.jit
def st_64(ptr, x, y, mask, multicast_op: tl.constexpr):
    """
    Stores 64 bits (two 32-bit registers) to memory.

    Mirrors st_128 but operates on 64-bit (v2) quantities.

    1.  **Standard Store (`multicast_op=False`)**:
        -   `st.global.v2.f32` — writes 64 bits to local global memory.

    2.  **Multicast Store (`multicast_op=True`)**:
        -   `multimem.st.relaxed.sys.global.v2.f32` — broadcasts 64 bits to all
            peers in the multicast group simultaneously.

    Args:
        ptr: destination pointer typed as uint64 (8-byte aligned).
        x, y: two tl.uint32 registers containing the data to store.
        mask: boolean predicate — if False, the store is skipped.
        multicast_op (tl.constexpr): False = local store, True = multicast broadcast.
    """
    if multicast_op:
        return tl.inline_asm_elementwise(
            """
            {
                .reg .pred %p0;
                setp.ne.s32 %p0, $4, 1;
                @%p0 bra end;
                multimem.st.relaxed.sys.global.v2.f32 [$1], {$2, $3};
                end:
            }
            """,
            "=r,l,r,r,r",
            args=[ptr, x, y, mask.to(tl.int32)],
            dtype=(tl.uint32),
            is_pure=False,
            pack=1,
        )
    else:
        return tl.inline_asm_elementwise(
            """
            {
                .reg .pred %p0;
                setp.ne.s32 %p0, $4, 1;
                @%p0 bra end;
                st.global.v2.f32 [$1], {$2, $3};
                end:
            }
            """,
            "=r,l,r,r,r",
            args=[ptr, x, y, mask.to(tl.int32)],
            dtype=(tl.uint32),
            is_pure=False,
            pack=1,
        )


@triton.jit
def ld_32(ptr, mask):
    """
    Loads 32 bits from local global memory into one 32-bit register.

    Uses `ld.global.u32`. Scalar version of ld_64/ld_128.

    Args:
        ptr: source pointer typed as uint32 (4-byte aligned).
        mask: boolean predicate — if False, the load is skipped.

    Returns:
        x: one tl.uint32 register containing 32 bits of loaded data.
    """
    return tl.inline_asm_elementwise(
        """
        {
            .reg .pred %p0;
            setp.ne.s32 %p0, $2, 1;
            @%p0 bra end;
            ld.global.u32 $0, [$1];
            end:
        }
        """,
        "=r,l,r",
        args=[ptr, mask.to(tl.int32)],
        dtype=(tl.uint32,),
        is_pure=True,
        pack=1,
    )


@triton.jit
def st_32(ptr, x, mask, multicast_op: tl.constexpr):
    """
    Stores 32 bits (one 32-bit register) to memory.

    Scalar version of st_64/st_128.

    1.  **Standard Store (`multicast_op=False`)**:
        -   `st.global.f32` — writes 32 bits to local global memory.

    2.  **Multicast Store (`multicast_op=True`)**:
        -   `multimem.st.relaxed.sys.global.f32` — broadcasts 32 bits to all
            peers in the multicast group simultaneously.

    Args:
        ptr: destination pointer typed as uint32 (4-byte aligned).
        x: one tl.uint32 register containing the data to store.
        mask: boolean predicate — if False, the store is skipped.
        multicast_op (tl.constexpr): False = local store, True = multicast broadcast.
    """
    if multicast_op:
        return tl.inline_asm_elementwise(
            """
            {
                .reg .pred %p0;
                setp.ne.s32 %p0, $3, 1;
                @%p0 bra end;
                multimem.st.relaxed.sys.global.f32 [$1], $2;
                end:
            }
            """,
            "=r,l,r,r",
            args=[ptr, x, mask.to(tl.int32)],
            dtype=(tl.uint32),
            is_pure=False,
            pack=1,
        )
    else:
        return tl.inline_asm_elementwise(
            """
            {
                .reg .pred %p0;
                setp.ne.s32 %p0, $3, 1;
                @%p0 bra end;
                st.global.f32 [$1], $2;
                end:
            }
            """,
            "=r,l,r,r",
            args=[ptr, x, mask.to(tl.int32)],
            dtype=(tl.uint32),
            is_pure=False,
            pack=1,
        )


@triton.jit
def asm_rsqrt(x, eps):
    """
    Computes the reciprocal square root of a float32 number using inline assembly.
    """
    return tl.inline_asm_elementwise(
        """
        {
            add.f32 $1, $1, $2;
            rsqrt.approx.f32 $0, $1;
        }
        """,
        "=f, f, f",
        args=[x, eps],
        dtype=(tl.float32),
        is_pure=True,
        pack=1,
    )


# ── All-to-all-v peer-to-peer primitives ──────────────────────────────────────
# The all-gather-v / reduce-scatter-v kernels above move data with NVLink-SHARP
# `multimem` (one instruction fans out/reduces across every peer). The all-to-all-v
# kernels instead address ONE peer at a time via that peer's symmetric-memory base
# pointer (`_SymmetricMemory.buffer_ptrs_dev[dst]`), so they need plain (non-multicast)
# 128-bit load/store at *system* scope. System scope + the kernel's release/acquire
# barrier make the cross-GPU write/read visible; a `.gpu`/default-scope op would only
# be ordered within the local device.


@triton.jit
def st_128_p2p(ptr, x, y, z, w, mask):
    """Unicast 128-bit store to a single peer's symmetric buffer (system scope).

    Emits `st.relaxed.sys.global.v4.f32` (NOT `multimem.st`): writes the 128 bits to
    exactly the one remote address `ptr` (a peer buffer base + offset). Relaxed
    ordering is fine — the dispatch kernel's end barrier issues a `red.release.sys`
    that orders every prior p2p store before the arrival signal, so peers that
    acquire that signal observe the data. Mirrors the non-multicast `st_128`, only
    the scope qualifier changes (`.sys`), which is required for cross-GPU visibility.
    """
    return tl.inline_asm_elementwise(
        """
        {
            .reg .pred %p0;
            setp.ne.s32 %p0, $6, 1;
            @%p0 bra end;
            st.relaxed.sys.global.v4.f32 [$1], {$2, $3, $4, $5};
            end:
        }
        """,
        "=r,l,r,r,r,r,r",
        args=[ptr, x, y, z, w, mask.to(tl.int32)],
        dtype=(tl.uint32),
        is_pure=False,
        pack=1,
    )


@triton.jit
def ld_128_p2p(ptr, mask):
    """Unicast 128-bit load from a single peer's symmetric buffer (system scope).

    Emits `ld.relaxed.sys.global.v4.u32` (NOT `multimem.ld_reduce`): reads the 128 bits
    from exactly the one remote address `ptr` at system scope. Used by the all-to-all-v
    pull combine, which gathers only from a token's destination ranks (vs the RSV kernel's
    switch-reduce over *all* peers). Ordering/visibility is provided by the combine kernel's
    start barrier (`ld.acquire.sys` on the signal) plus system scope, so a relaxed load
    suffices; the same `.relaxed.sys...v4` scope/vector combo the multimem path already
    uses. is_pure=False so the compiler never hoists it above that barrier.
    """
    return tl.inline_asm_elementwise(
        """
        {
            .reg .pred %p0;
            setp.ne.s32 %p0, $5, 1;
            @%p0 bra end;
            ld.relaxed.sys.global.v4.u32 {$0, $1, $2, $3}, [$4];
            end:
        }
        """,
        "=r,=r,=r,=r,l,r",
        args=[ptr, mask.to(tl.int32)],
        dtype=(tl.uint32, tl.uint32, tl.uint32, tl.uint32),
        is_pure=False,
        pack=1,
    )
