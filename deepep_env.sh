#!/bin/bash
# Source THIS inside the NGC / mcore-inference container to set up the DeepEP-v2
# build + runtime environment. Idempotent.
#
#   source /path/to/bench/deepep_env.sh
#
# Only needed for the DeepEP path (--impl deepep/both). The NVLS path is fully
# vendored under bench/nvls/ and needs none of this.
#
# Paths are derived from this script's location, so the repo is relocatable.
# The DeepEP *source* is NOT part of this repo — clone it separately (see README)
# and point DEEPEP_DIR at it (default: a "DeepEP" checkout next to the repo root).
#
# Fast path: the editable install leaves the compiled extension at
# $DEEPEP_DIR/deep_ep/_C*.so (persists on shared storage). On later (ephemeral)
# containers we skip the ~15-min recompile and just reinstall the runtime
# NCCL/NVSHMEM wheels, recreate the unversioned .so symlinks, and set PYTHONPATH.
#
# CUDA-13 container, NCCL >= 2.30.4 (Gin / device symmetric-memory API). Gin is
# left ENABLED (it is the v2 transport; scaleout contexts are dormant on 1 node).

BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DEEPEP_DIR="${DEEPEP_DIR:-$(cd "$BENCH_DIR/.." 2>/dev/null && pwd)/DeepEP}"

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-10.0}"      # sm_100 (B200); 9.0 for H100
export EP_JIT_CACHE_DIR="${EP_JIT_CACHE_DIR:-$BENCH_DIR/.deepep_jit_cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$BENCH_DIR/.pip_cache}"
mkdir -p "$EP_JIT_CACHE_DIR" "$PIP_CACHE_DIR"
export EP_NCCL_ROOT_DIR="${EP_NCCL_ROOT_DIR:-/usr/local/lib/python3.12/dist-packages/nvidia/nccl}"
export EP_NVSHMEM_ROOT_DIR="${EP_NVSHMEM_ROOT_DIR:-/usr/local/lib/python3.12/dist-packages/nvidia/nvshmem}"
export LD_LIBRARY_PATH="$EP_NCCL_ROOT_DIR/lib:$EP_NVSHMEM_ROOT_DIR/lib:${LD_LIBRARY_PATH:-}"

if python3 -c 'import deep_ep' 2>/dev/null; then
    :  # already importable in this container
elif ls "$DEEPEP_DIR"/deep_ep/_C*.so >/dev/null 2>&1; then
    echo "[deepep_env] using prebuilt _C extension at $DEEPEP_DIR (fast path; no recompile)"
    # Guard the wheel install + symlink creation with a node-local file lock. With one task
    # per GPU (multi-node launch) every local rank sources this concurrently and would
    # otherwise race on the shared container site-packages -- a half-written nvshmem wheel
    # gets mmap'd and SIGBUSes run.py. flock serializes: the first rank installs, the rest
    # find the requirement already satisfied and the symlinks present, then fall through.
    # Single-node (sourced once) takes the lock immediately, so this is a no-op there.
    (
        flock 9
        python3 -m pip install -q --no-deps 'nvidia-nccl-cu13>=2.30.4' nvidia-nvshmem-cu13 >/dev/null 2>&1 || true
        for pair in "$EP_NCCL_ROOT_DIR/lib:libnccl.so" "$EP_NVSHMEM_ROOT_DIR/lib:libnvshmem_host.so"; do
            d="${pair%:*}"; n="${pair##*:}"
            if [ ! -e "$d/$n" ]; then
                cand=$(ls "$d/$n".* 2>/dev/null | sort | tail -1)
                [ -n "$cand" ] && ln -sf "$(basename "$cand")" "$d/$n"
            fi
        done
    ) 9>"${TMPDIR:-/tmp}/deepep_env_install.lock"
    export PYTHONPATH="$DEEPEP_DIR:${PYTHONPATH:-}"
else
    echo "[deepep_env] no prebuilt extension found; building DeepEP from $DEEPEP_DIR"
    DEEPEP_DIR="$DEEPEP_DIR" bash "$BENCH_DIR/install_deepep_ngc.sh"
fi
