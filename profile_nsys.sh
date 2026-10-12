#!/bin/bash
# Nsight Systems profile of rank 0, one nsys run per batch size (so kernel summaries are not
# mixed across sizes). Every other rank runs the same benchmark without nsys. Launch like
# run.sbatch / launch_rank.sh: one Slurm task per GPU.
#
#   bash profile_nsys.sh <impls> <comma-separated global batch sizes> <tag>
#
# Rank-0 outputs, per batch size B:
#   profiles/<tag>_B<B>.nsys-rep                   (open in the Nsight Systems GUI)
#   profiles/<tag>_B<B>_cuda_gpu_kern_sum.csv      (per-kernel time summary)
# The per-layer latencies printed by run.py (job stdout) give the wall time to compare the
# kernel sums against; the difference is launch / inter-kernel gaps.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

IMPLS="$1"
BATCHES="$2"
TAG="$3"
OUT_DIR="$PWD/profiles"
mkdir -p "$OUT_DIR"
NSYS="$(command -v nsys || ls /usr/local/cuda*/bin/nsys 2>/dev/null | head -1)"

port="${MASTER_PORT_BASE:-29600}"
for B in ${BATCHES//,/ }; do
    port=$((port + 1))
    args=(--impl "$IMPLS" --batch-sizes "$B" --num-layers 88 --reps 5 --warmup 2 --timing graph)
    if [ "${SLURM_PROCID}" = 0 ]; then
        MASTER_PORT=$port "$NSYS" profile -t cuda --cuda-graph-trace=node -f true \
            -o "$OUT_DIR/${TAG}_B${B}" bash launch_rank.sh "${args[@]}"
    else
        MASTER_PORT=$port bash launch_rank.sh "${args[@]}"
    fi
done

if [ "${SLURM_PROCID}" = 0 ]; then
    for B in ${BATCHES//,/ }; do
        "$NSYS" stats -q -r cuda_gpu_kern_sum -f csv -o "$OUT_DIR/${TAG}_B${B}" \
            "$OUT_DIR/${TAG}_B${B}.nsys-rep" >/dev/null 2>&1
        echo "[profile_nsys] wrote $OUT_DIR/${TAG}_B${B}_cuda_gpu_kern_sum.csv"
    done
fi
