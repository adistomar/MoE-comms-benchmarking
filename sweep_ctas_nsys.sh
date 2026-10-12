#!/bin/bash
# CTA-count sweep for NVLS (--nvls-blocks), DeepEP (--deepep-num-sms) and A2A-V (--a2a-blocks),
# with rank 0 under Nsight Systems so each (impl, CTA count, B) config is split into its kernels
# via the NVTX range run.py opens around it. Launch like run.sbatch / launch_rank.sh: one Slurm
# task per GPU.
#
#   bash sweep_ctas_nsys.sh <tag> <global batch sizes> <deepep num_sms list> <a2a blocks list> \
#       [<nvls blocks list>] [<impls>]
#
# NVLS blocks default to the A2A list; impls default to nvls,deepep,a2a. Rank-0 outputs:
#   results/<tag>.csv                     per-config decode-step latency (max across ranks)
#   profiles/<tag>.nsys-rep, .sqlite      trace; summarize with analyze_nsys_sweep.py
#   results/<tag>_kernels.csv             per-config kernel medians (analyze_nsys_sweep.py)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

TAG="$1"
BATCHES="$2"
DEEPEP_SMS="$3"
A2A_BLOCKS="$4"
NVLS_BLOCKS="${5:-$A2A_BLOCKS}"
IMPLS="${6:-nvls,deepep,a2a}"
OUT_DIR="$PWD/profiles"
mkdir -p "$OUT_DIR"
NSYS="$(command -v nsys || ls /usr/local/cuda*/bin/nsys 2>/dev/null | head -1)"

args=(--impl "$IMPLS" --batch-sizes "$BATCHES" --deepep-num-sms "$DEEPEP_SMS"
      --nvls-blocks "$NVLS_BLOCKS" --a2a-blocks "$A2A_BLOCKS" --num-layers 88 --reps 10
      --warmup 3 --timing graph --out "${TAG}.csv")
if [ "${SLURM_PROCID}" = 0 ]; then
    "$NSYS" profile -t cuda,nvtx --cuda-graph-trace=node -f true -o "$OUT_DIR/$TAG" \
        bash launch_rank.sh "${args[@]}"
    "$NSYS" export -t sqlite -f true -o "$OUT_DIR/$TAG.sqlite" "$OUT_DIR/$TAG.nsys-rep" \
        && echo "[sweep_ctas_nsys] wrote $OUT_DIR/$TAG.sqlite"
    python3 analyze_nsys_sweep.py "$OUT_DIR/$TAG.sqlite" --csv "results/${TAG}_kernels.csv"
else
    bash launch_rank.sh "${args[@]}"
fi
