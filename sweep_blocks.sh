#!/bin/bash
# Block-count sweep of NVLS (--nvls-blocks), DeepEP (--deepep-num-sms) and A2A-V (--a2a-blocks),
# no profiler. Each impl runs in its own run.py process under a timeout, so a hang in one impl
# cannot lose the others' results. Launch like run.sbatch / launch_rank.sh: one Slurm task
# per GPU.
#
#   bash sweep_blocks.sh <tag> <global batch sizes> <deepep num_sms list> <nvls/a2a blocks list>
#
# Outputs (rank 0): results/<tag>_{deepep,nvls,a2a}.csv; plot all three with
#   python3 plot_block_scaling.py --csv <tag>_deepep.csv,<tag>_nvls.csv,<tag>_a2a.csv
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

TAG="$1"
BATCHES="$2"
DEEPEP_SMS="$3"
BLOCKS="$4"
IMPL_TIMEOUT="${IMPL_TIMEOUT:-480}"   # seconds per impl

port="${MASTER_PORT_BASE:-29510}"
for impl in deepep nvls a2a; do
    port=$((port + 1))
    MASTER_PORT=$port timeout --kill-after=15 "$IMPL_TIMEOUT" bash launch_rank.sh \
        --impl "$impl" --batch-sizes "$BATCHES" --deepep-num-sms "$DEEPEP_SMS" \
        --nvls-blocks "$BLOCKS" --a2a-blocks "$BLOCKS" --num-layers 88 --reps 10 --warmup 3 \
        --timing graph --out "${TAG}_${impl}.csv"
    rc=$?
    [ "$rc" -ne 0 ] && echo "[sweep_blocks] rank ${SLURM_PROCID}: impl=$impl exited rc=$rc"
done
exit 0
