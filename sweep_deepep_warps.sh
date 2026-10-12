#!/bin/bash
# DeepEP dispatch-warp sweep: one run.py process per EP_DISPATCH_TOTAL_WARPS value (the total
# dispatch warps per GPU that DeepEP spreads over num_sms CTAs; upstream hardcodes 512). Needs a
# DeepEP build with patches/deepep_dispatch_total_warps.patch (build_deepep_variant.sh); point
# DEEPEP_DIR at it. Launch like launch_rank.sh: one Slurm task per GPU.
#
#   DEEPEP_DIR=../DeepEP-warps bash sweep_deepep_warps.sh <tag> <global B list> \
#       <num_sms list> <total-warps list>
#
# Outputs (rank 0): results/<tag>_w<warps>.csv per value.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

TAG="$1"
BATCHES="$2"
DEEPEP_SMS="$3"
WARPS="$4"
IMPL_TIMEOUT="${IMPL_TIMEOUT:-480}"   # seconds per value

port="${MASTER_PORT_BASE:-29530}"
for w in ${WARPS//,/ }; do
    port=$((port + 1))
    EP_DISPATCH_TOTAL_WARPS=$w MASTER_PORT=$port timeout --kill-after=15 "$IMPL_TIMEOUT" \
        bash launch_rank.sh --impl deepep --batch-sizes "$BATCHES" --deepep-num-sms "$DEEPEP_SMS" \
        --num-layers 88 --reps 10 --warmup 3 --timing graph --out "${TAG}_w${w}.csv"
    rc=$?
    [ "$rc" -ne 0 ] && echo "[sweep_deepep_warps] rank ${SLURM_PROCID}: warps=$w exited rc=$rc"
done
exit 0
