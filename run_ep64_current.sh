#!/bin/bash
# One task per GPU, 16 nodes (EP=64), e2e image, current Megatron kernels. At the e2e buffer
# size (2048 tokens per rank), all three dispatchers, B = 64 .. 128K:
#   1. communication only (identity expert)
#   2. dispatch + Megatron's vllm_fused_moe experts + combine
set -u
REPO=/lustre/fs1/portfolios/nemotron/projects/nemotron_sw_pre/users/aditomar/sidsingh-Megatron-LM
cd "${REPO}/moe-bench/experiments/moe_comms_bench"
export RANK="${SLURM_PROCID}" WORLD_SIZE="${SLURM_NTASKS}" LOCAL_RANK="${SLURM_LOCALID}"
export TRITON_CACHE_DIR="/tmp/triton_cache_rank${RANK}" EP_DISABLE_GIN=1
E2E_B=$(python3 -c 'print(",".join(str(2**i) for i in range(6, 18)))')  # 64 .. 128K
COMMON=(--model super --impl a2a,nvls,deepep --reps 20 --warmup 6 --timing graph
        --results-dir "results/${SLURM_JOB_ID}" --batch-sizes "${E2E_B}" --per-rank-cap 2048
        --deepep-num-sms auto --megatron-path "${REPO}")
say() { if [ "${RANK}" = "0" ]; then echo "$(date +%T) $*"; fi; }
say "communication only"
MASTER_PORT=29541 python3 run.py "${COMMON[@]}" --out comm.csv
say "dispatch + experts + combine"
MASTER_PORT=29542 python3 run.py "${COMMON[@]}" --expert vllm --out vllm.csv
say "done"
exit 0
