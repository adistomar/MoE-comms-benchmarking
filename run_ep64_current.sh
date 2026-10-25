#!/bin/bash
# One task per GPU, 16 nodes (EP=64), e2e image, current Megatron kernels. At the e2e buffer
# size (2048 tokens per rank), all three dispatchers, B = 64 .. 128K:
#   1. communication only (identity expert)
#   2. dispatch + Megatron's vllm_fused_moe experts + combine
# Run inside the container on every task (srun --ntasks-per-node=4 ... bash run_ep64_current.sh).
# DeepEP uses its theoretical SM count (run.py's default, as Megatron does).
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
MEGATRON_PATH="${MEGATRON_PATH:-/lustre/fs1/portfolios/nemotron/projects/nemotron_sw_pre/users/aditomar/sidsingh-Megatron-LM}"
E2E_B=$(python3 -c 'print(",".join(str(2**i) for i in range(6, 18)))')  # 64 .. 128K
COMMON=(--model super --impl a2a,nvls,deepep --reps 20 --warmup 6 --timing graph
        --results-dir "results/${SLURM_JOB_ID}" --batch-sizes "${E2E_B}" --per-rank-cap 2048
        --megatron-path "${MEGATRON_PATH}")
say() { if [ "${SLURM_PROCID}" = "0" ]; then echo "$(date +%T) $*"; fi; }
say "communication only"
MASTER_PORT=29541 bash launch_rank.sh "${COMMON[@]}" --out comm.csv
say "dispatch + experts + combine"
MASTER_PORT=29542 bash launch_rank.sh "${COMMON[@]}" --expert vllm --out vllm.csv
say "done"
exit 0
