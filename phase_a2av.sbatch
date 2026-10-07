#!/bin/bash
#SBATCH --job-name=moe-phase-a2av
#SBATCH --partition=batch
#SBATCH --account=nemotron_sw_pre
#SBATCH --qos=short
#SBATCH --nodes=4
#SBATCH --exclusive
#SBATCH --gpus-per-node=4
#SBATCH --time=1:00:00
#SBATCH --output=moe-phase-a2av-%j.out
#
# Controlled comparison of the standalone A2Av implementation against the dynamic
# kernel's A2Av branch, split by phase. Run at --out-dtype BF16 so both move the same
# bytes: bench_a2av is bf16-only, and the dynamic kernel's fp32 mode reads twice the
# local bytes on the push, which would confound the structural difference being tested.
# The remaining difference is then structural only -- chiefly that the standalone version
# launches the send-index builder as its own kernel while the dynamic kernel folds it
# into dispatch.
set -uo pipefail
BENCH_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/lustre/fsw/portfolios/llmservice/users/ksanthanam/images/mcore_inference.sqsh}"
NNODES="${SLURM_JOB_NUM_NODES}"; EP=$(( NNODES * 4 ))
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
MASTER_PORT=$(( 29000 + SLURM_JOB_ID % 2000 ))
srun --ntasks="${EP}" --ntasks-per-node=4 --cpus-per-task=36 \
     --container-image "$CONTAINER_IMAGE" --container-mounts "/home:/home,/lustre:/lustre" \
     bash -lc "
        set -uo pipefail
        cd '$BENCH_DIR'
        export MASTER_ADDR='${MASTER_ADDR}' MASTER_PORT='${MASTER_PORT}'
        export RANK=\${SLURM_PROCID} WORLD_SIZE=\${SLURM_NTASKS} LOCAL_RANK=\${SLURM_LOCALID}
        export TRITON_CACHE_DIR=/tmp/triton_cache_rank\${SLURM_PROCID}
        python3 bench_phase_breakdown.py --out-dtype bf16 \
            --batch-sizes 256,1024,2048,4096,8192,16384,32768,65536 \
            --num-layers 88 --reps 20 --warmup 6 --out phase_a2av_${NNODES}n.csv
     "
