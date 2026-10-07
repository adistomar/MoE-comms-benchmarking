#!/bin/bash
#SBATCH --job-name=moe-val-multiseg
#SBATCH --partition=batch
#SBATCH --account=nemotron_sw_pre
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gpus-per-node=4
#SBATCH --time=0:40:00
#SBATCH --output=moe-val-multiseg-%j.out
# Correctness of the MULTI-SEGMENT path, which the previous validation never reached
# (every case had local_tokens <= SEG=256, i.e. one segment). The timed large-step
# numbers all run multi-segment, so this has to pass before those numbers mean anything.
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
        # max batch sets per_rank_cap, so ask for enough capacity for the 3-segment case.
        python3 run.py --impl nvls,a2av_push,dynamic \
            --batch-sizes 1,\$(( ${EP} * 549 )) --num-layers 1 --validate
     "
