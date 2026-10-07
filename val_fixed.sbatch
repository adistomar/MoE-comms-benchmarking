#!/bin/bash
#SBATCH --job-name=moe-val-fixed
#SBATCH --partition=batch
#SBATCH --account=nemotron_sw_pre
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gpus-per-node=4
#SBATCH --time=0:40:00
#SBATCH --output=moe-val-fixed-%j.out
# Correctness after folding the send-index builder into the A2AV dispatch kernel and
# widening the recv-combine grid. Covers multi-segment cases (the timed large-step path).
set -uo pipefail
BENCH_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
CONTAINER_IMAGE=/lustre/fsw/portfolios/llmservice/users/ksanthanam/images/mcore_inference.sqsh
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
        python3 run.py --impl nvls,a2av,a2av_rs,a2av_push,dynamic \
            --batch-sizes 1,\$(( ${EP} * 549 )) --num-layers 1 --validate || exit 1
     "

# The Megatron kernel tests spawn their own 4-way torchrun, so they need a SINGLE task
# that owns all 4 GPUs -- not one task per GPU like the step above.
srun --ntasks=1 --cpus-per-task=36 --gpus-per-node=4 \
     --container-image "$CONTAINER_IMAGE" --container-mounts "/home:/home,/lustre:/lustre" \
     bash -lc "
        set -uo pipefail
        cd '$BENCH_DIR/..'
        export PYTHONPATH=\$PWD/Megatron-LM
        export TRITON_CACHE_DIR=/tmp/triton_cache_mg
        echo '=== Megatron kernel tests (fused builder + multi-segment) ==='
        torchrun --nproc-per-node 4 --master_port=$(( MASTER_PORT + 1 )) test_a2av_megatron_kernels.py || exit 1
        torchrun --nproc-per-node 4 --master_port=$(( MASTER_PORT + 2 )) test_dynamic_megatron_kernels.py || exit 1
     "
