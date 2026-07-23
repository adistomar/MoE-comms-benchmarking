#!/bin/bash
#SBATCH --job-name=hier-mnode
#SBATCH --partition=batch
#SBATCH --account=nemotron_sw_pre
#SBATCH --qos=short
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --gpus-per-node=4
#SBATCH --time=0:25:00
#SBATCH --output=hier-mnode-%j.out
#
# Multi-node hierarchical test/bench. EP = (#nodes)*4. Override #nodes with `sbatch
# --nodes=N test_multinode.sbatch` (EP=8 -> 2 nodes, EP=64 -> 16 nodes).
#
# IMPORTANT: cross-node symmetric memory / NVLink multicast require ALL nodes to sit
# in ONE NVLink (NVL72) domain. If your scheduler can place nodes outside one NVL72,
# add the appropriate topology/block constraint here (cluster-specific) or the NVLS
# and hier symmetric-memory init will fail. This benchmark was originally validated
# single-node (EP=4); the 2-node run is the de-risk for cross-node symmetric memory.
#
# Override the command with TEST_CMD, e.g.:
#   sbatch --nodes=2 test_multinode.sbatch                     # EP=8 validate (default cmd)
#   TEST_CMD="run.py --impl nvls,nccl,hier --hier-g 2 --batch-sizes 1,16,128 --num-layers 88 \
#     --reps 20 --warmup 6 --timing eager --out results_hier_ep$((SLURM_NNODES*4)).csv" \
#     sbatch --nodes=16 test_multinode.sbatch                  # EP=64 sequential sweep

set -uo pipefail
BENCH_DIR="${BENCH_DIR:-/lustre/fsw/portfolios/nemotron/projects/nemotron_sw_pre/users/psinghania/MoE-comms-benchmarking}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/lustre/fsw/portfolios/nemotron/projects/nemotron_sw_pre/users/psinghania/gtp_latest/gtp_latest.sqsh}"
DEEPEP_DIR="${DEEPEP_DIR:-/lustre/fsw/portfolios/nemotron/projects/nemotron_sw_pre/users/psinghania/DeepEP}"
TEST_CMD="${TEST_CMD:-run.py --impl nvls,hier --hier-g 4 --validate}"
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_ADDR
# Set SOURCE_DEEPEP=1 at submit to source deepep_env.sh on each node (needed for --impl deepep).
export SOURCE_DEEPEP DEEPEP_DIR

srun --container-image "$CONTAINER_IMAGE" \
     --container-mounts "/home:/home,/lustre:/lustre" \
     bash -lc "
        set -uo pipefail
        cd '$BENCH_DIR'
        echo \"# node \$SLURM_NODEID of $SLURM_NNODES (master $MASTER_ADDR)\"
        if [ -n \"\${SOURCE_DEEPEP:-}\" ]; then
            export DEEPEP_DIR='$DEEPEP_DIR' EP_SUPPRESS_NCCL_CHECK=1
            export PYTHONPATH=\"\$DEEPEP_DIR:\${PYTHONPATH:-}\"   # force elastic checkout ahead of container deep_ep
            # deepep_env.sh reinstalls the runtime NCCL wheel; pin it to 2.30.4 (matches torch and
            # the DeepEP build) so the runtime NCCL version assertion passes.
            export NCCL_PACKAGE='nvidia-nccl-cu13==2.30.4'
            source ./deepep_env.sh
            # GIN enablement — mirrors gtp_benchmark/nemotron/a55b/set_env.sh (DeepEP-v2 NCCL-Gin device comm).
            export HYBRID_EP_CACHE_DIR=/tmp/hybrid_ep_cache_\${SLURM_NODEID:-0}; mkdir -p \"\$HYBRID_EP_CACHE_DIR\"
            export NCCL_NET=IB NCCL_NVLS_ENABLE=1 NCCL_GIN_ENABLE=1 NCCL_GIN_TYPE=2
            export NCCL_IB_DATA_DIRECT=1 NCCL_DMABUF_ENABLE=1 NCCL_SYM_GIN_KERNELS_ENABLE=1 NCCL_NET_MERGE_LEVEL=LOC
            export NCCL_SYM_CE_THRESHOLD=\${NCCL_SYM_CE_THRESHOLD:-1099511627776}
            # Custom GIN-enabled NCCL 2.30.7 (has the GIN transport; default ON for the DeepEP path).
            if [ \"\${USE_CUSTOM_NCCL:-1}\" = 1 ]; then
                CN=/lustre/fsw/portfolios/nemotron/projects/nemotron_sw_pre/users/psinghania/nccl/build/lib
                export LD_LIBRARY_PATH=\"\$CN:\${LD_LIBRARY_PATH:-}\"
                export LD_PRELOAD=\"\$CN/libnccl.so.2:\${LD_PRELOAD:-}\"
                echo \"# using custom GIN NCCL 2.30.7 (LD_PRELOAD)\"
            fi
        fi
        torchrun --nnodes=$SLURM_NNODES --nproc-per-node=4 \
                 --node-rank=\$SLURM_NODEID \
                 --master-addr=$MASTER_ADDR --master-port=29501 \
                 $TEST_CMD
     "
