#!/bin/bash
#SBATCH --job-name=moe-comms-bench
#SBATCH --partition=batch
#SBATCH --account=nemotron_sw_pre
#SBATCH --qos=interactive
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gpus-per-node=4
#SBATCH --time=1:00:00
#SBATCH --output=moe-comms-bench-%j.out
#
# Unified single-/multi-node decode benchmark of all three dispatchers (DeepEP-v2 A2A,
# Megatron NVLS AGv/RSv multicast, Megatron NCCL AllGather): one full 88-MoE-layer decode
# step swept over global batch size, then plotted. Same script for 1 node or N nodes.
#
# Launch model: one srun task PER GPU (--ntasks-per-node=4), each task = one EP rank
# running run.py. EP = world size = 4 * nodes, AUTOMATICALLY (run.py sets Config.ep_size
# = world). Concurrent per-rank `source deepep_env.sh` is safe: the wheel install is
# flock-guarded (see deepep_env.sh).
#
#   SINGLE NODE:   cd bench && sbatch run.sbatch                       # -> results.csv
#   MULTI-NODE (within one NVL72 rack): pass BOTH -N and --segment (equal, n<=18):
#       cd bench && sbatch -N 4  --segment=4  run.sbatch              # -> results_4n.csv
#       cd bench && sbatch -N 8  --segment=8  run.sbatch              # -> results_8n.csv
#       cd bench && sbatch -N 16 --segment=16 run.sbatch              # -> results_16n.csv
#
# --segment=<n> pins all requested nodes into the SAME physical NVL72 rack so cross-node
# dispatch/combine traffic rides NVLink (NVSwitch), not InfiniBand; it MUST equal -N.
# Everything downstream derives from $SLURM_JOB_NUM_NODES. EP=4*nodes must divide
# num_experts=512, so node counts 1/2/4/8/16/32 are valid.
#
# ADJUST for your cluster: the #SBATCH --account/--partition/--qos above, CONTAINER_IMAGE
# and --container-mounts below, and DEEPEP_DIR (the DeepEP source checkout). Submit FROM
# the bench directory.

set -uo pipefail
BENCH_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/lustre/fsw/portfolios/llmservice/users/ksanthanam/images/mcore_inference.sqsh}"
export DEEPEP_DIR="${DEEPEP_DIR:-$(cd "$BENCH_DIR/.." && pwd)/DeepEP}"

# rank 0 hosts the c10d store (torch env:// init) on the first node in the allocation;
# derive it from scontrol (idiom from inference-bench/reasoning/vllm/slurm_vllm_profile.sh
# :116-121). A fixed port is safe since --exclusive dedicates each node. NNODES/MASTER_*
# expand HERE (submit node) so all ranks get the same value; SLURM_PROCID/NTASKS/LOCALID
# are escaped so they expand per-task inside srun.
NNODES="${SLURM_JOB_NUM_NODES}"
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
MASTER_PORT="${MASTER_PORT:-29500}"
if [ "$NNODES" -eq 1 ]; then OUT="results"; else OUT="results_${NNODES}n"; fi
echo "launch: NNODES=${NNODES}  ranks=$(( NNODES * 4 ))  EP=$(( NNODES * 4 ))  MASTER=${MASTER_ADDR}:${MASTER_PORT}  out=${OUT}.csv"

# One task per GPU: NNODES*4 ranks total, 4 per node. Each task runs run.py as one rank.
srun --ntasks="$(( NNODES * 4 ))" --ntasks-per-node=4 --cpus-per-task=36 \
     --container-image "$CONTAINER_IMAGE" \
     --container-mounts "/home:/home,/lustre:/lustre" \
     bash -lc "
        set -uo pipefail
        cd '$BENCH_DIR'
        export DEEPEP_DIR='$DEEPEP_DIR'
        export MASTER_ADDR='${MASTER_ADDR}' MASTER_PORT='${MASTER_PORT}'
        export RANK=\${SLURM_PROCID} WORLD_SIZE=\${SLURM_NTASKS} LOCAL_RANK=\${SLURM_LOCALID}
        export TRITON_CACHE_DIR=/tmp/triton_cache_rank\${SLURM_PROCID}
        source ./deepep_env.sh
        cd '$BENCH_DIR'
        python3 run.py --impl all \
            --batch-sizes 1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536,131072,262144,524288,1048576 \
            --deepep-num-sms 148 --num-layers 88 \
            --reps 20 --warmup 6 --timing graph \
            --out ${OUT}.csv
        if [ \"\${SLURM_PROCID}\" = \"0\" ]; then
            python3 -m pip install -q matplotlib 2>/dev/null || true
            python3 plot_results.py --csv ${OUT}.csv --x B --out ${OUT}.png
        fi
     "
