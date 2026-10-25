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
# Unified single-/multi-node decode benchmark of the dispatchers (DeepEP EPBuffer, Megatron
# NVLS AGv/RSv multicast, optionally Megatron NCCL AllGather via IMPL=all, A2A-V via IMPL
# containing a2a): one full decode step (all of the model's MoE layers) swept over global batch
# size, then plotted. MODEL picks the Nemotron-3 preset (models.py: super = 40 MoE layers,
# latent 1024; ultra = 48 MoE layers, latent 2048). EXPERT=vllm adds Megatron's vllm_fused_moe
# between dispatch and combine (needs MEGATRON_PATH, a Megatron-LM checkout). Same script for
# 1 node or N nodes. CSVs go to results/, plots to plots/.
#
# Launch model: one srun task PER GPU (--ntasks-per-node=4), each task = one EP rank running
# launch_rank.sh -> run.py. EP = world size = 4 * nodes, AUTOMATICALLY (run.py sets
# Config.ep_size = world). The container image ships DeepEP >= 2.5 (build_deepep_image.sh).
#
#   SINGLE NODE:   sbatch run.sbatch                         # -> results/results_ultra.csv
#                  MODEL=super sbatch run.sbatch             # -> results/results_super.csv
#                  MODEL=super EXPERT=vllm MEGATRON_PATH=/path/to/Megatron-LM sbatch run.sbatch
#                                                            # -> results/results_super_vllm.csv
#   MULTI-NODE (within one NVL72 rack): pass BOTH -N and --segment (equal, n<=18):
#       sbatch -N 4  --segment=4  run.sbatch                 # -> results/results_ultra_4n.csv
#       sbatch -N 8  --segment=8  run.sbatch                 # -> results/results_ultra_8n.csv
#       IMPL=all sbatch -N 16 --segment=16 run.sbatch        # also runs NCCL
#
# --segment=<n> pins all requested nodes into the SAME physical NVL72 rack so cross-node
# dispatch/combine traffic rides NVLink (NVSwitch), not InfiniBand; it MUST equal -N.
# Everything downstream derives from $SLURM_JOB_NUM_NODES. EP=4*nodes must divide
# num_experts=512, so node counts 1/2/4/8/16 are valid.
#
# ADJUST for your cluster: the #SBATCH --account/--partition/--qos above, CONTAINER_IMAGE
# and --container-mounts below. Submit FROM the repo directory.

set -uo pipefail
BENCH_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/lustre/fsw/portfolios/nemotron/users/aditomar/images/mcore_inference+fa4+deepep.sqsh}"

# rank 0 hosts the c10d store (torch env:// init) on the first node in the allocation;
# derive it from scontrol (idiom from inference-bench/reasoning/vllm/slurm_vllm_profile.sh
# :116-121). A fixed port is safe since --exclusive dedicates each node. NNODES/MASTER_*
# expand HERE (submit node) so all ranks get the same value; SLURM_PROCID/NTASKS/LOCALID
# are escaped so they expand per-task inside srun.
NNODES="${SLURM_JOB_NUM_NODES}"
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
MASTER_PORT="${MASTER_PORT:-29500}"
IMPL="${IMPL:-both}"        # both = deepep + nvls; all = + nccl; or a list, e.g. deepep,nvls,a2a
MODEL="${MODEL:-ultra}"     # models.py preset: ultra | super (sets shape and MoE layer count)
EXPERT="${EXPERT:-identity}"  # identity (communication only) | vllm (Megatron's experts)
EXPERT_ARGS="--expert ${EXPERT}"
OUT="results_${MODEL}"
# Global batch sizes, 1 .. 1M. With experts, up to 128K: vllm_fused_moe's intermediates scale
# with B x top-k (about 170 GB per GPU at B=1M for Super).
B_MAX_LOG2=20
if [ "$EXPERT" != "identity" ]; then B_MAX_LOG2=17; fi
BATCH_SIZES="${BATCH_SIZES:-$(python3 -c "print(','.join(str(2**i) for i in range(${B_MAX_LOG2} + 1)))")}"
if [ "$EXPERT" != "identity" ]; then
    : "${MEGATRON_PATH:?EXPERT=${EXPERT} needs MEGATRON_PATH (a Megatron-LM checkout)}"
    EXPERT_ARGS="${EXPERT_ARGS} --megatron-path ${MEGATRON_PATH}"
    OUT="${OUT}_${EXPERT}"
fi
if [ "$NNODES" -gt 1 ]; then OUT="${OUT}_${NNODES}n"; fi
echo "launch: NNODES=${NNODES}  ranks=$(( NNODES * 4 ))  EP=$(( NNODES * 4 ))  model=${MODEL}  impl=${IMPL}  expert=${EXPERT}  MASTER=${MASTER_ADDR}:${MASTER_PORT}  out=results/${OUT}.csv"

# One task per GPU: NNODES*4 ranks total, 4 per node. Each task runs run.py as one rank.
# CSVs land in results/, plots in plots/.
srun --ntasks="$(( NNODES * 4 ))" --ntasks-per-node=4 --cpus-per-task=36 \
     --container-image "$CONTAINER_IMAGE" \
     --container-mounts "/home:/home,/lustre:/lustre" \
     bash -lc "
        set -uo pipefail
        cd '$BENCH_DIR'
        export MASTER_ADDR='${MASTER_ADDR}' MASTER_PORT='${MASTER_PORT}'
        bash launch_rank.sh --model ${MODEL} --impl ${IMPL} ${EXPERT_ARGS} \
            --batch-sizes ${BATCH_SIZES} \
            --reps 20 --warmup 6 --timing graph \
            --out ${OUT}.csv
        if [ \"\${SLURM_PROCID}\" = \"0\" ]; then
            python3 -m pip install -q matplotlib 2>/dev/null || true
            python3 plot_results.py --csv ${OUT}.csv --x B --out ${OUT}.png
        fi
     "
