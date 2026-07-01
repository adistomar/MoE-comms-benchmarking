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
# Full DeepEP-v2 (A2A) vs Megatron-NVLS (AGv/RSv) decode benchmark: one full
# 88-MoE-layer decode step, swept over global batch size, then plotted.
#
# ADJUST for your cluster: the #SBATCH --account/--partition/--qos above, the
# CONTAINER_IMAGE and --container-mounts below, and DEEPEP_DIR (the DeepEP source
# checkout — see README). Then submit FROM the bench directory:
#     cd bench && sbatch run.sbatch

set -uo pipefail
BENCH_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/lustre/fsw/portfolios/llmservice/users/ksanthanam/images/mcore_inference.sqsh}"
export DEEPEP_DIR="${DEEPEP_DIR:-$(cd "$BENCH_DIR/.." && pwd)/DeepEP}"

srun --container-image "$CONTAINER_IMAGE" \
     --container-mounts "/home:/home,/lustre:/lustre" \
     bash -lc "
        set -uo pipefail
        cd '$BENCH_DIR'
        export DEEPEP_DIR='$DEEPEP_DIR'
        source ./deepep_env.sh
        cd '$BENCH_DIR'
        torchrun --nproc_per_node=4 run.py --impl both \
            --batch-sizes 1,2,4,8,16,32,64,128 --deepep-num-sms 16 --num-layers 88 \
            --reps 20 --warmup 6 --timing graph --out results.csv
        python3 -m pip install -q matplotlib 2>/dev/null || true
        python3 plot_results.py --csv results.csv --x B --out results.png
     "
