#!/bin/bash
#SBATCH --job-name=moe-dyn-bench
#SBATCH --partition=batch
#SBATCH --account=nemotron_sw_pre
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gpus-per-node=4
#SBATCH --time=1:45:00
#SBATCH --output=moe-dyn-bench-%j.out
#
# AGv/RSv vs A2Av vs regime-adaptive ("dynamic") MoE dispatch/combine, swept over step
# size, at whatever EP the allocation gives. Runs TWO things per job:
#
#   1. run.py --impl nvls,a2av_push,dynamic   -> results_dynamic_${N}n.csv   (+ .png)
#      the three-way comparison a user cares about.
#   2. bench_phase_breakdown.py               -> phase_dynamic_${N}n.csv     (+ .png)
#      dispatch and combine timed SEPARATELY for nvls and for each PINNED branch of
#      dynamic. This is what locates the two crossovers, and they MOVE WITH EP: all-to-all-v
#      saves wire bytes in proportion to avg_dest/ep_size, so the larger the EP the earlier
#      it should win. The in-tree thresholds were measured at EP=4, so the `dynamic` curve
#      in (1) is only optimal at EP=4 — (2) is what says where it should switch here.
#
# Everything runs with --out-dtype fp32: that is the expert-output buffer Megatron's
# fused-MoE unpermute actually produces, and the dtype the in-tree thresholds target.
# CAVEAT when reading DeepEP against these: --out-dtype only applies to the impls that
# expose the knob (nvls, dynamic). DeepEP and a2av_push move bf16 on the combine wire
# either way, so they carry a 2x dtype advantage on the combine leg specifically.
#
# Launch model: one srun task PER GPU (--ntasks-per-node=4), each task = one EP rank.
# EP = world size = 4 * nodes, automatically (run.py sets Config.ep_size = world).
#
#   SINGLE NODE:  sbatch run_dynamic.sbatch                      # EP=4
#   MULTI-NODE (one NVL72 rack): pass BOTH -N and --segment, equal:
#       sbatch -N 4  --segment=4  run_dynamic.sbatch             # EP=16
#       sbatch -N 8  --segment=8  run_dynamic.sbatch             # EP=32
#       sbatch -N 16 --segment=16 run_dynamic.sbatch             # EP=64
#
# --segment=<n> pins the nodes into the SAME physical NVL72 rack so cross-node traffic
# rides NVLink/NVSwitch rather than InfiniBand; it MUST equal -N. EP=4*nodes must divide
# num_experts=512, and EP<=64 because destination ranks are held in a uint64 bitmask.
# Submit FROM the bench directory.

set -uo pipefail
BENCH_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/lustre/fsw/portfolios/llmservice/users/ksanthanam/images/mcore_inference.sqsh}"
# DeepEP source checkout (sibling of this repo). Its compiled _C extension persists on
# shared storage, so deepep_env.sh takes the fast path (wheel reinstall + symlinks only).
export DEEPEP_DIR="${DEEPEP_DIR:-$(cd "$BENCH_DIR/.." && pwd)/DeepEP}"
IMPLS="${IMPLS:-nvls,a2av_push,dynamic,deepep}"

NNODES="${SLURM_JOB_NUM_NODES}"
EP=$(( NNODES * 4 ))
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
# Derive the port from the job id so concurrently-running EP sweeps never collide.
MASTER_PORT="${MASTER_PORT:-$(( 29000 + SLURM_JOB_ID % 2000 ))}"
# TAG lets a re-run (e.g. after retuning thresholds) land beside the original instead of
# overwriting it. SKIP_PHASE=1 skips the per-phase breakdown, which only needs to be
# measured once per EP -- it characterises the two BRANCHES, not the threshold choice.
TAG="${TAG:-}"
OUT="results_dynamic_${NNODES}n${TAG}"
PHASE="phase_dynamic_${NNODES}n${TAG}"
SKIP_PHASE="${SKIP_PHASE:-0}"

# Step sizes = GLOBAL tokens across all EP ranks. Kept identical across EP so the curves
# are directly comparable; note the same global count means fewer tokens per rank as EP
# grows, so the small end gets progressively more launch-bound at high EP.
BATCHES="16,64,256,512,1024,2048,4096,8192,16384,32768,65536"
PHASE_BATCHES="256,1024,2048,4096,8192,16384,32768,65536"

echo "launch: NNODES=${NNODES} ranks=${EP} EP=${EP} MASTER=${MASTER_ADDR}:${MASTER_PORT}"
echo "        out=${OUT}.csv  phase=${PHASE}.csv"

srun --ntasks="${EP}" --ntasks-per-node=4 --cpus-per-task=36 \
     --container-image "$CONTAINER_IMAGE" \
     --container-mounts "/home:/home,/lustre:/lustre" \
     bash -lc "
        set -uo pipefail
        cd '$BENCH_DIR'
        export MASTER_ADDR='${MASTER_ADDR}' MASTER_PORT='${MASTER_PORT}'
        export RANK=\${SLURM_PROCID} WORLD_SIZE=\${SLURM_NTASKS} LOCAL_RANK=\${SLURM_LOCALID}
        export TRITON_CACHE_DIR=/tmp/triton_cache_rank\${SLURM_PROCID}
        export DEEPEP_DIR='$DEEPEP_DIR'
        # flock-guarded inside: with one task per GPU every local rank sources this
        # concurrently and would otherwise race on the shared container site-packages.
        case \"${IMPLS}\" in *deepep*) source ./deepep_env.sh ;; esac

        # Correctness first: a wrong kernel makes every latency number meaningless, and at
        # EP>4 the destination-bitmask and per-peer loops are exercised far harder.
        python3 run.py --impl ${IMPLS} \
            --batch-sizes 1,\$(( ${EP} * 2 )) --num-layers 1 --validate || exit 1

        python3 run.py --impl ${IMPLS} --out-dtype fp32 --deepep-num-sms 148 \
            --batch-sizes ${BATCHES} --num-layers 88 \
            --reps 20 --warmup 6 --timing graph --out ${OUT}.csv || exit 1

        if [ \"${SKIP_PHASE}\" != \"1\" ]; then
            python3 bench_phase_breakdown.py --out-dtype fp32 \
                --batch-sizes ${PHASE_BATCHES} --num-layers 88 \
                --reps 20 --warmup 6 --out ${PHASE}.csv || exit 1
        fi

        if [ \"\${SLURM_PROCID}\" = \"0\" ]; then
            python3 -m pip install -q matplotlib 2>/dev/null || true
            python3 plot_dynamic_compare.py --csv ${OUT}.csv --out ${OUT}.png \
                --dispatch-threshold 4096 --combine-threshold 2048 || true
            [ \"${SKIP_PHASE}\" = \"1\" ] || python3 plot_phase_breakdown.py --csv ${PHASE}.csv --out ${PHASE}.png \
                --dispatch-threshold 4096 --combine-threshold 2048 \
                --title 'MoE dispatch and combine, measured separately — EP=${EP}, ${NNODES} node(s) in one NVL72 rack, 512 experts, top-k=22, hidden=1024, fp32 expert-output buffer' || true
        fi
     "
