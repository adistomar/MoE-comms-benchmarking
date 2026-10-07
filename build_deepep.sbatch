#!/bin/bash
#SBATCH --job-name=deepep-build
#SBATCH --partition=batch
#SBATCH --account=nemotron_sw_pre
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gpus-per-node=4
#SBATCH --time=1:00:00
#SBATCH --output=deepep-build-%j.out
#
# Rebuild the DeepEP _C extension against THIS container's NCCL.
#
# Below NCCL 2.31, DeepEP asserts that its compile-time and runtime NCCL versions match
# exactly (csrc/kernels/backend/nccl.cu). The extension cached in the DeepEP checkout was
# built against an older container image, so every rank now dies with
# "NCCL_VERSION_CODE == nccl_runtime_version ... Please re-compile DeepEP."
# deepep_env.sh only rebuilds when no _C*.so is present, so the stale one is renamed
# aside before submitting this.
#
# The resulting .so lands in $DEEPEP_DIR (shared storage) and is picked up by every later
# benchmark job via deepep_env.sh's fast path, so this only needs to run once per image.
set -uo pipefail
BENCH_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/lustre/fsw/portfolios/llmservice/users/ksanthanam/images/mcore_inference.sqsh}"
export DEEPEP_DIR="${DEEPEP_DIR:-$(cd "$BENCH_DIR/.." && pwd)/DeepEP}"

srun --ntasks=1 --cpus-per-task=36 \
     --container-image "$CONTAINER_IMAGE" \
     --container-mounts "/home:/home,/lustre:/lustre" \
     bash -lc "
        set -uo pipefail
        cd '$BENCH_DIR'
        export DEEPEP_DIR='$DEEPEP_DIR'
        source ./deepep_env.sh
        echo '--- verifying import + NCCL match ---'
        python3 -c 'import deep_ep, torch; print(\"deep_ep OK\", deep_ep.__file__); print(\"torch nccl\", torch.cuda.nccl.version())'
     "
