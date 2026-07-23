#!/bin/bash
#SBATCH --job-name=deepep-build
#SBATCH --partition=batch
#SBATCH --account=nemotron_sw_pre
#SBATCH --qos=interactive
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gpus-per-node=4
#SBATCH --time=0:45:00
#SBATCH --output=deepep-build-%j.out
#
# One-time DeepEP build in the container (~15 min first time; cached after). Sources
# deepep_env.sh (installs NCCL/NVSHMEM wheels, builds the _C extension) then smoke-tests
# `import deep_ep`. DeepEP source must already be cloned at DEEPEP_DIR (elastic commit).

set -uo pipefail
BENCH_DIR="/lustre/fsw/portfolios/nemotron/projects/nemotron_sw_pre/users/psinghania/MoE-comms-benchmarking"
CONTAINER_IMAGE="/lustre/fsw/portfolios/nemotron/projects/nemotron_sw_pre/users/psinghania/gtp_latest/gtp_latest.sqsh"
export DEEPEP_DIR="/lustre/fsw/portfolios/nemotron/projects/nemotron_sw_pre/users/psinghania/DeepEP"

srun --container-image "$CONTAINER_IMAGE" \
     --container-mounts "/home:/home,/lustre:/lustre" \
     bash -lc "
        set -uo pipefail
        cd '$BENCH_DIR'
        export DEEPEP_DIR='$DEEPEP_DIR'
        # Torch ships NCCL 2.30.4; DeepEP was previously built vs the >=2.30.4 wheel (2.30.7),
        # causing a runtime 'NCCL too old' assert. Pin the build NCCL to EXACTLY 2.30.4 to match torch.
        export NCCL_PACKAGE='nvidia-nccl-cu13==2.30.4'
        echo '# ==== clean previous build (was compiled vs 2.30.7) ===='
        rm -f \"\$DEEPEP_DIR\"/deep_ep/_C*.so
        rm -rf \"\$DEEPEP_DIR\"/build \"\$DEEPEP_DIR\"/*.egg-info
        echo '# ==== rebuild DeepEP against NCCL 2.30.4 (install_deepep_ngc.sh) ===='
        bash ./install_deepep_ngc.sh 2>&1 | tail -45
        echo '# ==== smoke test ===='
        cd '$BENCH_DIR'
        PYTHONPATH=\"\$DEEPEP_DIR:\${PYTHONPATH:-}\" python3 -c 'import deep_ep; print(\"deep_ep:\", deep_ep.__file__); print(\"ElasticBuffer:\", hasattr(deep_ep, \"ElasticBuffer\"))'
     "
