#!/bin/bash
#SBATCH --job-name=hier-test
#SBATCH --partition=batch
#SBATCH --account=nemotron_sw_pre
#SBATCH --qos=interactive
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gpus-per-node=4
#SBATCH --time=0:20:00
#SBATCH --output=hier-test-%j.out
#
# Hierarchical-dispatch milestone tests (K0/K1/...). No deepep_env.sh needed.
# Override the test with TEST_CMD, e.g.:
#     TEST_CMD="test_hier.py --milestone k1 --g 2" sbatch test_hier.sbatch
# See memory hier-build-env: /lustre -> /scratch symlink; image gtp_latest.sqsh.

set -uo pipefail
BENCH_DIR="${BENCH_DIR:-/lustre/fsw/portfolios/nemotron/projects/nemotron_sw_pre/users/psinghania/MoE-comms-benchmarking}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/lustre/fsw/portfolios/nemotron/projects/nemotron_sw_pre/users/psinghania/gtp_latest/gtp_latest.sqsh}"
TEST_CMD="${TEST_CMD:-test_hier.py --milestone k1 --g 2}"

srun --container-image "$CONTAINER_IMAGE" \
     --container-mounts "/home:/home,/lustre:/lustre" \
     bash -lc "
        set -uo pipefail
        cd '$BENCH_DIR'
        echo '# container:' \$(python3 -c 'import torch,triton; print(torch.__version__, triton.__version__, torch.version.cuda)')
        torchrun --nproc_per_node=4 $TEST_CMD
     "
