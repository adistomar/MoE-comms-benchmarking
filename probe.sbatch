#!/bin/bash
#SBATCH --job-name=hier-probe
#SBATCH --partition=batch
#SBATCH --account=nemotron_sw_pre
#SBATCH --qos=interactive
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gpus-per-node=4
#SBATCH --time=0:15:00
#SBATCH --output=hier-probe-%j.out
#
# K-1 probe (probe_symm_mem.py): does the torch _SymmetricMemory handle expose
# per-peer DATA pointers usable for a directed (non-multicast) P2P store from a
# Triton kernel? Result decides the pointer path for the K1 outer-a2a kernel.
#
# Needs NO deepep_env.sh — the probe imports only torch/triton + the vendored
# nvls/ package, all present in the mcore_inference container. (Contrast run.sbatch,
# which sources deepep_env.sh for the DeepEP path.)
#
# ADJUST for your cluster: the #SBATCH --account/--partition/--qos above and the
# CONTAINER_IMAGE. Note /lustre -> /scratch is a symlink here, so the container
# mounts /lustre and references the repo via its /lustre path. Override the image
# without editing:  CONTAINER_IMAGE=/path/to.sqsh sbatch probe.sbatch
# Read the log:  tail -f hier-probe-<jobid>.out

set -uo pipefail
# In-container path to the repo (via the /lustre mount; NOT SLURM_SUBMIT_DIR, which
# would be the host /scratch path the container does not mount).
BENCH_DIR="${BENCH_DIR:-/lustre/fsw/portfolios/nemotron/projects/nemotron_sw_pre/users/psinghania/MoE-comms-benchmarking}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/lustre/fsw/portfolios/nemotron/projects/nemotron_sw_pre/users/psinghania/gtp_latest/gtp_latest.sqsh}"

srun --container-image "$CONTAINER_IMAGE" \
     --container-mounts "/home:/home,/lustre:/lustre" \
     bash -lc "
        set -uo pipefail
        cd '$BENCH_DIR'
        echo '# container:' \$(python3 -c 'import torch,triton; print(torch.__version__, triton.__version__, torch.version.cuda)')
        torchrun --nproc_per_node=4 probe_symm_mem.py
     "
