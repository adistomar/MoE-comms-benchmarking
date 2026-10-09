#!/bin/bash
# Per-task launcher for srun/pyxis jobs where each Slurm task is one EP rank (one task per
# GPU, no torchrun). Sets the torch env:// rendezvous variables from Slurm, sources the
# DeepEP environment, and runs run.py with the given arguments:
#
#   srun --ntasks-per-node=4 ... bash launch_rank.sh --impl both --out ep16.csv
#
# MASTER_ADDR defaults to the first host of the allocation. It is derived here (not with
# scontrol, which the container may not have) so the script also works when launched
# directly inside the container. Requires a prebuilt DeepEP for --impl deepep/both/all
# (run install_deepep_ngc.sh from a single task first).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

export RANK="${SLURM_PROCID}" WORLD_SIZE="${SLURM_NTASKS}" LOCAL_RANK="${SLURM_LOCALID}"
if [ -z "${MASTER_ADDR:-}" ]; then
    MASTER_ADDR="$(python3 - <<'PY'
import os, re
nodelist = os.environ["SLURM_JOB_NODELIST"]
# First top-level entry of e.g. "pre[001-004,010],other7" -> "pre001"
first = re.split(r",(?![^\[]*\])", nodelist)[0]
m = re.match(r"^([^\[]*)\[([^\]]*)\](.*)$", first)
print(first if m is None else m.group(1) + m.group(2).split(",")[0].split("-")[0] + m.group(3))
PY
)"
fi
export MASTER_ADDR MASTER_PORT="${MASTER_PORT:-29500}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_cache_rank${RANK}}"

# DeepEP's environment is only needed when run.py imports deep_ep (run.py's default is all).
impl="all"
args=("$@")
for i in "${!args[@]}"; do
    [ "${args[$i]}" = "--impl" ] && impl="${args[$((i + 1))]:-all}"
    case "${args[$i]}" in --impl=*) impl="${args[$i]#--impl=}" ;; esac
done
case ",$impl," in
    *,deepep,*|*,both,*|*,all,*) source ./deepep_env.sh || exit 1 ;;
esac
python3 run.py "$@"
