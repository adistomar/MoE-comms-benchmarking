#!/bin/bash
# Build mcore_inference+fa4+deepep.sqsh: a copy of mcore_inference+fa4.sqsh with
#   - NCCL upgraded from 2.29.7 to 2.32.3. DeepEP V2.5 needs NCCL >= 2.32.3, and it reuses
#     PyTorch's NCCL communicator, so PyTorch and DeepEP must load the same libnccl.
#   - DeepEP (upstream main) installed. Its GPU kernels are JIT-compiled at runtime by DeepJIT.
#   - libdw (DeepJIT) and ibstat (DeepEP's RDMA bandwidth detection).
#
# Runs INSIDE a pyxis container started from the base image (as remapped root) with /lustre
# mounted, then squashes the container's root filesystem into the new image, e.g.
#
#   srun -N1 --container-image "$BASE_IMAGE" --container-remap-root \
#        --container-mounts /lustre:/lustre bash build_deepep_image.sh
#
# Writes ${OUT_IMAGE} and a .manifest.txt of the versions next to it. See README.md (Setup).
set -euxo pipefail

IMAGES_DIR=${IMAGES_DIR:-/lustre/fsw/portfolios/nemotron/users/aditomar/images}
BASE_IMAGE=${BASE_IMAGE:-/lustre/fsw/portfolios/llmservice/users/ksanthanam/images/mcore_inference+fa4.sqsh}
OUT_IMAGE=${OUT_IMAGE:-${IMAGES_DIR}/mcore_inference+fa4+deepep.sqsh}
NCCL_PIN=${NCCL_PIN:-2.32.3}
DEEPEP_COMMIT=${DEEPEP_COMMIT:-f6aa38fe06cd4a0489e7730365d4e0114970a2f3}
# Inside the container's /tmp, which the export leaves out of the image.
DEEPEP_SRC=/tmp/DeepEP

# The NGC pip constraint file pins the image's packages; these installs are deliberate upgrades.
export PIP_CONSTRAINT=

# ── 1. System packages ──────────────────────────────────────────────────────────────────────
# libdw-dev: DeepJIT includes elfutils/libdwfl.h, and dlopens libdw.so.1 for stack traces.
# infiniband-diags: ibstat, which DeepEP uses to detect RDMA bandwidth.
# squashfs-tools: mksquashfs, to export the image.
apt-get update
apt-get install -y --no-install-recommends libdw-dev infiniband-diags squashfs-tools
apt-get clean
rm -rf /var/lib/apt/lists/*

# ── 2. NCCL: replace the system library and headers with the wheel's ───────────────────────
# PyTorch links the system libnccl.so.2 (it does not bundle NCCL), and DeepEP checks at import
# that the loaded libnccl is byte-identical to the one under its NCCL root (the wheel).
pip install --no-deps --no-cache-dir "nvidia-nccl-cu13==${NCCL_PIN}"
NCCL_ROOT=$(python -c 'import nvidia.nccl; print(list(nvidia.nccl.__path__)[0])')
SYS_LIB=/usr/lib/$(gcc -print-multiarch)
ls -l "${NCCL_ROOT}/lib" "${NCCL_ROOT}/include" "${SYS_LIB}"/libnccl*
find "${SYS_LIB}" -maxdepth 1 -name 'libnccl.so*' -delete
cp "${NCCL_ROOT}/lib/libnccl.so.2" "${SYS_LIB}/libnccl.so.${NCCL_PIN}"
ln -s "libnccl.so.${NCCL_PIN}" "${SYS_LIB}/libnccl.so.2"
ln -s libnccl.so.2 "${SYS_LIB}/libnccl.so"
for header in "${NCCL_ROOT}"/include/*; do
    rm -rf "/usr/include/$(basename "${header}")"
    cp -a "${header}" /usr/include/
done
ldconfig
sed -i "s/^NCCL_VERSION=.*/NCCL_VERSION=${NCCL_PIN}/" /etc/environment
# torch.cuda.nccl.version() is the version PyTorch was compiled against; check the library it
# actually maps, and that library's runtime version.
NCCL_PIN="${NCCL_PIN}" python - <<'EOF'
import ctypes, os, torch
mapped = sorted({line.split()[-1] for line in open('/proc/self/maps') if 'libnccl' in line})
version = ctypes.c_int()
ctypes.CDLL('libnccl.so.2').ncclGetVersion(ctypes.byref(version))
major, minor, patch = (int(x) for x in os.environ['NCCL_PIN'].split('.'))
print(f'libnccl mapped by torch: {mapped}, runtime version code: {version.value}')
assert mapped and all(path.endswith(f'libnccl.so.{os.environ["NCCL_PIN"]}') for path in mapped)
assert version.value == major * 10000 + minor * 100 + patch
EOF

# ── 3. DeepEP ────────────────────────────────────────────────────────────────────────────────
git clone https://github.com/deepseek-ai/DeepEP.git "${DEEPEP_SRC}"
cd "${DEEPEP_SRC}"
git checkout --detach "${DEEPEP_COMMIT}"
git submodule update --init --recursive
# EP_NCCL_ROOT_DIR is persisted into deep_ep/envs.py as the import-time default.
EP_NCCL_ROOT_DIR="${NCCL_ROOT}" pip install --no-build-isolation --no-deps --no-cache-dir -v .
cd /
python -c "import torch, deep_ep; from deep_ep import EPBuffer; print('deep_ep', deep_ep.__file__)"

# ── 4. Record versions ──────────────────────────────────────────────────────────────────────
MANIFEST=${OUT_IMAGE%.sqsh}.manifest.txt
{
    echo "image:        ${OUT_IMAGE}"
    echo "built:        $(date -u +%Y-%m-%dT%H:%M:%SZ) on $(hostname)"
    echo "base image:   ${BASE_IMAGE}"
    echo "              $(stat -c '%s bytes, modified %y' "${BASE_IMAGE}")"
    echo "DeepEP:       ${DEEPEP_COMMIT} ($(git -C "${DEEPEP_SRC}" log -1 --format='%ad %s' --date=short))"
    echo "DeepJIT:      $(git -C "${DEEPEP_SRC}" submodule status | tr -s ' ')"
    python - <<'EOF'
import ctypes, importlib.metadata as md, torch
version = ctypes.c_int()
ctypes.CDLL('libnccl.so.2').ncclGetVersion(ctypes.byref(version))
print(f"torch:        {torch.__version__}")
print(f"NCCL:         {version.value // 10000}.{version.value // 100 % 100}.{version.value % 100} at runtime"
      f" (torch compiled against {'.'.join(map(str, torch.cuda.nccl.version()))})")
print(f"CUDA:         {torch.version.cuda}")
for dist in ("deep_ep", "nvidia-nccl-cu13", "flashinfer-python", "transformer_engine", "triton"):
    try:
        print(f"{dist + ':':<14}{md.version(dist)}")
    except md.PackageNotFoundError:
        print(f"{dist + ':':<14}not installed")
EOF
} | tee "${MANIFEST}"

# ── 5. Export the root filesystem ────────────────────────────────────────────────────────────
rm -rf /root/.cache/pip
# Exclude everything mounted into this container at runtime (/lustre, /proc, /dev, driver
# libraries and binaries injected by the GPU hook, /etc/hosts, ...): directory mount points
# are kept empty, file mount points are dropped, as in an image fresh from enroot import.
# /tmp is excluded rather than emptied: Slurm's PMIx mounts a directory there that cannot be
# removed.
EXCLUDES=$(mktemp)
{
    echo 'tmp/*'
    echo 'var/tmp/*'
    awk '$5 != "/" { print $5 }' /proc/self/mountinfo | sort -u | while read -r mount_point; do
        if [ -d "${mount_point}" ]; then
            echo "${mount_point#/}/*"
        else
            echo "${mount_point#/}"
        fi
    done
} > "${EXCLUDES}"
cat "${EXCLUDES}"
rm -f "${OUT_IMAGE}.partial"
# Same compression as enroot's default (ENROOT_SQUASH_OPTIONS="-comp lzo -noD").
mksquashfs / "${OUT_IMAGE}.partial" -comp lzo -noD -no-progress -one-file-system \
    -wildcards -ef "${EXCLUDES}"
unsquashfs -s "${OUT_IMAGE}.partial"
mv "${OUT_IMAGE}.partial" "${OUT_IMAGE}"
ls -l "${OUT_IMAGE}"
