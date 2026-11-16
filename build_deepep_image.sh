#!/bin/bash
# Build mcore_inference+fa4+deepep+nccl_ep.sqsh: a copy of mcore_inference+fa4.sqsh with
#   - NCCL upgraded from 2.29.7 to 2.32.3. DeepEP V2.5 needs NCCL >= 2.32.3, and it reuses
#     PyTorch's NCCL communicator, so PyTorch and DeepEP must load the same libnccl.
#   - DeepEP (upstream main) installed. Its GPU kernels are JIT-compiled at runtime by DeepJIT.
#   - libdw (DeepJIT) and ibstat (DeepEP's RDMA bandwidth detection).
#   - NCCL EP (NVIDIA/nccl-extensions) built against that NCCL: the C library, headers, ep_test
#     and ep_bench under /opt/nccl_ep, and the Python bindings (import nccl.ep). Its kernels are
#     also JIT-compiled at runtime (nvcc), cached under /tmp/nccl_ep/jit.
#   - FlashInfer upgraded from 0.6.8 to 0.7.3 for flashinfer.moe_ep (the NVLink one-sided
#     all-to-all vLLM uses needs >= 0.7.2). Its prebuilt flashinfer-cubin / flashinfer-jit-cache
#     0.6.8 are removed (none exist for 0.7.3 on PyPI), so FlashInfer JIT-compiles its kernels.
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
OUT_IMAGE=${OUT_IMAGE:-${IMAGES_DIR}/mcore_inference+fa4+deepep+nccl_ep+flashinfer0.7.3.sqsh}
NCCL_PIN=${NCCL_PIN:-2.32.3}
DEEPEP_COMMIT=${DEEPEP_COMMIT:-f6aa38fe06cd4a0489e7730365d4e0114970a2f3}
NCCL_EXTENSIONS_COMMIT=${NCCL_EXTENSIONS_COMMIT:-3401dea13534bc24ca827fe6e671f1edcf51d9a0}
# Python dependencies of nccl.ep that the base image lacks (or has too old: cuda-pathfinder 1.4.0).
NCCL4PY_PIN=${NCCL4PY_PIN:-0.6.0}
CUDA_CORE_PIN=${CUDA_CORE_PIN:-1.2.2}
CUDA_PATHFINDER_PIN=${CUDA_PATHFINDER_PIN:-1.8.6}
FLASHINFER_PIN=${FLASHINFER_PIN:-0.7.3}
# flashinfer-python 0.7.3 requires cuda-tile >= 1.4, nvidia-cudnn-frontend >= 1.32 and
# nvidia-cutlass-dsl >= 4.8 (the image has 1.1.0, 1.18.0 and 4.6.0.dev0); `import flashinfer`
# needs the cutlass-dsl upgrade (cute.FastDivmodDivisorV2). FA4 in this image was built on
# cutlass-dsl 4.6: the build only reports whether it still imports.
CUDA_TILE_PIN=${CUDA_TILE_PIN:-1.8.0}
CUDNN_FRONTEND_PIN=${CUDNN_FRONTEND_PIN:-1.32.2}
CUTLASS_DSL_PIN=${CUTLASS_DSL_PIN:-4.8.1}
# GB200 (sm_100) and GB300 (sm_103). Only the library's precompiled kernels; the rest are
# JIT-compiled for the running GPU.
NCCL_EP_GENCODE=${NCCL_EP_GENCODE:-"-gencode=arch=compute_100,code=sm_100 -gencode=arch=compute_100a,code=sm_100a -gencode=arch=compute_103a,code=sm_103a"}
# Install prefix of the NCCL EP build: lib/, include/, test/nccl_ep/{ep_test,ep_bench}.
NCCL_EP_HOME=/opt/nccl_ep
# Sources go inside the container's /tmp, which the export leaves out of the image.
DEEPEP_SRC=/tmp/DeepEP
NCCL_EXTENSIONS_SRC=/tmp/nccl-extensions

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

# ── 4. NCCL EP ───────────────────────────────────────────────────────────────────────────────
# Built from source rather than `pip install nccl-extensions`: the PyPI wheel (0.1.0) pins
# nvidia-nccl-cu13==2.30.7, which would replace the NCCL 2.32.3 that DeepEP needs.
git clone https://github.com/NVIDIA/nccl-extensions.git "${NCCL_EXTENSIONS_SRC}"
cd "${NCCL_EXTENSIONS_SRC}"
git checkout --detach "${NCCL_EXTENSIONS_COMMIT}"
# NCCL_INCDIR/NCCL_LIBDIR replace the vendored NCCL submodule: headers from the NCCL wheel (they
# include the Device API), library from the system copy (it has the unversioned libnccl.so that
# -lnccl links). The include dir is baked into libnccl_ep as the JIT's default NCCL header path,
# so it must be one that stays in the image.
make -C nccl_ep -j"$(nproc)" MPI=1 MPI_HOME=/usr/local/mpi \
    NCCL_INCDIR="${NCCL_ROOT}/include" NCCL_LIBDIR="${SYS_LIB}" \
    BUILDDIR="${NCCL_EP_HOME}" NVCC_GENCODE="${NCCL_EP_GENCODE}" lib ep_test ep_bench
rm -rf "${NCCL_EP_HOME}/obj"
ls -lR "${NCCL_EP_HOME}/lib" "${NCCL_EP_HOME}/test"

# Python bindings. nccl.ep loads the libnccl_ep bundled in its package (lib/cu13/, chosen by the
# cuda.bindings major) and sets NCCL_EP_HOME to its package dir for the JIT headers, so stage the
# library and headers just built into the package before installing it.
pip install --no-deps --no-cache-dir "nccl4py==${NCCL4PY_PIN}" "cuda-core==${CUDA_CORE_PIN}" \
    "cuda-pathfinder==${CUDA_PATHFINDER_PIN}"
mkdir -p python/nccl/ep/lib/cu13 python/nccl/ep/include
cp -L "${NCCL_EP_HOME}/lib/libnccl_ep.so.0" python/nccl/ep/lib/cu13/
cp -a "${NCCL_EP_HOME}/include/." python/nccl/ep/include/
CUDA_HOME=/usr/local/cuda pip install --no-build-isolation --no-deps --no-cache-dir -v ./python
cd /
# After torch, nccl4py and libnccl_ep must reuse torch's libnccl rather than map a second copy.
NCCL_PIN="${NCCL_PIN}" python - <<'EOF'
import os, torch
import nccl.ep as ep
from nccl.bindings._internal import nccl as nccl_loader
nccl_loader._inspect_function_pointers()
print('nccl.ep', ep.get_lib_version(), ep.get_lib_path(), 'NCCL_EP_HOME', os.environ['NCCL_EP_HOME'])
mapped = sorted({line.split()[-1] for line in open('/proc/self/maps') if 'libnccl' in line})
print('mapped:', mapped)
core_nccl = [path for path in mapped if '/libnccl.so' in path]
assert len(core_nccl) == 1 and core_nccl[0].endswith(f'libnccl.so.{os.environ["NCCL_PIN"]}'), mapped
assert any('libnccl_ep.so' in path for path in mapped), mapped
EOF

# ── 5. FlashInfer 0.7.3 ──────────────────────────────────────────────────────────────────────
# --no-deps: its declared dependencies would also replace nccl-extensions with the PyPI wheel
# (NCCL 2.30.7); the ones it needs are pinned above (cutlass-dsl's CUDA 12 libs left out).
pip uninstall -y flashinfer-cubin flashinfer-jit-cache
pip install --no-deps --no-cache-dir "flashinfer-python==${FLASHINFER_PIN}" \
    "cuda-tile==${CUDA_TILE_PIN}" "nvidia-cudnn-frontend==${CUDNN_FRONTEND_PIN}" \
    "nvidia-cutlass-dsl==${CUTLASS_DSL_PIN}" "nvidia-cutlass-dsl-libs-base==${CUTLASS_DSL_PIN}" \
    "nvidia-cutlass-dsl-libs-core==${CUTLASS_DSL_PIN}" "nvidia-cutlass-dsl-libs-cu13==${CUTLASS_DSL_PIN}"
python - <<'EOF'
import torch, flashinfer
from flashinfer.moe_ep import (BootstrapConfig, MoEEpCommParams, NVLinkOneSidedAlltoAll,
                               NVLinkOneSidedConfig, create_communication)
print('flashinfer', flashinfer.__version__, flashinfer.__file__)
EOF
if FA4_ERROR=$(python -W ignore -c 'import flash_attn.cute.interface' 2>&1); then FA4_IMPORT=ok
else FA4_IMPORT="FAILED: $(echo "${FA4_ERROR}" | tail -1)"; fi
echo "FA4 (flash_attn.cute) import with cutlass-dsl ${CUTLASS_DSL_PIN}: ${FA4_IMPORT}"

# ── 6. Record versions ──────────────────────────────────────────────────────────────────────
MANIFEST=${OUT_IMAGE%.sqsh}.manifest.txt
{
    echo "image:        ${OUT_IMAGE}"
    echo "built:        $(date -u +%Y-%m-%dT%H:%M:%SZ) on $(hostname)"
    echo "base image:   ${BASE_IMAGE}"
    echo "              $(stat -c '%s bytes, modified %y' "${BASE_IMAGE}")"
    echo "DeepEP:       ${DEEPEP_COMMIT} ($(git -C "${DEEPEP_SRC}" log -1 --format='%ad %s' --date=short))"
    echo "DeepJIT:      $(git -C "${DEEPEP_SRC}" submodule status | tr -s ' ')"
    echo "nccl-ext:     ${NCCL_EXTENSIONS_COMMIT} ($(git -C "${NCCL_EXTENSIONS_SRC}" log -1 --format='%ad %s' --date=short))"
    echo "NCCL EP:      ${NCCL_EP_HOME} ($(readlink "${NCCL_EP_HOME}/lib/libnccl_ep.so.0"); ${NCCL_EP_GENCODE})"
    echo "FA4 import:   ${FA4_IMPORT}"
    python - <<'EOF'
import ctypes, importlib.metadata as md, torch
version = ctypes.c_int()
ctypes.CDLL('libnccl.so.2').ncclGetVersion(ctypes.byref(version))
print(f"torch:        {torch.__version__}")
print(f"NCCL:         {version.value // 10000}.{version.value // 100 % 100}.{version.value % 100} at runtime"
      f" (torch compiled against {'.'.join(map(str, torch.cuda.nccl.version()))})")
print(f"CUDA:         {torch.version.cuda}")
for dist in ("deep_ep", "nvidia-nccl-cu13", "nccl-extensions", "nccl4py", "cuda-core", "cuda-pathfinder",
             "flashinfer-python", "flashinfer-cubin", "flashinfer-jit-cache", "cuda-tile",
             "nvidia-cudnn-frontend", "nvidia-cutlass-dsl", "flash-attn-4", "transformer_engine",
             "triton"):
    try:
        print(f"{dist + ':':<14}{md.version(dist)}")
    except md.PackageNotFoundError:
        print(f"{dist + ':':<14}not installed")
EOF
} | tee "${MANIFEST}"

# ── 7. Export the root filesystem ────────────────────────────────────────────────────────────
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
