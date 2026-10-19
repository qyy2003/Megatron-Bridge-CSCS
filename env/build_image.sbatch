#!/bin/bash
#SBATCH --job-name=mbcscs-image
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.log
#
# Build the training container and import it as a squashfs for the Container Engine:
#
#   nvcr.io/nvidia/nemo:${NEMO_TAG}            (ships Megatron-Bridge v0.6.2)
#   + CSCS Alps network stack                  (libfabric/CXI, NCCL 2.30.7 incl. the
#                                               NCCL inspector plugin, aws-ofi-nccl, ...)
#
# The Alps layer is built with CSCS's own recipe (eth-cscs/alps-extended-images,
# pinned in env/versions.env) through their manual-build helper, so the network stack
# is the one CSCS validates for Slingshot. Usage, from the repo root:
#
#   source config.sh && sbatch env/build_image.sbatch
#
# Output: $IMAGE_DIR/$IMAGE_NAME (+ a .versions.txt with what is inside).
set -euo pipefail

WORKDIR=${WORKDIR:-$SLURM_SUBMIT_DIR}
source "$WORKDIR/env/versions.env"
: "${IMAGE_DIR:?IMAGE_DIR is not set - source config.sh before sbatch}"

echo "[$(date)] building nemo:${NEMO_TAG} + alps stack (${ALPS_IMAGES_COMMIT:0:7}) on $(hostname)"

BUILD_ROOT=/dev/shm/$USER/mbcscs-build-$SLURM_JOB_ID
mkdir -p "$BUILD_ROOT/tmp" "$IMAGE_DIR"
# Rootless podman storage holds files owned by sub-UIDs: remove it from inside the
# user namespace (tmpfs is wiped at job end anyway).
trap 'podman unshare rm -rf "$BUILD_ROOT" 2>/dev/null || rm -rf "$BUILD_ROOT" 2>/dev/null || true' EXIT

# Podman storage must live on tmpfs on Alps. Use a job-private storage config so the
# user's ~/.config/containers/storage.conf is never touched.
export CONTAINERS_STORAGE_CONF=$BUILD_ROOT/storage.conf
cat > "$CONTAINERS_STORAGE_CONF" <<EOF
[storage]
driver = "overlay"
runroot = "$BUILD_ROOT/runroot"
graphroot = "$BUILD_ROOT/root"
EOF
export TMPDIR=$BUILD_ROOT/tmp
export ENROOT_TEMP_PATH=$BUILD_ROOT/tmp
podman info 2>/dev/null | grep -A 2 "store:" || true

################ 1) Alps layer via CSCS's recipe ################
ALPS_SRC=$BUILD_ROOT/alps-extended-images
git clone -q "$ALPS_IMAGES_REPO" "$ALPS_SRC"
git -C "$ALPS_SRC" checkout -q "$ALPS_IMAGES_COMMIT"
cp -r "$WORKDIR/env/alps-variant/nemo-$NEMO_TAG" "$ALPS_SRC/Alps-Images/NGC/"

cd "$ALPS_SRC"

# Stage 0: CSCS's "purge" step as a layer of its own. Rootless podman on Alps cannot, within
# one build step, re-create a directory that the same step deleted from a lower image layer
# ("mv: cannot overwrite non-directory ... with directory": the rename sees the whiteout;
# reproduced on this base). The Alps step purges NeMo's network stack (NCCL 2.30.7 .debs,
# HPC-X, EFA, ...) and re-installs it in a single RUN, so the purge has to happen one layer
# earlier; the Alps step's own purge is then a no-op. The nvshmem4py wheel (unused here) is
# skipped for the same reason (its pip install can reinstall packages in place).
PREPURGED=localhost/mbcscs/nemo-prepurged:$NEMO_TAG
REMOVE_HPCX_DIRS=$(source "$ALPS_SRC/Alps-Images/NGC/nemo-$NEMO_TAG/profile.env" && echo "$REMOVE_HPCX_DIRS")
cat > "$BUILD_ROOT/Containerfile.prepurge" <<'EOF'
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
ARG REMOVE_HPCX_DIRS=""
# Same flat layout as Containerfile.ngc-alps (the scripts resolve each other via SCRIPT_DIR).
COPY Alps-Images/common/alps-stack-versions.env Alps-Images/common/package-helpers.sh \
     Alps-Images/common/install-alps-hpc-stack.sh Alps-Images/NGC/install-alps-cuda-components.sh \
     Alps-Images/NGC/install-alps-cuda-stack.sh /opt/alps-prepurge/
RUN REMOVE_HPCX_DIRS="${REMOVE_HPCX_DIRS}" bash /opt/alps-prepurge/install-alps-cuda-stack.sh purge \
 && rm -rf /opt/alps-prepurge
ENV NVSHMEM_ENABLE_PYTHON=0
EOF
echo "[$(date)] stage 0: purge preinstalled network stack -> $PREPURGED"
podman build --network=host --format docker -f "$BUILD_ROOT/Containerfile.prepurge" -t "$PREPURGED" \
    --build-arg BASE_IMAGE="jfrog.svc.cscs.ch/nvcr/nvidia/nemo:$NEMO_TAG" \
    --build-arg REMOVE_HPCX_DIRS="$REMOVE_HPCX_DIRS" .

# Stage 1: the Alps stack with CSCS's helper (same refs/hashes as their CI), on the purged base.
IMAGE_PREFIX=localhost/mbcscs ALPS_REV=$ALPS_REV \
    manual-build/manual-build.sh base cuda nemo "$NEMO_TAG" "$BUILD_ROOT/build-base.sh"
# CSCS docs: since 2026-07-30 podman builds on Alps need --network=host.
sed -i 's/^podman build /podman build --network=host /' "$BUILD_ROOT/build-base.sh"
sed -i "s|--build-arg BASE_IMAGE=\"[^\"]*\"|--build-arg BASE_IMAGE=\"$PREPURGED\"|" "$BUILD_ROOT/build-base.sh"
grep -q "BASE_IMAGE=\"$PREPURGED\"" "$BUILD_ROOT/build-base.sh" || { echo "could not point build-base.sh at $PREPURGED"; exit 1; }
IMAGE_REF=$(sed -n 's/^printf .Also tagged %s\\n. "\(.*\)"$/\1/p' "$BUILD_ROOT/build-base.sh")
[ -n "$IMAGE_REF" ] || { echo "could not parse the image tag from build-base.sh"; exit 1; }
echo "[$(date)] stage 1: podman build -> $IMAGE_REF"
bash "$BUILD_ROOT/build-base.sh"

# Stage 2: NeMo 26.08.01 ships /opt/Megatron-Bridge (and /workspace/tutorials) with directories
# of mode 644 (no x bit). Root does not care, but the Container Engine runs as the user, for whom
# they are not traversable ("Permission denied" importing megatron.core from the image's copy).
# Training jobs mount their code snapshot over /opt/Megatron-Bridge anyway; fix it for all users.
printf 'FROM %s\nRUN chmod -R a+rX /opt/Megatron-Bridge /workspace\n' "$IMAGE_REF" > "$BUILD_ROOT/Containerfile.perms"
podman build --network=host --format docker -f "$BUILD_ROOT/Containerfile.perms" \
    -t "localhost/mbcscs/nemo-alps-final:$NEMO_TAG" "$BUILD_ROOT"
IMAGE_REF=localhost/mbcscs/nemo-alps-final:$NEMO_TAG

################ 2) Import as squashfs ################
# Big file read by every node at job start: stripe it (CSCS Lustre guide).
lfs setstripe --component-end 4M --stripe-count 1 --component-end 64M --stripe-count 4 \
    --component-end -1 --stripe-count 32 --stripe-size 4M "$IMAGE_DIR" 2>/dev/null || true
OUT=$IMAGE_DIR/$IMAGE_NAME
rm -f "$OUT.tmp"
echo "[$(date)] enroot import -> $OUT"
# enroot can exit non-zero in its rootless-podman cleanup after a complete write (seen on
# Alps), so judge the result by the squashfs itself.
enroot import -x mount -o "$OUT.tmp" "podman://$IMAGE_REF" || echo "enroot import exited with $?; validating output"
unsquashfs -s "$OUT.tmp" > /dev/null || { echo "ERROR: $OUT.tmp is not a valid squashfs"; exit 1; }
mv -f "$OUT.tmp" "$OUT"
ls -lh "$OUT"

################ 3) Record and sanity-check what is inside ################
EDF=$BUILD_ROOT/verify.toml
cat > "$EDF" <<EOF
image = "$OUT"
mounts = ["/capstor", "/iopsstor", "/users"]
writable = true
entrypoint = true
[env]
PMIX_MCA_psec = "native"
[annotations]
com.hooks.cxi.enabled = "false"
EOF
cd "$WORKDIR"
srun --mpi=pmix --network=disable_rdzv_get --environment="$EDF" bash -c '
set +e
echo "image: '"$IMAGE_REF"'  built: $(date -u +%FT%TZ)"
echo "nemo container: '"$NEMO_TAG"'  alps-extended-images: '"$ALPS_IMAGES_COMMIT"'"
git -C /opt/Megatron-Bridge log -1 --format="container Megatron-Bridge: %H %d %s" 2>/dev/null \
    || echo "container Megatron-Bridge: $(ls /opt/Megatron-Bridge 2>&1 | head -3)"
python - <<PY
import ctypes, importlib, importlib.metadata as md
for pkg in ["torch", "transformer-engine", "megatron-core", "megatron-bridge", "transformers",
            "flash-linear-attention", "wandb", "nvidia-resiliency-ext", "pyarrow", "tokenizers"]:
    try:
        print(f"{pkg}: {md.version(pkg)}")
    except Exception as e:
        print(f"{pkg}: not found ({e})")
import torch
libs = sorted({l.split()[-1] for l in open("/proc/self/maps") if "libnccl.so" in l})
v = ctypes.c_int(); ctypes.CDLL(libs[0] if libs else "libnccl.so.2").ncclGetVersion(ctypes.byref(v))
print(f"NCCL loaded by torch: {libs} -> version {v.value}")
print(f"torch {torch.__version__} cuda {torch.version.cuda} gpus {torch.cuda.device_count()}")
import megatron.bridge, megatron.core
print("megatron.bridge from", megatron.bridge.__file__)
print("megatron.core   from", megatron.core.__file__)
PY
ls -l /usr/local/lib/libnccl-profiler-inspector.so
nsys --version
which numactl zstd gzip
env | grep -E "^(NCCL|FI)_" | sort
' 2>&1 | tee "$OUT.versions.txt"

echo "[$(date)] done: $OUT"
