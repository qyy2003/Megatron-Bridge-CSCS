#!/bin/bash
# Build (or find) the content-addressed code snapshot a job mounts at /opt/Megatron-Bridge:
# pinned Megatron-Bridge + Megatron-LM + patches. Prints "SNAP=<dir>" as the last line.
#
#   tools/build_code_snapshot.sh <src> <code_root> <mbridge_commit> <mlm_commit> [patch sets]
#
#   <src>          directory holding patches/ (a launch snapshot $SRC, or the repo itself)
#   <code_root>    where snapshots live ($MBCSCS_IOPS/code)
#   [patch sets]   comma-separated optional sets: patches/optional/<set>/<tree>/*.patch
#
# Always applied: patches/<tree>/*.patch; then the optional sets in the given order. A different
# set gives a different snapshot, so jobs with different sets can run at the same time. The
# user's submodule working trees are never touched.
#
# train/job_body.sh calls this at every job start. It is only file copying, so it may also be run
# on the login node ahead of a job (git archive of the pinned commits + git apply): on a slow
# iopsstor a fresh build has taken 10-20 min, which otherwise comes out of the job's wall time.
set -o pipefail
SRC=${1:?src}; CODE_ROOT=${2:?code_root}; MBRIDGE_COMMIT=${3:?mbridge commit}; MLM_COMMIT=${4:?mlm commit}
PATCHES=${5:-}
WORKDIR=${WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}

patch_files() {  # <tree>: the patch files for that tree, in application order
    local f set
    for f in "$SRC/patches/$1"/*.patch; do [ -e "$f" ] && echo "$f"; done
    for set in ${PATCHES//,/ }; do
        for f in "$SRC/patches/optional/$set/$1"/*.patch; do [ -e "$f" ] && echo "$f"; done
    done
    return 0
}
for set in ${PATCHES//,/ }; do
    [ -d "$SRC/patches/optional/$set" ] || { echo "ERROR: patch set '$set' not found in $SRC/patches/optional"; exit 1; }
done
PATCH_HASH=$( { patch_files megatron-bridge; echo --; patch_files megatron-lm; } | while read -r f; do
    if [ "$f" = -- ]; then echo --; else cat "$f"; fi; done | sha256sum | cut -c1-12)
SNAP=$CODE_ROOT/mb-${MBRIDGE_COMMIT:0:10}_mlm-${MLM_COMMIT:0:10}_p-$PATCH_HASH
echo "code snapshot: $SNAP${PATCHES:+ (optional patch sets: $PATCHES)}"
mkdir -p "$CODE_ROOT"
(
    flock 9
    if [ ! -f "$SNAP/.src_ready" ]; then
        set -e
        export GIT_CEILING_DIRECTORIES=$CODE_ROOT
        TMP=$SNAP.tmp.${SLURM_JOB_ID:-$(hostname -s)-$$}
        rm -rf "$TMP"
        mkdir -p "$TMP/3rdparty/Megatron-LM"
        # Seconds normally, up to ~20 min when iopsstor metadata is slow (job 3706233); a stall
        # (job 3706202) must not eat the whole job.
        timeout 1800 git -C "$WORKDIR/Megatron-Bridge" archive "$MBRIDGE_COMMIT" | timeout 1800 tar -x -C "$TMP"
        timeout 1800 git -C "$WORKDIR/Megatron-Bridge/3rdparty/Megatron-LM" archive "$MLM_COMMIT" \
            | timeout 1800 tar -x -C "$TMP/3rdparty/Megatron-LM"
        for target in megatron-bridge:. megatron-lm:3rdparty/Megatron-LM; do
            for p in $(patch_files "${target%%:*}"); do
                echo "  apply ${p#"$SRC"/patches/}"
                (cd "$TMP/${target#*:}" && git apply --whitespace=nowarn "$p")
            done
        done
        touch "$TMP/.src_ready"
        rm -rf "$SNAP"   # a leftover without .src_ready is an aborted attempt
        mv "$TMP" "$SNAP"
    fi
) 9>"$CODE_ROOT/.lock" || { echo "ERROR: could not build the code snapshot"; exit 1; }
echo "SNAP=$SNAP"
