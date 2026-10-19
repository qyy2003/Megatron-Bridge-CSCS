#!/bin/bash
#SBATCH --job-name=mbcscs-tokenize
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --time=04:00:00
#SBATCH --array=0-22
#SBATCH --output=logs/%x-%A_%a.log
#SBATCH --error=logs/%x-%A_%a.log
#
# Tokenize the Nemotron 3.5 Lightning open-data blend (23 sources, data/lightning35_sources.json)
# with the Qwen3.5 tokenizer into $DATA_ROOT/qwen35/<source>/*.{bin,idx}: one array task per
# source, one node each (CPU only). Re-running continues where it stopped; raising
# TARGET_TOKENS adds more files. Then build the blend:  python3.11 data/build_blend.py
#
#   source config.sh
#   sbatch data/tokenize.sbatch                                      # up to 25B tokens per source
#   TARGET_TOKENS=5e9 sbatch data/tokenize.sbatch                    # smaller budget
#   sbatch --array=3,4 data/tokenize.sbatch                          # only some sources
#   SOURCE_INDICES=0-22 MAX_FILES=1 MAX_ROWS=20000 \
#       sbatch -p debug -t 00:30:00 --array=0 data/tokenize.sbatch   # smoke test: all sources, one job
set -euo pipefail

WORKDIR=${WORKDIR:-$SLURM_SUBMIT_DIR}
source "$WORKDIR/env/versions.env"
: "${DATA_ROOT:?source config.sh before sbatch}"
TARGET_TOKENS=${TARGET_TOKENS:-25e9}
# Smoke tests (MAX_ROWS) never write into the real dataset.
OUT_ROOT=${OUT_ROOT:-$DATA_ROOT/qwen35${MAX_ROWS:+-smoke}}
TOK_DIR=$(ls -d "$HF_HOME/hub/models--${HF_MODEL_ID//\//--}/snapshots/$HF_MODEL_REVISION")

mkdir -p "$OUT_ROOT"
# Multi-GB shards read by many ranks: progressive striping (CSCS Lustre guide).
lfs setstripe --component-end 4M --stripe-count 1 --component-end 64M --stripe-count 4 \
    --component-end -1 --stripe-count 16 --stripe-size 4M "$OUT_ROOT" 2>/dev/null || true

EDF=$WORKDIR/logs/edf-tokenize-${SLURM_JOB_ID}.toml
cat > "$EDF" <<EOF
image = "$IMAGE_DIR/$IMAGE_NAME"
# The image's own /opt/Megatron-Bridge has non-traversable directories (mode 644) for non-root
# users; mount the submodule tree over it (only the IndexedDataset reader is used here).
mounts = ["/capstor", "/iopsstor", "/users", "$WORKDIR/Megatron-Bridge:/opt/Megatron-Bridge"]
workdir = "$WORKDIR"
writable = true
entrypoint = true
[env]
PMIX_MCA_psec = "native"
[annotations]
com.hooks.cxi.enabled = "false"
EOF

ARGS=(--sources "$WORKDIR/data/lightning35_sources.json"
      --mirror "$SOURCE_MIRROR" --out-root "$OUT_ROOT" --tokenizer-dir "$TOK_DIR"
      --target-tokens "$TARGET_TOKENS" --workers "$((SLURM_CPUS_PER_TASK - 8))")
[ -n "${MAX_FILES:-}" ] && ARGS+=(--max-files "$MAX_FILES")
[ -n "${MAX_ROWS:-}" ] && ARGS+=(--max-rows-per-file "$MAX_ROWS" --verify)

# One source per array task, or SOURCE_INDICES="0-22" / "0,5,9" processed in this one job.
INDICES=${SLURM_ARRAY_TASK_ID:-0}
if [ -n "${SOURCE_INDICES:-}" ]; then
    INDICES=$(echo "$SOURCE_INDICES" | tr ',' '\n' | while IFS=- read -r a b; do seq "$a" "${b:-$a}"; done)
fi

echo "START $(date) on $(hostname): sources $(echo $INDICES)"
for i in $INDICES; do
    srun --mpi=pmix --network=disable_rdzv_get --environment="$EDF" --cpus-per-task="$SLURM_CPUS_PER_TASK" \
        python "$WORKDIR/data/tokenize_parquet.py" "${ARGS[@]}" --source-index "$i"
done
echo "END $(date)"
