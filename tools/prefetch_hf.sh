#!/bin/bash
# Download the Qwen3.5-35B-A3B and dense (0.8B/2B/4B/9B) configs + tokenizers (no weights, a few MB)
# into $HF_HOME, so
# training and tokenization jobs can run with HF_HUB_OFFLINE=1: hundreds of ranks resolving
# the model on the Hub at start-up get rate limited, and the recipe builds the model from
# AutoConfig.from_pretrained(<model id>) on every rank.
#
# Run on the login node:  ./tools/prefetch_hf.sh
set -euo pipefail
WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$WORKDIR/config.sh"
source "$WORKDIR/env/versions.env"
mkdir -p "$HF_HOME"

PIXI=$(command -v pixi || echo "$HOME/.local/aarch64/pixi/bin/pixi")
# revision="main" (not the sha) so that refs/main is recorded: the recipe loads the model id
# without a revision, which offline mode resolves through refs/main.
# The MoE model, then the dense ones (MODEL=0.8b|2b|4b|9b), each as <id>@<pinned revision>.
for entry in "$HF_MODEL_ID@$HF_MODEL_REVISION" ${HF_DENSE_MODELS:-}; do
    model=${entry%@*}; pinned=${entry#*@}
    HF_HOME=$HF_HOME "$PIXI" exec -s "python=3.12" -s "huggingface_hub>=0.34" -- python - <<EOF
from huggingface_hub import snapshot_download
path = snapshot_download("$model", revision="main",
                         allow_patterns=["*.json", "*.txt", "*.jinja", "*.model", "*.py"])
sha = path.rstrip("/").split("/")[-1]
print("snapshot:", path)
if sha != "$pinned":
    print("WARNING: main is now", sha, "but env/versions.env pins $pinned")
EOF
    touch "$HF_HOME/.prefetched-${model//\//--}"
    ls "$HF_HOME/hub/models--${model//\//--}/snapshots/"*/
done
