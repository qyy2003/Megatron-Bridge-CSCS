#!/bin/bash
# code_restore.sh
#
# Put the two upstream trees in the state the jobs run: pinned commit + every patch applied
# to the working tree, so you can read / edit / debug the exact code.
#
#   Megatron-Bridge/                         <- patches/megatron-bridge/*.patch
#   Megatron-Bridge/3rdparty/Megatron-LM/    <- patches/megatron-lm/*.patch
#
# Jobs never use these working trees directly: launch.sh snapshots the pinned commits and
# applies patches/ itself. So edits only reach a job after ./code_store.sh.
#
# Usage:
#   ./code_restore.sh           # refuse if a tree has uncommitted changes
#   ./code_restore.sh -f        # discard them first (tracked files + files created by patches)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
source "$ROOT/env/versions.env"

FORCE=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        -f|--force) FORCE=1; shift ;;
        -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# Paths a set of patches creates (they are untracked after a reset and would block git apply).
new_files() { grep -h -B1 '^new file mode' "$@" 2>/dev/null | sed -n 's|^diff --git a/\(.*\) b/.*|\1|p' || true; }

restore() {  # <repo dir> <pinned commit> <patch dir>
    local repo=$1 pin=$2 pdir=$3 p
    local name=${repo#"$ROOT/"}
    shopt -s nullglob
    local patches=("$pdir"/*.patch)
    shopt -u nullglob

    if [[ "$(git -C "$repo" rev-parse HEAD)" != "$pin" ]]; then
        echo "WARNING: $name HEAD is $(git -C "$repo" rev-parse --short HEAD), jobs use $pin (env/versions.env)"
    fi
    if [[ -n "$(git -C "$repo" status --porcelain --ignore-submodules=all)" ]]; then
        if [[ $FORCE -eq 1 ]]; then
            echo "$name: discarding uncommitted changes"
            git -C "$repo" checkout -- .
            [[ ${#patches[@]} -gt 0 ]] && new_files "${patches[@]}" | while read -r f; do rm -f "$repo/$f"; done
        else
            echo "$name has uncommitted changes:"
            git -C "$repo" status --short --ignore-submodules=all
            echo "Save first: ./code_store.sh   or discard: ./code_restore.sh -f"
            exit 1
        fi
    fi
    if [[ ${#patches[@]} -eq 0 ]]; then
        echo "$name: no patches"
        return
    fi
    git -C "$repo" apply --check "${patches[@]}" || { echo "$name: patches do not apply cleanly" >&2; exit 1; }
    git -C "$repo" apply "${patches[@]}"
    echo "$name: applied ${#patches[@]} patch(es):"
    for p in "${patches[@]}"; do echo "  $(basename "$p")"; done
}

restore "$ROOT/Megatron-Bridge" "$MBRIDGE_COMMIT" "$ROOT/patches/megatron-bridge"
restore "$ROOT/Megatron-Bridge/3rdparty/Megatron-LM" "$MLM_COMMIT" "$ROOT/patches/megatron-lm"
