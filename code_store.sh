#!/bin/bash
# code_store.sh
#
# Save your work-in-progress edits in Megatron-Bridge/ and/or its Megatron-LM as patches.
#
#   no args:   for each tree with edits, (over)write patches/<tree>/1000-cache.patch with
#              "everything beyond the named patches", i.e. the diff against pinned commit +
#              all non-cache patches. Slot 1000 sorts after 0001, 0002, ... so jobs apply it last.
#   -N name:   save to the cache as above, then promote each tree's cache to the next free
#              slot below 1000: patches/<tree>/NNNN-name.patch. Add a comment header to it
#              (problem / what / where / how to port, see How_To_Edit.md) before committing.
#
# New files are included (intent-to-add), the nested submodule pointer is ignored.
# The working trees are left exactly as they were, so you can keep editing.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"

NAME=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        -N) [[ $# -lt 2 ]] && { echo "Error: -N requires a name" >&2; exit 1; }; NAME="$2"; shift 2 ;;
        -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

store() {  # <repo dir> <patch dir>
    local repo=$1 pdir=$2 p
    local name=${repo#"$ROOT/"}
    mkdir -p "$pdir"
    local cache=$pdir/1000-cache.patch
    local named=()
    shopt -s nullglob
    for p in "$pdir"/*.patch; do [[ "$(basename "$p")" == *-cache.patch ]] || named+=("$p"); done
    shopt -u nullglob

    # Remove the named patches (newest first) so the tree is "pinned + cache + new edits".
    local reversed=()
    for ((i = ${#named[@]} - 1; i >= 0; i--)); do reversed+=("${named[i]}"); done
    if [[ ${#reversed[@]} -gt 0 ]]; then
        if ! git -C "$repo" apply -R --check "${reversed[@]}" 2>/dev/null; then
            echo "$name: named patches are not applied in the working tree; run ./code_restore.sh first" >&2
            exit 1
        fi
        git -C "$repo" apply -R "${reversed[@]}"
    fi

    git -C "$repo" add -A --intent-to-add
    local diff
    diff=$(git -C "$repo" diff --ignore-submodules=all --binary)
    git -C "$repo" reset -q

    if [[ -n "$diff" ]]; then
        printf '%s\n' "$diff" > "$cache"
        echo "$name: saved $(basename "$cache") ($(wc -l < "$cache") lines)"
    else
        rm -f "$cache"
        echo "$name: no changes beyond the named patches"
    fi
    [[ ${#named[@]} -gt 0 ]] && git -C "$repo" apply "${named[@]}"

    if [[ -n "$NAME" && -f "$cache" ]]; then
        local max=0 n
        for p in "${named[@]}"; do
            n=$((10#$(basename "$p" | grep -oE '^[0-9]{4}' || echo 0)))
            (( n < 1000 && n > max )) && max=$n
        done
        local dest
        dest=$pdir/$(printf "%04d" $((max + 1)))-$NAME.patch
        [[ -e "$dest" ]] && { echo "Error: $dest exists" >&2; exit 1; }
        mv "$cache" "$dest"
        echo "$name: promoted to $(basename "$dest")"
    fi
    return 0
}

store "$ROOT/Megatron-Bridge" "$ROOT/patches/megatron-bridge"
store "$ROOT/Megatron-Bridge/3rdparty/Megatron-LM" "$ROOT/patches/megatron-lm"
