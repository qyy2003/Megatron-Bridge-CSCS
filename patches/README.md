# Patches

- `megatron-bridge/*.patch` apply inside `Megatron-Bridge/` (pinned at v0.6.2)
- `megatron-lm/*.patch` apply inside `Megatron-Bridge/3rdparty/Megatron-LM/` (pinned at a0f793d)

Applied in file-name order on top of the pinned commits by every job (see `How_To_Edit.md`).
`1000-cache.patch` is private WIP written by `./code_store.sh` and is git-ignored.

Currently empty: none of the GH200 fixes of the earlier Megatron-LM project are needed with
these versions (details in `How_To_Edit.md`).
