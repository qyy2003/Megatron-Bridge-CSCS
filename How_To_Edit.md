# How to edit Megatron-Bridge / Megatron-LM

Both upstream trees are pinned git submodules (`env/versions.env`):

| Tree | Pinned at | Patches |
|---|---|---|
| `Megatron-Bridge/` | `v0.6.2` (`c0e164e`) | `patches/megatron-bridge/*.patch` |
| `Megatron-Bridge/3rdparty/Megatron-LM/` | `a0f793d` (pinned by Bridge v0.6.2) | `patches/megatron-lm/*.patch` |

**Patches are the canonical home for changes.** A job never runs your working trees: at
start it builds a content-addressed snapshot (`$MBCSCS_IOPS/code/mb-<sha>_mlm-<sha>_p-<patch hash>`)
from the pinned commits plus every patch, and mounts it at `/opt/Megatron-Bridge` in the
container (over the image's own copy, the way NVIDIA's Lightning recipe overlays Bridge).
Same patches = same snapshot, so jobs, resubmissions and reruns share it; each job log prints
which snapshot it used.

## Workflow

```bash
./code_restore.sh                     # pinned + all patches applied to the working trees
<edit Megatron-Bridge/... and/or Megatron-Bridge/3rdparty/Megatron-LM/...>
./code_store.sh                       # WIP -> patches/<tree>/1000-cache.patch (per tree)
./launch.sh smoke                     # the job picks the cache up
./code_store.sh -N my-feature         # done: promote -> patches/<tree>/0NNN-my-feature.patch
git add patches/ && git commit
```

| File | Purpose | In git? |
|---|---|---|
| `patches/<tree>/1000-cache.patch` | private WIP, overwritten by every `code_store.sh` | no |
| `patches/<tree>/0NNN-name.patch` | named, reviewed, applied in order | yes |

Script options:

```bash
./code_restore.sh        # refuses if a tree has uncommitted changes
./code_restore.sh -f     # discard them (tracked files + files created by patches), then apply
./code_store.sh          # save WIP of both trees
./code_store.sh -N name  # save, then promote each tree's cache to the next free 0NNN-name.patch
```

`code_store.sh` includes new files, ignores the nested submodule pointer, subtracts the named
patches for you and leaves the working trees as they were. `launch.sh` refuses to submit if a
patch does not apply to the pinned commit (checked against a throw-away index, independent of
your working tree).

## Writing a patch header

Put `#` comment lines before the first `diff --git` line (git apply ignores them):

```
# Patch: <one line>
#
# Problem:   what goes wrong, with the symptom as it appears in the log
# What:      file / function changed, and how
# Porting:   how to find the spot again if line numbers move in a newer release,
#            and when the patch can be dropped
#
# Applied to: Megatron-Bridge v0.6.2 (or Megatron-LM a0f793d)
```

## Upgrading

1. `git -C Megatron-Bridge fetch --tags && git -C Megatron-Bridge checkout <new tag> &&
   git -C Megatron-Bridge submodule update --init --recursive`
2. Update `MBRIDGE_*`, `MLM_COMMIT` and the matching `NEMO_TAG` in `env/versions.env`
   (rebuild the image), then `./code_restore.sh` and fix patches that no longer apply,
   using their headers.

## Which patches were carried over from the Megatron-LM project

None were needed. Checked against this Megatron-LM (`a0f793d`) and Bridge `v0.6.2`:

| Old patch (core_v0.16.1) | Status here |
|---|---|
| 0001 tokens/sec/GPU logging | Bridge logs `throughput/device/tokens_per_sec` (and TFLOP/s/GPU) to W&B/TB with `logger.log_throughput_to_tensorboard` (on in `conf/`) |
| 0002 checkpoint SIGSEGV (fork per write bucket) | the async writer now uses threads, no child processes |
| 0003 NVML energy monitor SIGSEGV | Bridge only creates the monitor with `logger.log_energy=true` (off) |
| 0004 distrib-optimizer `padding` key on load | upstream now skips non-tensor keys in `_set_main_param_and_optimizer_states` |
