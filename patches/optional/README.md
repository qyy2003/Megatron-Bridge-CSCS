# Optional patch sets (per job)

`patches/<tree>/*.patch` are applied to every job. A directory `patches/optional/<set>/` holds
patches that only jobs launched with `PATCHES=<set>` get:

```
patches/optional/<set>/megatron-bridge/NNNN-name.patch   # inside Megatron-Bridge/
patches/optional/<set>/megatron-lm/NNNN-name.patch       # inside Megatron-Bridge/3rdparty/Megatron-LM/
```

- Applied after the always-on patches of the same tree, sets in the order given
  (`PATCHES=a,b`), files in name order. Same header convention as the other patches.
- Each combination builds its own content-addressed code snapshot, so jobs with different sets
  can be queued and run at the same time; the run name gets `-p<set>` (train runs therefore
  never share checkpoints across sets).
- Write one with the usual workflow: edit the submodule working tree, `git -C <tree> diff > file`,
  then restore the working tree (`./code_store.sh` would make it an always-on patch instead).

Example: `PATCHES=my-change EXP_SUFFIX=a ./launch.sh train 8 1T` next to the same command
without `PATCHES`. A change that is only a config value (e.g. the optimizer: Megatron-Core
already has Muon, `optimizer.optimizer=muon` + `optimizer.muon_*`) needs no patch: use `EXTRA=`.

Sets in this repository:

| Set | What | Used by |
|---|---|---|
| `muon-zero1` | ZeRO-1 for Megatron-Core's layer-wise Muon (shard-aligned buffers) and optimizer instances (GLOBAL_DP) | `OPTIMIZER=muon-zero1` |
| `dion` | Dion (own implementation) with compressed cross-instance sync; needs `muon-zero1` first | `OPTIMIZER=dion-zero1` |
| `muon-spectral` | Muon with an exact, reweighted spectrum (polar / power / top-fraction) for the spectral experiments | `MUON_SPECTRAL=...` |
