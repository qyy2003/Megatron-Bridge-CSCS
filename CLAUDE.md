# Instructions for coding agents

Read `HANDOVER.md` first: state, where things are, how to restart, and what is in progress. Then
`README.md` (usage) and `WORKLOG.md` (history, with Slurm job IDs).

Note: `Megatron-Bridge/` and `Megatron-Bridge/3rdparty/Megatron-LM/` are pinned upstream
submodules with their own `CLAUDE.md`. Change them only through patch files in `patches/` (see
`How_To_Edit.md`), never by committing inside the submodules.

## Working rules (from the user)

1. **Slurm only.** Never run experiments on the login node (GPU work, tokenization, image builds,
   training): use `sbatch`/`srun` (account in `config.sh`). The login node is fine for file copying,
   downloads and log analysis.
2. **Check before you submit; the queue is slow.** Before every submission:
   - static checks: `bash -n`, `python -m py_compile`, unit tests;
   - `SUBMIT=0 ./launch.sh …` to write the job script without submitting, then inspect it: nodes,
     time, `train_iters`, resubmits, and no W&B key (`grep -cF "$WANDB_API_KEY"` must be 0).

   Then start with the smallest job (debug partition) before scaling up.
3. **Keep `WORKLOG.md` current.** Log every change, decision, submission and result immediately,
   including failures and preliminary numbers. Mark rows "submitted"/"running" and update them when
   results arrive.
4. **Commit only at big milestones.** One commit per milestone, no incremental commits. Never push
   unless the user asks. Commit messages end with a `Co-Authored-By:` trailer naming the agent.
5. **Inspect before anything destructive.** Look at the targets read-only first, and prefer a
   non-destructive alternative. Split inspect → targeted delete (only if needed) → dry run → submit.
   Never bundle `rm -rf` into launch commands.
6. **Production runs need full verification.** Launch an expensive production run only when it is
   fully verified (lifecycle, memory at the final batch, logging volume), and leave a markdown
   overview of it (like `PRODUCTION_RUN.md`).
7. **Fix every bug you find and analyse every result.** Report outcomes as they are: failed tests,
   out-of-memory errors, skipped steps.

## Secrets

The W&B key lives only in `config.sh`, which is git-ignored. Never print it or write it into
committed files or generated job scripts. Scan staged changes before committing:
`git diff --cached | grep -cF "$WANDB_API_KEY"` must be 0.
