#!/usr/bin/env python3
"""Global-batch-size schedule arithmetic for launch.sh (standard library only).

Mirrors Megatron-Core's StepBatchsizeNumMicroBatchesCalculator, which Megatron-Bridge uses
for train.step_batch_size_schedule (patches/megatron-bridge/0001-step-batch-size-schedule.patch):
the schedule is "THRESHOLD:BATCH THRESHOLD:BATCH ...", thresholds in tokens (K/M/B/T suffixes),
converted to samples by integer division with the sequence length; an iteration that starts
with S consumed samples uses the batch size of the last threshold <= S.

Bridge counts the LR schedule in samples (lr_*_iters x global_batch_size, stepped by the
samples each iteration really consumed), with global_batch_size = the final batch size. So with
a schedule, warmup and decay "iterations" are given in units of final-size batches.

Prints shell assignments on stdout (for eval) and a human-readable table on stderr:
  SCHEDULE TRAIN_ITERS TRAIN_TOKENS LR_WARMUP_ITERS LR_DECAY_ITERS FINAL_GBS

Examples:
  batch_schedule.py --seq-length 4096 --train-tokens 1T --ramp 512:2560:4:3%
  batch_schedule.py --seq-length 4096 --train-tokens 500B --schedule "0:512 10B:1024 20B:2048"
"""

import argparse
import sys

SUFFIX = {"K": 10**3, "M": 10**6, "B": 10**9, "T": 10**12}


def parse_number(text):
    """'1T', '500B', '2.5e12', '1048576' -> int."""
    text = text.strip().upper()
    mult = SUFFIX.get(text[-1:], 1)
    if mult != 1:
        text = text[:-1]
    return int(float(text) * mult)


def human(tokens):
    for unit, value in (("T", 10**12), ("B", 10**9), ("M", 10**6)):
        if tokens >= value:
            return "%.4g%s" % (tokens / value, unit)
    return str(tokens)


def parse_schedule(text, seq_length):
    """Same parsing as Megatron-Core: sorted [(threshold_samples, batch_size)]."""
    entries = []
    for item in text.replace(",", " ").split():
        threshold, batch = item.split(":", 1)
        entries.append((parse_number(threshold) // seq_length, parse_number(batch)))
    entries.sort()
    if not entries or entries[0][0] != 0:
        sys.exit("schedule must start at threshold 0")
    if any(b[0] <= a[0] for a, b in zip(entries, entries[1:])):
        sys.exit("schedule thresholds must be strictly increasing (in samples)")
    return entries


def ramp_schedule(spec, train_tokens):
    """'START:FINAL:STEPS:RAMP' -> schedule string with STEPS equal increments over RAMP tokens.

    RAMP is a token count or a percentage of the training tokens (DeepSeek-V3 ramped its batch
    size over the first 469B of 14.8T tokens, ~3%).
    """
    start, final, steps, ramp = spec.split(":")
    start, final, steps = parse_number(start), parse_number(final), int(steps)
    if ramp.endswith("%"):
        if train_tokens is None:
            sys.exit("a ramp given in % of the run needs --train-tokens")
        ramp_tokens = int(train_tokens * float(ramp[:-1]) / 100)
    else:
        ramp_tokens = parse_number(ramp)
    if steps < 1 or final <= start or (final - start) % steps:
        sys.exit("ramp: FINAL-START must be a positive multiple of STEPS")
    inc = (final - start) // steps
    return " ".join("%d:%d" % (k * ramp_tokens // steps, start + k * inc) for k in range(steps + 1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seq-length", type=int, required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--schedule", help='"THRESHOLD:BATCH ..." in tokens')
    g.add_argument("--ramp", help="START:FINAL:STEPS:RAMP_TOKENS_OR_PERCENT, e.g. 512:2560:4:3%%")
    g2 = p.add_mutually_exclusive_group(required=True)
    g2.add_argument("--train-tokens", help="token budget, e.g. 1T; training stops after the batch reaching it")
    g2.add_argument("--train-iters", type=int, help="iteration budget instead of tokens")
    p.add_argument("--warmup-tokens", default="0", help="LR warmup length in tokens")
    p.add_argument("--divisor", type=int, default=1, help="micro-batch size x data-parallel size")
    a = p.parse_args()

    train_tokens = parse_number(a.train_tokens) if a.train_tokens else None
    text = a.schedule if a.schedule else ramp_schedule(a.ramp, train_tokens)
    sched = parse_schedule(text, a.seq_length)
    final = sched[-1][1]
    bad = [b for _, b in sched if b % a.divisor]
    if bad:
        sys.exit("batch sizes %s are not multiples of micro-batch x data-parallel size = %d" % (bad, a.divisor))

    def gbs_at(samples):
        batch = sched[0][1]
        for threshold, b in sched:
            if samples >= threshold:
                batch = b
            else:
                break
        return batch

    # Simulate the iterations (same loop as Megatron-LM's update_train_iters).
    target = train_tokens // a.seq_length if train_tokens is not None else None
    consumed, iters, starts = 0, 0, []
    while (consumed < target) if target is not None else (iters < a.train_iters):
        b = gbs_at(consumed)
        if not starts or starts[-1][2] != b:
            starts.append((iters + 1, consumed, b))
        consumed += b
        iters += 1

    warmup_samples = parse_number(a.warmup_tokens) // a.seq_length
    print("SCHEDULE='%s'" % text)
    print("TRAIN_ITERS=%d" % iters)
    print("TRAIN_TOKENS=%d" % (consumed * a.seq_length))
    print("LR_WARMUP_ITERS=%d" % round(warmup_samples / final))
    print("LR_DECAY_ITERS=%d" % max(1, round(consumed / final)))
    print("FINAL_GBS=%d" % final)

    out = sys.stderr
    out.write("global batch size schedule (seq %d):\n" % a.seq_length)
    for it, samples, b in starts:
        out.write("  from iteration %8d  (%9s tokens): %5d sequences = %5.2fM tokens/batch, %d microbatches/rank\n"
                  % (it, human(samples * a.seq_length), b, b * a.seq_length / 1e6, b // a.divisor))
    out.write("  total: %d iterations, %s tokens; LR warmup %s tokens, decay over all of them\n"
              % (iters, human(consumed * a.seq_length), human(warmup_samples * a.seq_length)))


if __name__ == "__main__":
    main()
