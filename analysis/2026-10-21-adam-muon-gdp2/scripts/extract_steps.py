"""Per-step metrics of the three 8-node runs from the job logs (last rank's iteration lines) -> data/steps.csv."""
import csv, re, sys
from pathlib import Path

LOGS = Path(__file__).resolve().parents[3] / "logs"
RUNS = {  # label: job log
    "adam": "mbcscs-train-q35a3b-gbs512to2560-sl4096-3705060.log",
    "muon": "mbcscs-train-q35a3b-gbs512to2560-sl4096-muon-max500-3706874.log",
    "muon-zero1-gdp2": "mbcscs-train-q35a3b-gbs512to2560-sl4096-muon-zero1-gdp2-max500-3708048.log",
}
FIELDS = {"step_ms": r"elapsed time per iteration \(ms\): ([\d.]+)", "lr": r"learning rate: ([\dE.+-]+)",
          "lm_loss": r"lm loss: ([\dE.+-]+)", "mtp_loss": r"mtp_1 loss: ([\dE.+-]+)",
          "gbal_loss": r"global_load_balancing_loss: ([\dE.+-]+)", "grad_norm": r"grad norm: ([\d.]+)",
          "tflops": r"TFLOP/s/GPU\): ([\d.]+)"}
out = Path(__file__).resolve().parents[1] / "data" / "steps.csv"
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["run", "iter", "time"] + list(FIELDS))
    for run, log in RUNS.items():
        last = {}
        for line in open(LOGS / log, errors="replace"):
            if " iteration " not in line or not line.lstrip().startswith("31:"):
                continue
            m = re.search(r"\[([\d-]+ [\d:]+)\] iteration +(\d+)/", line)
            if not m:  # warnings and validation lines that contain " iteration "
                continue
            row = [run, int(m.group(2)), m.group(1)]
            for rx in FIELDS.values():
                v = re.search(rx, line)
                row.append(v.group(1) if v else "")
            last[row[1]] = row  # a repeated iteration (after a restart) keeps the last record
        for it in sorted(last):
            w.writerow(last[it])
        print(run, len(last), "steps")
