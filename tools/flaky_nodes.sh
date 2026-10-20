#!/bin/bash
# Print (comma-separated, compressed) the nodes that recently dropped out of Slurm, for
# sbatch --exclude. Used by launch.sh and by every resubmission in train/job_body.sh.
#
# Why: since mid-October a set of ~70-100 Clariden nodes keeps going "DOWN* Not responding" for an
# instant (hundreds of events per day, mostly the same nodes); slurmctld then kills every job on
# them (NODE_FAIL), although the node and our processes are fine (jobs 3703471 and 3704232 lost
# their work since the last checkpoint this way). Excluding the nodes with >= 1 such event in the
# last 12 h or >= 3 in the last 48 h would have avoided 91% of those events over 18-20 Oct, at the
# cost of ~100 of 1343 nodes.
#
# Usage: tools/flaky_nodes.sh [recent_hours=12] [long_hours=48] [min_events_long=3]
# Never fails: prints nothing if the accounting database cannot be queried.
RECENT=${1:-12} LONG=${2:-48} K=${3:-3}
now=$(date +%s)
list=$(SLURM_TIME_FORMAT=%s timeout 60 sacctmgr -n -P show event \
        Start="$(date -d "-$LONG hours" +%Y-%m-%dT%H:%M)" End=now format=NodeName,TimeStart,Reason%120 2>/dev/null |
    awk -F'|' -v now="$now" -v r=$((RECENT * 3600)) -v l=$((LONG * 3600)) -v k="$K" '
        $3 ~ /Not responding/ && $2 ~ /^[0-9]+$/ {
            if (now - $2 <= r) bad[$1] = 1
            if (now - $2 <= l && ++cnt[$1] >= k) bad[$1] = 1
        }
        END { for (n in bad) print n }' | sort -u | paste -sd, -)
[ -n "$list" ] || exit 0
scontrol show hostlist "$list" 2>/dev/null || echo "$list"
