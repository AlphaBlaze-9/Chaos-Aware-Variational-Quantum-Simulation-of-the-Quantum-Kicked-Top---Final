"""
clear_checkpoint_entries.py -- selectively force specific (N,k) points in
figures/depth_scaling_checkpoint.json to recompute, instead of clearing the
whole file (which would waste the N=4 points -- those already have full
10/10-step coverage and don't need to be redone).

Usage:
    python clear_checkpoint_entries.py

Edit KEYS_TO_CLEAR below if you want a different subset.
"""
import json
import os

CHECKPOINT_PATH = "figures/depth_scaling_checkpoint.json"

# These are exactly the points that are currently incomplete/uninformative:
#   8,2.5   -- chaotic,     only 1/10 steps (lower bound only)
#   8,1.5   -- regular-alt, only 1/10 steps (the single trivially-shallow
#              step your own paper text already says isn't informative)
#   6,2.5   -- chaotic,     only 4/10 steps
#   6,1.5   -- regular-alt, only 7/10 steps
#   10,2.5  -- chaotic,     0/10 steps (NaN -- the 50-restart attempt never
#              even finished t=1; this is the one the paper's Table III
#              footnote already documents as falling back to a separate
#              12-restart run)
#   10,0.5  -- regular,     only 3/10 steps
#   10,1.5  -- regular-alt, only 1/10 steps
#
# NOT included: 4,2.5 / 4,0.5 / 4,1.5 -- already 10/10 steps, fully
# converged, no reason to burn compute recomputing these.
KEYS_TO_CLEAR = [
    "6,2.5", "6,1.5",
    "8,2.5", "8,1.5",
    "10,2.5", "10,0.5", "10,1.5",
]


def main():
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"No checkpoint file found at {CHECKPOINT_PATH} -- nothing to clear "
              f"(everything will compute fresh on next run anyway).")
        return

    with open(CHECKPOINT_PATH) as fh:
        data = json.load(fh)

    results = data.get("results", {})
    removed, missing = [], []
    for key in KEYS_TO_CLEAR:
        if key in results:
            del results[key]
            removed.append(key)
        else:
            missing.append(key)

    data["results"] = results
    with open(CHECKPOINT_PATH, "w") as fh:
        json.dump(data, fh, indent=2)

    print(f"Removed {len(removed)} entr{'y' if len(removed)==1 else 'ies'} "
          f"from checkpoint: {removed}")
    if missing:
        print(f"(Not found, already absent: {missing})")
    print(f"Remaining cached entries: {sorted(results.keys())}")
    print("\nNext run of large_scale_scaling.py will recompute exactly the "
          "entries removed above, and reuse everything else.")


if __name__ == "__main__":
    main()
