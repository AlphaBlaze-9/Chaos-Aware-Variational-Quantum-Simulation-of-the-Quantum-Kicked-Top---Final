"""
clear_n10_entries.py -- clears ONLY the N=10 entries from the machine-tagged
checkpoint, leaving N=4/6/8 (all complete and good) untouched.

Use this after raising TIME_BUDGET_TABLE[10] in large_scale_scaling_v4.py --
otherwise the script will silently restore the stale nan/0-step (and thin
3-step / 1-step) N=10 results from checkpoint instead of retrying them with
the new budget.

Usage:
    python clear_n10_entries.py
"""
import json
import os
import glob

# Auto-detect the machine-tagged checkpoint file in the current directory's
# figures/ folder, so you don't have to hardcode the machine name.
candidates = glob.glob("figures/depth_scaling_checkpoint__*.json")
if not candidates:
    print("No checkpoint file found matching figures/depth_scaling_checkpoint__*.json "
          "-- nothing to clear (N=10 will compute fresh on next run anyway).")
    raise SystemExit(0)
if len(candidates) > 1:
    print(f"Found multiple checkpoint files: {candidates}")
    print("Edit CHECKPOINT_PATH below to pick the right one explicitly, then rerun.")
    raise SystemExit(1)

CHECKPOINT_PATH = candidates[0]
print(f"Using checkpoint file: {CHECKPOINT_PATH}")

# Only N=10 entries -- N=4/6/8 are complete and shouldn't be recomputed.
KEYS_TO_CLEAR = ["10,2.5", "10,0.5", "10,1.5"]


def main():
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

    print(f"Removed {len(removed)} entr{'y' if len(removed)==1 else 'ies'}: {removed}")
    if missing:
        print(f"(Not found, already absent: {missing})")
    print(f"Remaining cached entries: {sorted(results.keys())}")
    print("\nN=4/6/8 will restore from checkpoint (unchanged). "
          "N=10 will recompute fresh under whatever TIME_BUDGET_TABLE[10] "
          "you've now set.")


if __name__ == "__main__":
    main()
