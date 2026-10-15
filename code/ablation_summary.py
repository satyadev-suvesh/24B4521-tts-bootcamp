#!/usr/bin/env python3
"""
Summarize the Task 3 ablation (run_ablation_t3.ps1): hinglish validation CER with Whisper and MMS
for Latin input (Task 2 text), the naive letter mapping, the respelling rules and the oracle
respelling, each averaged over seeds 0-2.

    python code/ablation_summary.py ../work/ablation_scores.csv
"""
import csv, re, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

GROUPS = {"val_control": "latin (Task 2 text)", "abl_naive": "naive letter mapping",
          "val_idea": "respelling rules (Task 3)", "abl_oracle": "oracle respelling"}


def main():
    rows = [r for r in csv.DictReader(open(sys.argv[1], encoding="utf-8")) if r["category"] == "hinglish"]
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        name = re.sub(r"_s\d+$", "", Path(r["folder"]).name)
        seed = re.search(r"_s(\d+)$", Path(r["folder"]).name)
        acc[name]["whisper"].append(float(r["cer_whisper"]))
        acc[name]["mms"].append(float(r["cer_mms"]))
        acc[name]["seeds"].append(int(seed.group(1)) if seed else 0)
    print(f"{'text given to the model':28s} {'Whisper':>8s} {'MMS':>8s} {'mean':>8s}   (hinglish CER, 6 items x 3 seeds)")
    for key, label in GROUPS.items():
        if key in acc:
            w, m = np.mean(acc[key]["whisper"]), np.mean(acc[key]["mms"])
            print(f"{label:28s} {w:8.4f} {m:8.4f} {(w + m) / 2:8.4f}")


if __name__ == "__main__":
    main()
