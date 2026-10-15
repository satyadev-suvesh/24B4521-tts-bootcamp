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
    asr = [k[4:] for k in rows[0] if k.startswith("cer_")]
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        name = re.sub(r"_s\d+$", "", Path(r["folder"]).name)
        for n in asr:
            acc[name][n].append(float(r[f"cer_{n}"]))
    print(f"{'text given to the model':28s}" + "".join(f"{n:>11s}" for n in asr) + f"{'mean':>8s}   (hinglish CER, 6 items x 3 seeds)")
    for key, label in GROUPS.items():
        if key in acc:
            m = [np.mean(acc[key][n]) for n in asr]
            print(f"{label:28s}" + "".join(f"{v:11.4f}" for v in m) + f"{np.mean(m):8.4f}")


if __name__ == "__main__":
    main()
