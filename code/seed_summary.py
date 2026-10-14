"""Summarize the Task 2 control vs the Task 3 idea over seeds 0-2 (scores written by metrics.py).

    python submission/code/seed_summary.py      (run from the kit folder)
"""
import json

rows = []
for name in ["val_control", "val_idea"]:
    for s in [0, 1, 2]:
        p = f"work/{name}.json" if s == 0 else f"work/{name}_s{s}.json"
        d = json.load(open(p, encoding="utf-8"))
        rows.append((name, s, d["cer"], d["cer_by_category"]["hinglish"], d["sim"]))
for r in rows:
    print(f"{r[0]:12s} seed {r[1]}: CER {r[2]:.4f}  hinglish {r[3]:.4f}  SIM {r[4]:.4f}")
for name in ["val_control", "val_idea"]:
    sel = [r for r in rows if r[0] == name]
    print(f"{name:12s} mean: CER {sum(r[2] for r in sel) / 3:.4f}  hinglish {sum(r[3] for r in sel) / 3:.4f}"
          f"  SIM {sum(r[4] for r in sel) / 3:.4f}")
