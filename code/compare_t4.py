#!/usr/bin/env python3
"""
Compare Task 4 sweep settings with the Task 3 system, item by item and seed by seed.

Input: the per-item CSV of eval_extra.py over the Task 3 folders (one per seed) and the sweep
folders (<name>_s<seed>). For each setting it prints the mean CER (Whisper, MMS and their
average), the ratio to the Task 3 system at the same seeds, SIM, time, speed-up, and the risk
that a 40-item set (like the evaluation set) ends above the 1.2x CER limit: pairs (item, seed)
are resampled 40 at a time and the CER ratio is recomputed (paired bootstrap).

    python code/compare_t4.py ../work/t4sweep_scores.csv --t3 val_idea,val_idea_s1,val_idea_s2 \
        --timing-dir ../work/t4sweep --baseline-s 436.1
"""
import argparse, csv, json, re
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scores_csv")
    ap.add_argument("--t3", required=True, help="Task 3 folder names for seeds 0,1,2 (last path part)")
    ap.add_argument("--timing-dir")
    ap.add_argument("--baseline-s", type=float)
    ap.add_argument("--n-eval", type=int, default=40)
    ap.add_argument("--boot", type=int, default=20000)
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.scores_csv, encoding="utf-8")))
    by = defaultdict(dict)                       # folder name -> id -> row
    for r in rows:
        by[Path(r["folder"]).name][r["id"]] = r
    t3 = a.t3.split(",")
    ref = {s: by[name] for s, name in enumerate(t3)}
    groups = defaultdict(dict)                   # config -> seed -> folder rows
    for name, d in by.items():
        m = re.fullmatch(r"(.+)_s(\d+)", name)
        if m and name not in t3:
            groups[m.group(1)][int(m.group(2))] = d
    groups = {"t3 (fp32, 32 steps)": ref, **dict(sorted(groups.items()))}

    rng = np.random.default_rng(0)
    f = lambda r, k: float(r[k])
    print(f"{'setting':22s} {'seeds':>5s} {'CER_w':>7s} {'CER_mms':>7s} {'CER_avg':>7s} {'ratio':>6s} "
          f"{'hgl_avg':>7s} {'SIM_e':>6s} {'SIM_w':>6s} {'time_s':>7s} {'speedup':>7s} {'P(>1.2)':>7s}")
    for cfg, seeds in groups.items():
        pairs_x, pairs_r = [], []
        cw, cm, se, sw, hg = [], [], [], [], []
        for s, d in seeds.items():
            if s not in ref:
                continue
            for iid, r in d.items():
                x = (f(r, "cer_whisper") + f(r, "cer_mms")) / 2
                rr = ref[s][iid]
                pairs_x.append(x)
                pairs_r.append((f(rr, "cer_whisper") + f(rr, "cer_mms")) / 2)
                cw.append(f(r, "cer_whisper")); cm.append(f(r, "cer_mms"))
                se.append(f(r, "sim_ecapa")); sw.append(f(r, "sim_wavlm"))
                if r["category"] == "hinglish":
                    hg.append(x)
        if not pairs_x:
            continue
        px, pr = np.array(pairs_x), np.array(pairs_r)
        idx = rng.integers(0, len(px), size=(a.boot, a.n_eval))
        ratio_b = px[idx].sum(1) / np.maximum(pr[idx].sum(1), 1e-9)
        t = []
        if a.timing_dir:
            for s in seeds:
                p = Path(a.timing_dir) / f"{cfg}_s{s}_timing.json"
                if p.exists():
                    t.append(json.loads(p.read_text(encoding="utf-8"))["total_synth_s"])
        tm = float(np.mean(t)) if t else float("nan")
        sp = a.baseline_s / tm if (a.baseline_s and t) else float("nan")
        print(f"{cfg:22s} {len(seeds):>5d} {np.mean(cw):7.4f} {np.mean(cm):7.4f} {px.mean():7.4f} "
              f"{px.sum() / pr.sum():6.3f} {np.mean(hg):7.4f} {np.mean(se):6.4f} {np.mean(sw):6.4f} "
              f"{tm:7.1f} {sp:7.2f} {(ratio_b > 1.2).mean():7.3f}")


if __name__ == "__main__":
    main()
