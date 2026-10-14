#!/usr/bin/env python3
"""
Per-word view of the English words in hinglish validation items: for each system's transcript
(metrics.py --csv output), align it to the `spoken` reference word by word and report what the
recognizer heard at each English word and that word's character error.

    python code/analyze_words.py baseline=work/baseline_val_scores.csv idea=work/val_idea.csv
"""
import csv, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from synth import KIT  # noqa: E402
sys.path.insert(0, str(KIT))
from metrics import canon, _edit_distance  # noqa: E402

WORD = re.compile(r"[^\s।,?!.]+")


def align(ref, hyp):
    """Word-level DP alignment; substitution cost = character error of the pair (0..1).
    Returns, for each ref word, the hyp words aligned to it (possibly several or none)."""
    R, H = [canon(w) for w in ref], [canon(w) for w in hyp]
    n, m = len(R), len(H)
    INF = 1e9
    D = [[INF] * (m + 1) for _ in range(n + 1)]
    B = [[None] * (m + 1) for _ in range(n + 1)]
    D[0][0] = 0
    for i in range(n + 1):
        for j in range(m + 1):
            if i < n and D[i][j] + 1 < D[i + 1][j]:                       # ref word deleted
                D[i + 1][j], B[i + 1][j] = D[i][j] + 1, (i, j, "del")
            if j < m and D[i][j] + 1 < D[i][j + 1]:                       # extra hyp word
                D[i][j + 1], B[i][j + 1] = D[i][j] + 1, (i, j, "ins")
            if i < n and j < m:
                c = min(1.0, _edit_distance(R[i], H[j]) / max(len(R[i]), 1))
                if D[i][j] + c < D[i + 1][j + 1]:
                    D[i + 1][j + 1], B[i + 1][j + 1] = D[i][j] + c, (i, j, "sub")
                # two hyp words for one ref word (word split by the recognizer)
                if j + 1 < m:
                    c2 = min(1.0, _edit_distance(R[i], H[j] + H[j + 1]) / max(len(R[i]), 1))
                    if D[i][j] + c2 < D[i + 1][j + 2]:
                        D[i + 1][j + 2], B[i + 1][j + 2] = D[i][j] + c2, (i, j, "sub2")
    out = [[] for _ in range(n)]
    i, j = n, m
    while (i, j) != (0, 0):
        pi, pj, op = B[i][j]
        if op == "sub":
            out[pi].append(hyp[pj])
        elif op == "sub2":
            out[pi] += [hyp[pj], hyp[pj + 1]]
        i, j = pi, pj
    return out


def main():
    systems = dict(arg.split("=", 1) for arg in sys.argv[1:])
    hyps = {name: {r["id"]: r["hyp"] for r in csv.DictReader(open(p, encoding="utf-8"))} for name, p in systems.items()}
    items = [r for r in csv.DictReader(open(KIT / "data" / "val_items.csv", encoding="utf-8")) if r["category"] == "hinglish"]
    totals = {name: [0, 0] for name in systems}
    for it in items:
        src = WORD.findall(it["text"])
        ref = WORD.findall(it["spoken"].split("||")[0])
        eng = [k for k, w in enumerate(src) if re.search(r"[A-Za-z]", w)]
        print(f"\n{it['id']}: {it['text']}")
        aligned = {name: align(ref, WORD.findall(h.get(it["id"], ""))) for name, h in hyps.items()}
        for k in eng:
            row = f"  {src[k]:10s} ref {ref[k]:10s}"
            for name in systems:
                heard = " ".join(aligned[name][k]) or "∅"
                e = min(len(canon(ref[k])), _edit_distance(canon(ref[k]), canon(heard.replace("∅", ""))))
                totals[name][0] += e
                totals[name][1] += len(canon(ref[k]))
                row += f" | {name}: {heard} ({e})"
            print(row)
    print("\nEnglish-word character error (sum of per-word edits / reference chars):")
    for name, (e, n) in totals.items():
        print(f"  {name:10s} {e}/{n} = {e / max(n, 1):.3f}")


if __name__ == "__main__":
    main()
