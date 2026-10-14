#!/usr/bin/env python3
"""
Run the kit's audio sanity checks (metrics.sanity: duration, silence share, speaking rate,
clipping) on an evaluation folder. Eval items have no `spoken` reference, so the character count
comes from the text that was synthesized (e.g. the normalized text), canonicalized like metrics.py.

    python code/sanity_eval.py predictions/t2_norm --text-csv predictions/t2_normalized.csv --text-column normalized_text
"""
import argparse, csv, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from synth import KIT  # noqa: E402
sys.path.insert(0, str(KIT))
import metrics as M  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--items", default=str(KIT / "data" / "eval_items.csv"))
    ap.add_argument("--text-csv")
    ap.add_argument("--text-column", default="normalized_text")
    a = ap.parse_args()
    items = list(csv.DictReader(open(a.items, encoding="utf-8")))
    text = {it["id"]: it["text"] for it in items}
    if a.text_csv:
        text.update({r["id"]: r[a.text_column] for r in csv.DictReader(open(a.text_csv, encoding="utf-8"))})
    bad = 0
    for it in items:
        p = Path(a.folder) / f"{it['id']}.wav"
        if not p.exists():
            print(f"{it['id']}: MISSING"); bad += 1; continue
        y, sr = M.load_audio(p)
        ok, reasons, st = M.sanity(y, 16000, len(M.canon(text[it["id"]])))
        flag = "ok " if ok and sr == 16000 else "BAD"
        bad += flag == "BAD"
        print(f"{flag} {it['id']:10s} dur {st['dur_s']:6.2f}s  silence {st.get('silence_share', 0):.2f}  "
              f"rate {st.get('chars_per_s', 0):5.1f} c/s  clip {st.get('clip_share', 0):.4f}  {'; '.join(reasons)}")
    print(f"\n{len(items) - bad}/{len(items)} pass the sanity checks")


if __name__ == "__main__":
    main()
