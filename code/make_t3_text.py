#!/usr/bin/env python3
"""
Task 3 text: the Task 2 normalizer, then the Indian-English respeller for every Latin word left.

    python code/make_t3_text.py --out code/t3_text.csv      # 64 rows: id,split,text,t3_text

The audio is then made exactly like Task 2 (baseline decoding, seed 0, T4 noise):
    python code/synth.py --items ../data/val_items.csv  --text-csv code/t3_text.csv --text-column t3_text --out predictions/val_idea --noise t4 --seed 0
    python code/synth.py --items ../data/eval_items.csv --text-csv code/t3_text.csv --text-column t3_text --out predictions/t3_idea --noise t4 --seed 0
"""
import argparse, csv, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from normalize import normalize  # noqa: E402
from translit import transliterate  # noqa: E402
from synth import KIT  # noqa: E402


def t3_text(text: str) -> str:
    return transliterate(normalize(text))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "t3_text.csv"))
    a = ap.parse_args()
    rows = []
    for split, name in (("val", "val_items.csv"), ("eval", "eval_items.csv")):
        with open(KIT / "data" / name, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append({"id": r["id"], "split": split, "text": r["text"], "t3_text": t3_text(r["text"])})
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "split", "text", "t3_text"])
        w.writeheader()
        w.writerows(rows)
    latin = [r["id"] for r in rows if re.search(r"[A-Za-z]", r["t3_text"])]
    print(f"wrote {len(rows)} rows to {a.out}; rows with Latin letters left: {latin or 'none'}")


if __name__ == "__main__":
    main()
