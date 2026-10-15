#!/usr/bin/env python3
"""
Task 3 analysis (not part of the submitted system): how much of the gain comes from the script
alone, and how far the respelling rules are from the best possible respelling.

Three texts for the 6 hinglish validation items, all after the Task 2 normalizer:
  naive  - every Latin letter mapped to one fixed Devanagari letter (a->अ/ा, c->क, t->त, ...), no
           English spelling knowledge at all: tests "Devanagari script alone"
  rules  - the Task 3 respeller (code/translit.py), i.e. the submitted system
  oracle - the item's `spoken` reference (how the words should sound, in Devanagari): the best
           any respelling could do with this model

    python code/ablation_t3.py --out ../work/ablation_text.csv
then synthesize each column with code/synth.py (baseline decoding, T4 noise) and score with
eval_extra.py / analyze_words.py.
"""
import argparse, csv, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from normalize import normalize  # noqa: E402
from translit import transliterate  # noqa: E402
from synth import KIT  # noqa: E402

# one Devanagari letter per Latin letter (consonants) / (independent vowel, matra)
NAIVE_C = {"b": "ब", "c": "क", "d": "द", "f": "फ", "g": "ग", "h": "ह", "j": "ज", "k": "क", "l": "ल",
           "m": "म", "n": "न", "p": "प", "q": "क", "r": "र", "s": "स", "t": "त", "v": "व", "w": "व",
           "x": "क्स", "y": "य", "z": "ज"}
NAIVE_V = {"a": ("अ", ""), "e": ("ए", "े"), "i": ("इ", "ि"), "o": ("ओ", "ो"), "u": ("उ", "ु")}


def naive_word(w):
    out, prev_c = [], False
    for ch in w.lower():
        if ch in NAIVE_V:
            ind, matra = NAIVE_V[ch]
            out.append(matra if prev_c else ind)
            prev_c = False
        elif ch in NAIVE_C:
            if prev_c:
                out.append("्")
            out.append(NAIVE_C[ch])
            prev_c = True
    return "".join(out)


def naive(text):
    return re.sub(r"[A-Za-z]+", lambda m: naive_word(m.group(0)), text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    with open(KIT / "data" / "val_items.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["category"] != "hinglish":
                continue
            n = normalize(r["text"])
            rules = transliterate(n)
            # oracle: the rules text with each word replaced by the reference word at the same
            # position (punctuation kept, so durations are computed from comparable text)
            words, spoken = rules.split(), r["spoken"].split()
            assert len(words) == len(spoken), r["id"]
            oracle = " ".join(re.sub(r"^[^\s।,?!.]+", s, w) for w, s in zip(words, spoken))
            rows.append({"id": r["id"], "naive": naive(n), "rules": rules, "oracle": oracle})
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "naive", "rules", "oracle"])
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r["id"], "|", r["naive"], "|", r["rules"], "|", r["oracle"])


if __name__ == "__main__":
    main()
