#!/usr/bin/env python3
"""Text-level check of the respeller on the English words of the validation items only.

Each Latin word in a hinglish validation item is paired with the word at the same position in the
item's `spoken` reference, and compared after the scorer's own canonicalization (metrics.canon:
nukta dropped, ऑ->ओ, nasal conjunct->anusvara). Prints per-word results and the character error
rate over the English words.
"""
import csv, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from synth import KIT  # noqa: E402  (finds the kit folder)
sys.path.insert(0, str(KIT))
from metrics import canon, _edit_distance  # noqa: E402
from translit import respell_word  # noqa: E402


def pairs():
    with open(KIT / "data" / "val_items.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            src = re.findall(r"[^\s।,?!.]+", r["text"])
            ref = re.findall(r"[^\s।,?!.]+", r["spoken"].split("||")[0])
            if len(src) != len(ref):
                continue
            for a, b in zip(src, ref):
                if re.search(r"[A-Za-z]", a) and not a.isupper():
                    yield r["id"], a, b


PROMPT_LOANWORDS = [("prompt_V3", "market", "मार्केट")]   # validation prompt transcript (V3)


def main():
    errs = tot = 0
    for iid, en, ref in list(pairs()) + PROMPT_LOANWORDS:
        hyp = respell_word(en)
        d = _edit_distance(canon(ref), canon(hyp))
        errs += d
        tot += len(canon(ref))
        print(f"{iid}  {en:10s} -> {hyp:10s}  ref {ref:10s}  {'OK' if d == 0 else f'{d} edit(s)'}")
    print(f"\nEnglish-word CER (text level, canon): {errs}/{tot} = {errs / max(tot, 1):.3f}")


if __name__ == "__main__":
    main()
