#!/usr/bin/env python3
"""
Task 2 - rule-based Hindi text normalizer (no models).

    from normalize import normalize
    normalize("आपके खाते में 12/03/2026 को ₹1,25,000 जमा हुए हैं।")
    # -> "आपके खाते में बारह मार्च दो हज़ार छब्बीस को एक लाख पच्चीस हज़ार रुपये जमा हुए हैं।"

    # write predictions/t2_normalized.csv for all 64 items (24 val + 40 eval)
    python code/normalize.py --out predictions/t2_normalized.csv

Rewrites numbers, dates, rupee amounts, percentages and all-caps abbreviations into Hindi words
in Devanagari, as they are spoken. English words in Latin script are left unchanged (Task 2 rule).

Conventions, all taken from the validation items (data/val_items.csv, `spoken` column):
  * dates are dd/mm/yyyy -> "<day> <month name> <year>"                    (val_hd_01, _03, _08)
  * "₹<amount> [लाख|करोड़]" -> "<amount words> [लाख|करोड़] रुपये"            (val_hd_01, _02, _04)
  * "₹<rupees>.<paise>" -> "<rupees> रुपये <paise> पैसे"                     (val_hd_08)
  * amounts carry comma grouping (₹2,340.50; 1,47,382), so a comma-less number of 4+ digits is a
    year (1100-2099) or an identifier read digit by digit                  (val_hd_05, _07)
  * all-caps abbreviations are spelled letter by letter: RBI -> आर बी आई    (val_hd_03, _04, _07)
The rest (number words, month names, letter names, Indian lakh/crore grouping, decimals) is
general Hindi convention, not a word list read off any item.
"""
import argparse
import csv
import re
import sys
import unicodedata
from pathlib import Path

# ----------------------------------------------------------------------------------------------
# Word tables (general Hindi conventions)
# ----------------------------------------------------------------------------------------------
ONES = ("शून्य एक दो तीन चार पाँच छह सात आठ नौ दस ग्यारह बारह तेरह चौदह पंद्रह सोलह सत्रह अठारह "
        "उन्नीस बीस इक्कीस बाईस तेईस चौबीस पच्चीस छब्बीस सत्ताईस अट्ठाईस उनतीस तीस इकतीस बत्तीस "
        "तैंतीस चौंतीस पैंतीस छत्तीस सैंतीस अड़तीस उनतालीस चालीस इकतालीस बयालीस तैंतालीस चवालीस "
        "पैंतालीस छियालीस सैंतालीस अड़तालीस उनचास पचास इक्यावन बावन तिरपन चौवन पचपन छप्पन सत्तावन "
        "अट्ठावन उनसठ साठ इकसठ बासठ तिरसठ चौंसठ पैंसठ छियासठ सड़सठ अड़सठ उनहत्तर सत्तर इकहत्तर "
        "बहत्तर तिहत्तर चौहत्तर पचहत्तर छिहत्तर सतहत्तर अठहत्तर उन्यासी अस्सी इक्यासी बयासी तिरासी "
        "चौरासी पचासी छियासी सत्तासी अट्ठासी नवासी नब्बे इक्यानवे बानवे तिरानवे चौरानवे पचानवे "
        "छियानवे सत्तानवे अट्ठानवे निन्यानवे").split()
assert len(ONES) == 100

MONTHS = [None, "जनवरी", "फ़रवरी", "मार्च", "अप्रैल", "मई", "जून", "जुलाई", "अगस्त", "सितंबर",
          "अक्टूबर", "नवंबर", "दिसंबर"]

# English letter names as said in Indian English, in Devanagari
LETTERS = dict(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                   "ए बी सी डी ई एफ़ जी एच आई जे के एल एम एन ओ पी क्यू आर एस टी यू वी डब्ल्यू एक्स वाई ज़ेड".split()))

# ordinal forms that are irregular (1st..6th); the rest are cardinal + वाँ/वीं/वें
ORDINAL_STEM = {1: "पहल", 2: "दूसर", 3: "तीसर", 4: "चौथ", 6: "छठ"}

SCALE_WORDS = "लाख|करोड़|करोड|हज़ार|हजार|अरब"
DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
NUKTA_FORMS = {"क़": "क़", "ख़": "ख़", "ग़": "ग़", "ज़": "ज़",
               "ड़": "ड़", "ढ़": "ढ़", "फ़": "फ़", "य़": "य़"}

# digit runs are matched without \b because digits and Devanagari letters are both \w
NUM_L, NUM_R = r"(?<![\d.,])", r"(?![\d])"


# ----------------------------------------------------------------------------------------------
# Number reading
# ----------------------------------------------------------------------------------------------
def int_words(n: int) -> str:
    """Cardinal in the Indian system: 1,25,000 -> एक लाख पच्चीस हज़ार."""
    if n < 100:
        return ONES[n]
    parts = []
    crore, n = divmod(n, 10 ** 7)
    if crore:
        parts.append(f"{int_words(crore)} करोड़")
    lakh, n = divmod(n, 10 ** 5)
    if lakh:
        parts.append(f"{ONES[lakh]} लाख")
    thousand, n = divmod(n, 1000)
    if thousand:
        parts.append(f"{ONES[thousand]} हज़ार")
    hundred, n = divmod(n, 100)
    if hundred:
        parts.append(f"{ONES[hundred]} सौ")
    if n:
        parts.append(ONES[n])
    return " ".join(parts)


def year_words(n: int) -> str:
    """Years: 1947 -> उन्नीस सौ सैंतालीस; 2026 -> दो हज़ार छब्बीस."""
    if 1100 <= n <= 1999:
        hi, lo = divmod(n, 100)
        return f"{ONES[hi]} सौ" + (f" {ONES[lo]}" if lo else "")
    return int_words(n)


def digit_words(s: str) -> str:
    """Digit by digit: 12951 -> एक दो नौ पाँच एक."""
    return " ".join(ONES[int(d)] for d in s if d.isdigit())


def decimal_words(int_part: str, frac: str) -> str:
    """8.25 -> आठ दशमलव दो पाँच."""
    return f"{int_words(int(int_part.replace(',', '') or 0))} दशमलव {digit_words(frac)}"


def plain_number_words(s: str) -> str:
    """A bare number token. Grouped numbers are quantities; comma-less 4+ digit numbers are
    years (1100-2099) or identifiers, read digit by digit."""
    if "," in s:
        return int_words(int(s.replace(",", "")))
    if len(s) >= 4:
        n = int(s)
        if len(s) == 4 and 1100 <= n <= 2099:
            return year_words(n)
        return digit_words(s)
    if len(s) > 1 and s.startswith("0"):
        return digit_words(s)
    return int_words(int(s))


# ----------------------------------------------------------------------------------------------
# Rewrite rules, applied in order
# ----------------------------------------------------------------------------------------------
def _date(m):
    d, mo, y = int(m.group(1)), int(m.group(2)), m.group(3)
    if not (1 <= d <= 31 and 1 <= mo <= 12):
        return m.group(0)
    year = int(y) if len(y) == 4 else 2000 + int(y)
    return f"{ONES[d]} {MONTHS[mo]} {year_words(year)}"


def _rupees(m):
    amount, frac, scale = m.group(1), m.group(2), m.group(3)
    n = int(amount.replace(",", ""))
    if scale:  # ₹38 लाख, ₹2.5 करोड़
        scale = {"करोड": "करोड़", "हजार": "हज़ार"}.get(scale, scale)
        num = decimal_words(amount, frac) if frac else int_words(n)
        return f"{num} {scale} रुपये"
    out = f"{int_words(n)} {'रुपया' if n == 1 else 'रुपये'}"
    if frac:
        paise = int((frac + "0")[:2])
        if paise:
            out += f" {int_words(paise)} पैसे"
    return out


def _percent(m):
    i, f = m.group(1), m.group(2)
    return (decimal_words(i, f) if f else int_words(int(i.replace(",", "")))) + " प्रतिशत"


def _time(m):
    h, mi = int(m.group(1)), int(m.group(2))
    if h > 23 or mi > 59:
        return m.group(0)
    return f"{int_words(h)} बजकर {int_words(mi)} मिनट" if mi else f"{int_words(h)} बजे"


def _ordinal(m):
    n, suffix = int(m.group(1)), m.group(2)
    vowel = suffix[-1] if suffix[-1] in "ाीे" else "ा"
    if n in ORDINAL_STEM:
        return ORDINAL_STEM[n] + vowel
    stem = {"ा": "वाँ", "ी": "वीं", "े": "वें"}[vowel]
    return int_words(n) + stem


def _abbrev(m):
    return " ".join(LETTERS[c] for c in m.group(0).replace(".", "") if c in LETTERS)


RULES = [
    # word-number hyphen: चंद्रयान-3 -> चंद्रयान 3 (Devanagari compounds like बार-बार are untouched)
    (re.compile(r"(?<=[^\W\d_])-(?=\d)"), " "),
    # dates dd/mm/yyyy (also - or . separators)
    (re.compile(NUM_L + r"(\d{1,2})[/.-](\d{1,2})[/.-](\d{4}|\d{2})" + NUM_R), _date),
    # rupee amounts, optional paise and scale word
    (re.compile(r"(?:₹|\bRs\.?|\bINR)\s?(\d[\d,]*)(?:\.(\d+))?(?:\s*(" + SCALE_WORDS + r")(?![ऀ-ॿ]))?"), _rupees),
    # percentages
    (re.compile(NUM_L + r"(\d[\d,]*)(?:\.(\d+))?\s?%"), _percent),
    # clock times hh:mm
    (re.compile(NUM_L + r"(\d{1,2}):(\d{2})" + NUM_R), _time),
    # ordinals 5वीं, 21वाँ, 3रा
    (re.compile(NUM_L + r"(\d+)\s?(वाँ|वां|वीं|वें|वा|वी|वे|ला|ली|ले|रा|री|रे|था|थी|थे|ठा|ठी|ठे)(?![ऀ-ॿ])"), _ordinal),
    # numeric ranges 10-12 -> दस से बारह
    (re.compile(NUM_L + r"(\d{1,4})\s?[-–]\s?(\d{1,4})" + NUM_R),
     lambda m: f"{plain_number_words(m.group(1))} से {plain_number_words(m.group(2))}"),
    # decimals
    (re.compile(NUM_L + r"(\d[\d,]*)\.(\d+)" + NUM_R), lambda m: decimal_words(m.group(1), m.group(2))),
    # grouped and plain integers
    (re.compile(NUM_L + r"(\d{1,3}(?:,\d{2,3})+|\d+)" + NUM_R), lambda m: plain_number_words(m.group(1))),
    # all-caps abbreviations (2+ letters, optional dots): RBI, A.T.M.
    (re.compile(r"(?<![A-Za-z])(?:[A-Z]\.){2,}|(?<![A-Za-z])[A-Z]{2,}(?![a-z])"), _abbrev),
]


def normalize(text: str) -> str:
    s = unicodedata.normalize("NFC", text)
    s = "".join(NUKTA_FORMS.get(c, c) for c in s)      # one spelling for nukta letters
    s = s.translate(DEV_DIGITS)                         # ०-९ -> 0-9
    for pattern, repl in RULES:
        s = pattern.sub(repl, s)
    s = re.sub(r"[ \t]+", " ", s).strip()
    return s


# ----------------------------------------------------------------------------------------------
# CLI: write t2_normalized.csv
# ----------------------------------------------------------------------------------------------
def kit_dir():
    import os
    env = os.environ.get("TTS_KIT")
    if env:
        return Path(env)
    for p in Path(__file__).resolve().parents:
        if (p / "metrics.py").exists() and (p / "data" / "val_items.csv").exists():
            return p
    sys.exit("Cannot find the kit folder; set TTS_KIT=/path/to/tts_bootcamp_kit")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="predictions/t2_normalized.csv")
    ap.add_argument("--kit", default=None, help="kit folder (default: $TTS_KIT or a parent folder)")
    a = ap.parse_args()
    kit = Path(a.kit) if a.kit else kit_dir()
    rows = []
    for split, name in (("val", "val_items.csv"), ("eval", "eval_items.csv")):
        with open(kit / "data" / name, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append({"id": r["id"], "split": split, "text": r["text"], "normalized_text": normalize(r["text"])})
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "split", "text", "normalized_text"])
        w.writeheader()
        w.writerows(rows)
    left = [r["id"] for r in rows if re.search(r"[0-9₹%]", r["normalized_text"])]
    print(f"wrote {len(rows)} rows to {out}; items with digits/₹/% left: {left or 'none'}")


if __name__ == "__main__":
    main()
