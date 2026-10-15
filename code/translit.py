#!/usr/bin/env python3
"""
Task 3 - rule-based Indian-English respeller: Latin-script English words -> Devanagari.

IndicF5 keeps Latin letters in its vocabulary (inherited from F5-TTS's English pretraining) but was
fine-tuned on Indic-script transcripts, where English loanwords are written in Devanagari (the kit's
own IndicVoices-R prompt transcripts write ऑडर, मार्केट, टाइम, वेट, पिज़्ज़ा, पैकेट). Latin input is
therefore out of distribution and comes out garbled. This module rewrites each English word the way
an Indian speaker says it, in Devanagari, using only hand-written rules:

  1. grapheme rules: digraphs (sh ch th ph ck ng tch dge), silent letters (final e, gh, kn, wr,
     mb), soft c/g, -tion/-sion/-ture, double consonants
  2. Indian-English vowel rules: magic e (a_e -> ए, i_e -> आइ, o_e -> ओ, u_e -> यू), r-coloured
     vowels (ar -> आर, er/ir/ur -> अर), broad a before ss/st/sk/ft (आ), short a -> ऐ,
     short o -> ऑ, short u -> अ, ee/ea -> ई, oo -> ऊ, igh -> आइ, final y -> ई, -ing -> इंग,
     unstressed final -el/-en/-em/-al -> अ
  3. Hindi orthography: consonant clusters joined with virama, a nasal before a consonant of the
     same place written as anusvara (ं), word-final consonants left bare (Hindi schwa deletion)

No pronunciation dictionary or model is used. No word-specific entries: the rules were developed on
the 17 English words of the validation items only.

    from translit import transliterate
    transliterate("आज office में internet बहुत slow चल रहा है।")
"""
import re

VOWEL_LETTERS = set("aeiou")

# vowel code -> (independent form, matra)
VOWEL_FORMS = {
    "a": ("अ", ""), "aa": ("आ", "ा"), "ae": ("ऐ", "ै"), "e": ("ए", "े"), "i": ("इ", "ि"),
    "ii": ("ई", "ी"), "u": ("उ", "ु"), "uu": ("ऊ", "ू"), "o": ("ओ", "ो"), "ao": ("ऑ", "ॉ"),
}
NASALS = {"न", "म"}
# consonants before which a nasal is written as anusvara (same place of articulation, or s/sh)
ANUSVARA_BEFORE = {
    "न": set("कखगघचछजझटठडढतथदधसश") | {"ज़", "फ़"},
    "म": set("पफबभ") | {"फ़"},
}


def _is_v(ch):
    return ch in VOWEL_LETTERS


class _Word:
    """Scanner over one lowercase English word that emits ('C', letter) / ('V', code) tokens."""

    def __init__(self, w):
        self.w = w
        self.n = len(w)
        self.toks = []
        self.nsyl = max(1, len(re.findall(r"[aeiou]+|y(?![aeiou])", w[:-1] if w.endswith("e") and len(w) > 2 else w)))

    def at(self, j):
        return self.w[j] if 0 <= j < self.n else ""

    def C(self, *letters):
        for c in letters:
            self.toks.append(("C", c))

    def V(self, *codes):
        for c in codes:
            self.toks.append(("V", c))

    # --- context helpers ------------------------------------------------------------------
    def cons_run(self, j):
        """Number of consonant letters starting at j (y counts as consonant only before a vowel)."""
        k = j
        while k < self.n and not _is_v(self.w[k]) and not (self.w[k] == "y" and k > j):
            k += 1
        return k - j

    def magic_e(self, j):
        """Vowel at j followed by one consonant and a final silent e (optionally +s/+d)."""
        c = self.at(j + 1)
        if not c or _is_v(c) or c in "wxy" or self.cons_run(j + 1) != 1:
            return False
        tail = self.w[j + 2:]
        return tail in ("e", "es", "ed") and not (c == "r")

    def last_syllable(self, j):
        """True if no vowel letter follows position j except a silent final e."""
        rest = self.w[j + 1:]
        rest = rest[:-1] if rest.endswith("e") else rest
        return not re.search(r"[aeiouy]", rest)

    def open_syllable(self, j):
        """Vowel at j followed by exactly one consonant and then a vowel (x = /ks/ closes it)."""
        return self.cons_run(j + 1) == 1 and self.at(j + 1) != "x" and _is_v(self.at(j + 2))


def _vowel(s, i):
    """Handle the vowel (group) at i; return new index."""
    w, at = s.w, s.at
    c = w[i]
    nxt, nxt2, nxt3 = at(i + 1), at(i + 2), at(i + 3)
    first = i == 0
    final_syl = s.last_syllable(i)
    multi = s.nsyl > 1

    if c == "a":
        if nxt in "iy" and nxt:                                   # rain, day
            s.V("e"); return i + 2
        if nxt in "uw" and nxt:                                   # auto, law
            s.V("ao"); return i + 2
        if nxt == "r":
            if nxt2 == "e" and i + 3 == s.n:                      # care -> केयर
                s.V("e"); s.C("य"); s.V("a"); s.C("र"); return i + 3
            if _is_v(nxt2):                                       # carry, parent
                s.V("ae"); return i + 1
            if final_syl and multi:                               # dollar, regular
                s.V("a"); return i + 1
            s.V("aa"); return i + 1                               # car, card
        if nxt == "l" and nxt2 in ("k",):                         # walk, talk: l silent
            s.V("ao"); return i + 2
        if nxt == "l" and (nxt2 == "l" or (nxt2 in "t" and nxt2)):  # call, salt
            s.V("ao"); return i + 1
        if w[i + 1:] in ("tion", "tions"):                        # station, nation (not national)
            s.V("e"); return i + 1
        if re.fullmatch(r"nge[sd]?", w[i + 1:]):                  # change, range: magic e over soft -nge
            s.V("e"); return i + 1
        if s.magic_e(i) or (s.cons_run(i + 1) == 1 and w[i + 2:] == "le"):  # late, table
            s.V("e"); return i + 1
        if i == s.n - 1:                                          # data, pizza
            s.V("aa"); return i + 1
        if w[i + 1:i + 3] in ("ss", "st", "sk", "sp", "ft", "th") or w[i + 1:] in ("s",):  # class, ask
            s.V("aa"); return i + 1
        if final_syl and multi and nxt in "lnm" and (i + 2 == s.n):  # total, human
            s.V("a"); return i + 1
        if first and multi and s.open_syllable(i):                # about, again
            s.V("a"); return i + 1
        s.V("ae"); return i + 1                                   # cat, cancel

    if c == "e":
        if nxt == "e":
            if nxt2 == "r" and (i + 3 == s.n or not _is_v(nxt3)):  # career, engineer
                s.V("i"); s.C("य"); s.V("a"); s.C("र"); return i + 3
            s.V("ii"); return i + 2                               # sleep, weekend
        if nxt == "a":
            if nxt2 == "r" and (i + 3 == s.n or not _is_v(nxt3)):  # near, clear
                s.V("i"); s.C("य"); s.V("a"); s.C("र"); return i + 3
            s.V("ii"); return i + 2                               # seat, speak
        if nxt == "i":
            if w.startswith("gh", i + 2):                          # eight, weight
                s.V("e"); return i + 4
            s.V("ii"); return i + 2                               # receive
        if nxt == "y":                                            # key, money / they
            s.V("ii" if i + 2 == s.n and multi else "e"); return i + 2
        if nxt in "wu" and nxt:                                   # new, euro
            prev = at(i - 1)
            if prev in ("r", "l", "j") or w[max(0, i - 2):i] in ("ch", "sh"):
                s.V("uu")
            else:
                s.C("य"); s.V("uu")
            return i + 2
        if nxt == "r":
            if nxt2 == "e" and i + 3 == s.n:                      # here
                s.V("i"); s.C("य"); s.V("a"); s.C("र"); return i + 3
            if _is_v(nxt2) or nxt2 == "y":                        # very, period
                s.V("e"); return i + 1
            s.V("a"); return i + 1                                # proper, internet, service
        # silent final e (and -es/-ed endings handled by caller)
        if i == s.n - 1:
            if s.n <= 2 or (s.n == 3 and w[:2] in ("sh", "th")):  # be, me, he, she
                s.V("ii")
            elif at(i - 1) == "l" and i >= 2 and not _is_v(at(i - 2)):  # table, simple -> -अल
                # schwa goes before the l: undo the l, add schwa, re-add l
                if s.toks and s.toks[-1] == ("C", "ल"):
                    s.toks.pop(); s.V("a"); s.C("ल")
            return i + 1
        if s.magic_e(i) and multi is False:                       # theme
            s.V("ii"); return i + 1
        if first and s.open_syllable(i) and multi:                # even, equal
            s.V("ii"); return i + 1
        if final_syl and multi and nxt in "lnm" and i + 2 == s.n:  # hotel, open, system
            s.V("a"); return i + 1
        s.V("e"); return i + 1                                    # message, address, weekend

    if c == "i":
        if w.startswith("gh", i + 1):                             # flight, high
            s.V("aa", "i"); return i + 3
        if nxt == "e":
            if i + 2 == s.n:                                      # movie / die
                if multi:
                    s.V("ii")
                else:
                    s.V("aa", "i")
                return i + 2
            if nxt2 == "r" or nxt2 == "s" and i + 3 == s.n:       # cries
                s.V("aa", "i"); return i + 2
            s.V("ii"); return i + 2                               # field, piece
        if nxt == "r":
            if nxt2 == "e" and i + 3 == s.n:                      # fire
                s.V("aa"); s.C("य"); s.V("a"); s.C("र"); return i + 3
            if not _is_v(nxt2):                                   # first, girl
                s.V("a"); return i + 1
        if nxt in ("o", "a", "u") and nxt:                        # radio, india, medium
            s.V("i"); s.C("य"); return i + 1
        if s.magic_e(i) and multi and w[i + 1:i + 3] in ("ce", "ve"):  # office, active: unstressed
            s.V("i"); return i + 1
        if s.magic_e(i):                                          # line, time
            s.V("aa", "i"); return i + 1
        if w[i + 1:i + 3] in ("nd", "ld") and i + 3 >= s.n:       # find, child
            s.V("aa", "i"); return i + 1
        if i == s.n - 1:                                          # taxi
            s.V("ii"); return i + 1
        s.V("i"); return i + 1                                    # link, office, walking

    if c == "o":
        if nxt == "o":
            if nxt2 == "k":                                       # book, look
                s.V("u"); return i + 2
            s.V("uu"); return i + 2                               # food, room
        if nxt == "a":                                            # road
            s.V("o"); return i + 2
        if nxt == "i":                                            # point
            s.V("ao", "i"); return i + 2
        if nxt == "y":                                            # boy
            s.V("ao"); s.C("य"); return i + 2
        if nxt == "u":
            if w.startswith("gh", i + 2):                         # thought
                s.V("ao"); return i + 4
            if nxt2 == "r":                                       # four, course, your
                s.V("o"); s.C("र"); return i + 3
            if nxt2 == "s" and i + 3 == s.n:                      # famous
                s.V("a"); s.C("स"); return i + 3
            s.V("aa", "u"); return i + 2                          # out, about, house
        if nxt == "w":
            if w.startswith("er", i + 2):                         # power, tower
                s.V("aa"); s.C("व"); s.V("a"); s.C("र"); return i + 4
            if i + 2 == s.n:                                      # slow, show / how, now
                if at(i - 1) in ("l", "r") or s.n >= 4:
                    s.V("o")
                else:
                    s.V("aa", "u")
                return i + 2
            s.V("aa", "u"); return i + 2                          # down, town
        if nxt == "r":
            if final_syl and multi and i + 2 == s.n:              # doctor, motor
                s.V("a"); return i + 1
            s.V("o"); return i + 1                                # story, sport
        if s.magic_e(i):                                          # home, code
            s.V("o"); return i + 1
        if w[i + 1:i + 3] in ("ld",) or (w[i + 1:i + 3] == "st" and i + 3 == s.n):  # old, most
            s.V("o"); return i + 1
        if i == s.n - 1:                                          # go, photo
            s.V("o"); return i + 1
        if s.open_syllable(i):                                    # hotel, total, open
            s.V("o"); return i + 1
        s.V("ao"); return i + 1                                   # office, shopping, online

    if c == "u":
        prev = at(i - 1)
        soft_y = prev not in ("r", "l", "j", "t", "d", "s", "z") and not w[max(0, i - 2):i] in ("ch", "sh")
        if nxt == "e" and i + 2 == s.n:                           # blue, value
            if soft_y and prev and not _is_v(prev):
                s.C("य")
            s.V("uu"); return i + 2
        if nxt == "i":                                            # suit, fruit
            s.V("uu"); return i + 2
        if nxt == "r":
            if nxt2 == "e" and i + 3 == s.n:                      # pure, sure
                s.C("य"); s.V("o"); s.C("र"); return i + 3
            if not _is_v(nxt2):                                   # turn, nurse
                s.V("a"); return i + 1
        if nxt == "a":                                            # usual, actual
            s.V("u"); return i + 1
        if s.magic_e(i) or (s.open_syllable(i) and not first) or (first and s.open_syllable(i)):
            if soft_y and (first or not _is_v(prev)):              # use, music, computer
                s.C("य")
            s.V("uu"); return i + 1
        if i == s.n - 1:                                          # menu
            if soft_y:
                s.C("य")
            s.V("uu"); return i + 1
        if prev in ("p", "b", "f") and w[i + 1:i + 3] in ("ll", "sh"):  # pull, push
            s.V("u"); return i + 1
        s.V("a"); return i + 1                                    # cup, number

    raise ValueError(c)


def _consonant(s, i):
    """Handle the consonant (group) at i; return new index."""
    w, at = s.w, s.at
    c = w[i]
    nxt = at(i + 1)
    rest = w[i:]

    if rest.startswith("tch"):
        s.C("च"); return i + 3
    if rest.startswith("dge"):
        s.C("ज"); return i + 3
    if rest.startswith("sch"):
        s.C("स", "क"); return i + 3
    if rest.startswith("tion") or rest.startswith("ssion") or rest.startswith("cian"):
        k = 4 if rest.startswith("tion") or rest.startswith("cian") else 5
        s.C("श"); s.V("a"); s.C("न"); return i + k
    if rest.startswith("sion"):
        s.C("ज़" if i > 0 and _is_v(at(i - 1)) else "श"); s.V("a"); s.C("न"); return i + 4
    if rest.startswith("tial") or rest.startswith("cial"):
        s.C("श"); s.V("a"); s.C("ल"); return i + 4
    if rest.startswith("ture"):
        s.C("च"); s.V("a"); s.C("र"); return i + 4
    if rest.startswith("ph"):
        s.C("फ़"); return i + 2
    if rest.startswith("sh"):
        s.C("श"); return i + 2
    if rest.startswith("ch"):
        s.C("क" if at(i + 2) in ("r", "l") else "च"); return i + 2
    if rest.startswith("th"):
        s.C("थ"); return i + 2
    if rest.startswith("wh"):
        s.C("व"); return i + 2
    if rest.startswith("ck"):
        s.C("क"); return i + 2
    if re.fullmatch(r"nge[sd]?", rest):                          # change, challenge: soft g after n
        s.C("न", "ज"); return i + 2
    if rest.startswith("ng"):
        s.C("न", "ग"); return i + 2
    if i == 0 and rest.startswith("kn"):
        s.C("न"); return i + 2
    if i == 0 and rest.startswith("wr"):
        s.C("र"); return i + 2
    if rest.startswith("gn") and (i == 0 or i + 2 == s.n):
        s.C("न"); return i + 2
    if rest == "mb":
        s.C("म"); return i + 2
    if rest.startswith("gh"):
        if i == 0:
            s.C("ग")
        return i + 2                                              # silent after a vowel
    if rest.startswith("qu"):
        if rest == "que":
            s.C("क"); return i + 3
        s.C("क", "व"); return i + 2
    if c == "x":
        if i == 0:
            s.C("ज़")
        else:
            s.C("क", "स")
        return i + 1
    if c == "c":
        if nxt == "c":
            if at(i + 2) in ("e", "i", "y"):                      # accept
                s.C("क", "स")
            else:
                s.C("क")
            return i + 2
        s.C("स" if nxt in ("e", "i", "y") and nxt else "क"); return i + 1
    if c == "g":
        if nxt == "g":
            s.C("ग"); return i + 2
        soft = nxt in ("e", "i", "y") and nxt
        if soft and i == 0 and not (at(i + 2) == "n"):            # get, give (initial g stays hard)
            soft = False
        s.C("ज" if soft else "ग"); return i + 1
    if c == "s":
        if nxt == "s":
            s.C("स"); return i + 2
        if i + 1 == s.n and i >= 1 and (at(i - 1) in "aeoy" and at(i - 1)) and s.n > 3:   # days, goes
            s.C("ज़"); return i + 1
        if 0 < i and _is_v(at(i - 1)) and _is_v(nxt) and nxt != "u":  # music, easy
            s.C("ज़"); return i + 1
        s.C("स"); return i + 1
    if c == "y":
        s.C("य"); return i + 1
    if c == "h":
        if i > 0 and not _is_v(nxt):                              # oh, ah
            return i + 1
        s.C("ह"); return i + 1
    if c == "d" and rest == "ed" and False:
        pass
    simple = {"b": "ब", "d": "ड", "f": "फ़", "j": "ज", "k": "क", "l": "ल", "m": "म", "n": "न",
              "p": "प", "r": "र", "t": "ट", "v": "व", "w": "व", "z": "ज़"}
    if c in simple:
        k = i + 2 if nxt == c else i + 1                          # double consonants -> single
        s.C(simple[c]); return k
    return i + 1                                                  # unknown symbol: skip


def _scan(word):
    s = _Word(word)
    w = word
    i = 0
    while i < s.n:
        ch = w[i]
        # -ed / -es endings
        if w[i:] == "ed" and i > 0 and not _is_v(w[i - 1]) and s.n > 3:
            if w[i - 1] in "td":
                s.V("i"); s.C("ड")
            else:
                s.C("ड")
            break
        if w[i:] == "es" and i > 0 and s.n > 3 and not _is_v(w[i - 1]):
            if w[i - 1] in "sxz" or w[i - 2:i] in ("ch", "sh") or w[i - 1] in "cg":
                s.V("i"); s.C("ज़")
            else:
                s.C("ज़")
            break
        if ch == "y" and i > 0 and not _is_v(w[i + 1] if i + 1 < s.n else ""):
            # y as a vowel
            if i == s.n - 1:
                if s.nsyl > 1 or any(_is_v(x) for x in w[:i]):
                    s.V("ii")                                     # happy, city
                else:
                    s.V("aa", "i")                                # my, fly
            elif s.magic_e(i):
                s.V("aa", "i")                                    # type
            else:
                s.V("i")                                          # system
            i += 1
            continue
        if _is_v(ch):
            i = _vowel(s, i)
        else:
            i = _consonant(s, i)
    return s.toks


def _render(toks):
    out = []
    for k, (typ, val) in enumerate(toks):
        prev = toks[k - 1] if k > 0 else None
        nxt = toks[k + 1] if k + 1 < len(toks) else None
        if typ == "V":
            ind, matra = VOWEL_FORMS[val]
            out.append(matra if prev and prev[0] == "C" else ind)
            continue
        if (val in NASALS and nxt and nxt[0] == "C" and nxt[1] in ANUSVARA_BEFORE[val]
                and prev and prev[0] == "V"):
            out.append("ं")
            continue
        out.append(val)
        if nxt and nxt[0] == "C":
            # a nasal before a consonant of another place is left bare, as Hindi spells
            # loanwords (ऑनलाइन, इनपुट); schwa deletion removes its vowel when read
            # (except before the glide य: menu -> मेन्यू, as in अन्य)
            if not (val in NASALS and prev and prev[0] == "V" and nxt[1] != "य"):
                out.append("्")
    return "".join(out)


def respell_word(word: str) -> str:
    """One English word (Latin letters) -> Devanagari respelling."""
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return word
    if len(w) == 1:                       # a lone letter is read by its name (I -> आई, x -> एक्स)
        from normalize import LETTERS
        return LETTERS[w.upper()]
    return _render(_scan(w))


LATIN_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def transliterate(text: str) -> str:
    """Respell every Latin-script word in `text`; everything else is unchanged."""
    return LATIN_WORD.sub(lambda m: respell_word(m.group(0)), text)


if __name__ == "__main__":
    import sys
    for arg in sys.argv[1:]:
        print(arg, "->", transliterate(arg))
