# Idea card

**(1) Failure:** english_word_error (hinglish). Latin-script English words become Hindi-sounding non-words: val_hg_01 (office/internet/slow → "अजसे/सर्ण/कोई", CER 0.45), val_hg_06 (weekend/movie/shopping → "वियन/अहीं/चहें", CER 0.32).

**(2) Hypothesis & method:** IndicF5's Latin letters are leftovers from F5-TTS pretraining; its Indic training transcripts write loanwords in Devanagari (kit prompts: मार्केट, ऑडर), so Latin input is out-of-distribution. After Task 2 normalization, I respell each Latin word in Devanagari with hand-written rules only: digraphs/silent letters (sh, ph, igh, final e), Indian-English vowels (a_e→ए, short a→ऐ, short o→ऑ, ar→आर, er→अर, ing→इंग), Hindi orthography (virama clusters, nasal→anusvara). No word list; decoding unchanged.

**(3) Prediction:** hinglish CER 0.30 → ~0.10 on validation (~65% relative), about half on evaluation (unseen words). Other categories unchanged. Risks: irregular vowels/stress (movie→मोवी), recognizer spelling variants, shifted durations.

**(4) Result:** _(added after the experiment)_
