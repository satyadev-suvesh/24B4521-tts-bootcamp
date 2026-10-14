# Idea card

**(1) Failure:** english\_word\_error (hinglish). Latin-script English words become non-sense hindi-sounding words: 84% character error on the English words of all six items, whereas the Hindi words around them are almost perfect. 

Examples - val\_hg\_01: office/internet/slow → "अजसे/सर्ण/कोई" (CER 0.45); val\_hg\_06: weekend/movie/shopping → "वियन/अहीं/चहें" (CER 0.32).



**(2) Hypothesis \& method:** IndicF5's Latin letters are leftovers from F5-TTS pretraining; its Indic training transcripts write loanwords in Devanagari (kit prompts show it: मार्केट, ऑडर), so Latin input is out-of-distribution. After Task 2 normalization, we re-spell each Latin word in Devanagari with hand-written rules only - digraphs/silent letters (sh, ph, igh, final e), Indian-English vowels (a\_e→ए, short a→ऐ, short o→ऑ, etc.), Hindi orthography (virama clusters, nasal→anusvara). No word list, and decoding unchanged.



**(3) Prediction:** hinglish CER drops from 0.30 to \~0.10 on validation (\~65% relative), and about half on evaluation (unseen words). Other categories unchanged. Risks - irregular vowels/stress (movie→मोवी), recognizer spelling variants, and shifted durations.



**(4) Result:** *(added after the experiment)*

