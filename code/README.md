# 24B4521 – TTS bootcamp (Hindi–Hinglish voice cloning with IndicF5)

Everything here runs from a copy of the kit (`tts_bootcamp_kit/`). This repository is meant to sit
**inside the kit folder** as `submission/` (or set `TTS_KIT=/path/to/kit`), so the scripts can
import `baseline/tts.py` and `metrics.py` and read `data/`. No kit data, model weights or caches
are included.

```
submission/
  code/            scripts (this README, normalize.py, translit.py, synth.py, t4_noise.py, ...)
  predictions/     t1_val_errors.csv, t2_normalized.csv, t2_norm/ (40 wav), val_control/ (24 wav)
  results.json     validation numbers from metrics.py
  idea.md          idea card
```

## AI assistant disclosure

**Claude Code** (Anthropic's AI coding assistant) was used throughout. It wrote the code in
`code/`, ran the commands in this README, and helped analyse the results (vocabulary inspection,
reproducing the T4 baseline, the Task 2/3 rules). `idea.md` and `report.md` are written by the
participant in their own words.

## Machine and environment (exact commands used)

Windows 11, NVIDIA RTX 4050 Laptop GPU (6 GB), Python 3.12.1, PyTorch 2.5.1+cu121. Run from the kit folder:

```powershell
uv venv .venv --python 3.12
uv pip install --python .venv\Scripts\python.exe torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
# IndicF5's setup.py pins numpy<=1.26.4, which conflicts with the kit's numpy>=2.1, and the
# commit pinned in baseline/requirements.txt exists only on Hugging Face, not on GitHub.
# So: install the package from GitHub main without deps, then its dependencies.
uv pip install --python .venv\Scripts\python.exe --no-deps "git+https://github.com/ai4bharat/IndicF5.git@13f7c4d627cc10111aea8fe9c0039462cacacdc7"
uv pip install --python .venv\Scripts\python.exe accelerate cached_path click datasets "ema_pytorch>=0.5.2" "hydra-core>=1.3.0" jieba "librosa==0.11.0" matplotlib "numpy>=2.1,<2.3" pydub pypinyin safetensors soundfile tomli torchdiffeq "tqdm>=4.65.0" "transformers==4.49.0" transformers_stream_generator vocos wandb "x_transformers>=1.31.14" pyyaml pandas "speechbrain>=1.0" jiwer
hf auth login            # token of an account that accepted the IndicF5 terms
```

Windows only: SpeechBrain tries to symlink the ECAPA files into `~/.cache/bootcamp_ecapa`, which
needs admin rights. Copying them there once avoids it (`metrics.py` is unchanged):

```powershell
$src = (Get-ChildItem "$env:USERPROFILE\.cache\huggingface\hub\models--speechbrain--spkrec-ecapa-voxceleb\snapshots")[0].FullName
New-Item -ItemType Directory -Force "$env:USERPROFILE\.cache\bootcamp_ecapa"
Copy-Item "$src\*" "$env:USERPROFILE\.cache\bootcamp_ecapa\"
Copy-Item "$src\label_encoder.txt" "$env:USERPROFILE\.cache\bootcamp_ecapa\label_encoder.ckpt"
```

## Reproducing the T4 baseline on another GPU (`code/t4_noise.py`)

The only random step in IndicF5 inference is the initial noise `torch.randn(frames, 100)` in
`CFM.sample()`. PyTorch's CUDA sampler (Philox4x32-10 + Box-Muller in a grid-stride kernel)
assigns random sub-streams by launch grid, and the grid is sized from the GPU's SM count, so the
same seed gives *different* noise on a T4 (40 SMs) and on an RTX 4050 (20 SMs). Synthesizing with
`baseline/run_baseline.py` on this laptop therefore gives the same durations as
`data/baseline_val/` but different audio (waveform correlation ≈ 0).

`t4_noise.py` re-implements that sampler in numpy. `python code/t4_noise.py` checks it against
the local GPU (max abs difference 2e-6, i.e. float rounding). `code/synth.py --noise t4` then draws
the noise a T4 would draw, and turns off TF32 (a T4 has none). With this, the kit's baseline audio
is reproduced on the laptop: correlation 1.00000 and relative error ≤ 0.1 % on val_hi_02,
val_hg_01, val_hd_01 and val_lg_02 (the last is two chunks, so the generator offset is tracked
correctly). Model, seed and settings are untouched; only the hardware dependence is removed. All
audio in `predictions/` was made this way with seed 0, so it is directly comparable to the kit's
baseline outputs.

## Task 1 – measure the baseline

```powershell
.venv\Scripts\python.exe metrics.py data/baseline_val --json work/baseline_val_scores.json --csv work/baseline_val_scores.csv
```

Output: CER 0.2421, SIM 0.8303. This matches `data/baseline_published.json` exactly.

`predictions/t1_val_errors.csv` tags all 14 items with CER > 0.20, using the transcripts and the
analysis below:

* hinglish, all 6 items → `english_word_error`. Every Latin-script word becomes a Hindi-sounding
  non-word or is dropped (office→"अजसे", internet→"सर्ण", slow→"कोई", hotel→"ओयो",
  shopping→"चहें"; "online class"→"अर्ली").
* hard, all 8 items → `number_error`. Digits, dates and amounts come out as non-words
  (12/03/2026→"पेड़म शोर एन्यो…"). Three items also contain an abbreviation (RBI, ATM, GST), but
  the number spans are 2–3× longer in the reference, so number_error is the main error.

Two mechanisms, found by reading the model code (`f5_tts/infer/utils_infer.py`,
`f5_tts/model/utils.py`) and its vocabulary (`checkpoints/vocab.txt`, 2545 tokens):

1. `₹` is not in the vocabulary. Unknown characters map to index 0, the space, so the model never
   sees the rupee sign and never says "रुपये". Latin letters and ASCII digits *are* in the vocabulary,
   but as tokens left over from F5-TTS's English/Chinese pretraining (the vocabulary still holds
   1325 pinyin syllables). IndicF5's Indic fine-tuning data writes English loanwords in
   Devanagari, as the kit's own IndicVoices-R prompt transcripts do (ऑडर, मार्केट).
2. The output duration is `ref_frames × chars(gen_text) / chars(ref_text)`. A digit is one
   character but several syllables when spoken, so hard items get far too little time. The
   baseline reads them at 14–17 reference characters/s against about 9 for Hindi items
   (val_hd_03: 16.4, val_hd_07: 16.8).

## Task 2 – text frontend

`code/normalize.py` is a rule-based `normalize(text)` with no model. It handles Indian-grouped
numbers (lakh/crore), ₹ amounts (with paise and scale words), dd/mm/yyyy dates, percentages, years,
digit-by-digit identifiers, decimals, ordinals, times and ranges, and spells out all-caps
abbreviations letter by letter. Latin-script English words are left unchanged. Its conventions
come from the validation items (see the module docstring); number words, month names and letter
names are general Hindi. On all 8 hard validation items the normalized text equals the `spoken`
reference (text-level CER 0.000).

```powershell
cd submission
$env:TTS_KIT = "C:\path\to\kit"
..\.venv\Scripts\python.exe code\normalize.py --out predictions\t2_normalized.csv
# eval audio: normalized text, everything else as in the baseline (seed 0, T4 noise)
..\.venv\Scripts\python.exe code\synth.py --items ..\data\eval_items.csv --text-csv predictions\t2_normalized.csv --text-column normalized_text --out predictions\t2_norm --noise t4 --seed 0
# validation audio with the Task 2 text (= val_control for Task 3), and its scores
..\.venv\Scripts\python.exe code\synth.py --items ..\data\val_items.csv --text-csv predictions\t2_normalized.csv --text-column normalized_text --out predictions\val_control --noise t4 --seed 0
cd ..
.venv\Scripts\python.exe metrics.py submission/predictions/val_control --json work/val_control.json --csv work/val_control.csv
```

Result (seed 0): overall CER 0.2421 → **0.1209**, hard 0.4583 → **0.0946**, SIM 0.8303 → 0.8355.
hindi, hinglish and long are bit-for-bit unchanged, because their text is unchanged and the noise
is the T4's.

Robustness check with recognizers outside the kit (`code/eval_extra.py`, Meta MMS-1B-all with its
Hindi adapter). Latin letters are suppressed and digits are converted to words with `normalize()`,
because MMS writes numbers as digits. Hard CER goes 0.447 → 0.189 under MMS. Its remaining hard
errors are its own digit-grouping mistakes (it hears "दो हज़ार छब्बीस" as "2,26"). On hinglish, MMS
agrees with the kit's Whisper (0.296 vs 0.302).

```powershell
.venv\Scripts\python.exe submission\code\eval_extra.py data/baseline_val submission/predictions/val_control --json work/extra.json --csv work/extra.csv
```

Without the T4 emulation, the official command is equivalent apart from the noise:
`python baseline/run_baseline.py --items data/eval_items.csv --out predictions/t2_norm --text-csv predictions/t2_normalized.csv --text-column normalized_text`.

## Task 3 – idea (planned; see idea.md)

`code/translit.py` respells Latin-script English words in Devanagari with hand-written rules for
graphemes, Indian-English vowels and Hindi orthography. It has no word list; the rules were
developed on the validation words only. `python code/test_translit.py` checks the rules at text
level on the 17 English words of the validation items, plus the loanword in the V3 prompt
transcript: 13 of 18 are exact after the scorer's canonicalization. The audio experiment
(`val_idea/`, `t3_idea/`) runs only after the idea card was submitted.

## Task 4 – faster (groundwork; final timing after Task 3)

Profiling (`work/profile_baseline.py`): 99 % of synthesis time is `CFM.sample()`, i.e. 32 Euler
steps × 2 transformer passes for classifier-free guidance. Prompt preprocessing, the vocoder and
resampling together take under 1 %. `code/fast.py` re-implements `CFM.sample()`/`infer_process()`
with the same weights, duration rule, chunking, cross-fade and initial noise. It changes only how
the work is executed:

* the conditional and unconditional guidance passes run as one batch of 2
* the text embedding (4 ConvNeXt blocks) is computed once per item instead of at every step,
  since it does not depend on the noisy audio or the time
* the reference clip is preprocessed once per prompt
* bf16 autocast for the transformer. fp16 overflows to inf/NaN in this model, which is probably
  why IndicF5's loader keeps fp32; bf16 has fp32's exponent range
* fewer flow-matching steps (`nfe_step`, an allowed decoding setting)

Correctness: in fp32 with 32 steps, `fast.py --check` reproduces the baseline engine's output
(correlation 1.00000).

Step-count sweep on the 24 validation items with the Task 2 text (seed 0, T4 noise, scored with
metrics.py; `work/sweep_fast.ps1`):

| engine | steps | CER | SIM | synthesis time |
|---|---|---|---|---|
| baseline wrapper, fp32 (val_control) | 32 | 0.1209 | 0.8355 | 444.8 s |
| fast.py bf16 | 32 | 0.1236 | 0.8349 | (GPU shared with another job) |
| fast.py bf16 | 16 | 0.1242 | 0.8338 | 75.9 s |
| fast.py bf16 | 12 | 0.1200 | 0.8352 | 57.7 s |
| fast.py bf16 | 8 | 0.1434 | 0.8215 | 38.7 s |

12 steps keeps CER and SIM within noise of the full model at about 7.7× speed. 8 steps starts to
lose words (+19 % CER) and speaker similarity. The official speed-up is measured on the Task 3
system against `baseline/run_baseline.py` in the same session.

The same seed with different noise (native RTX 4050 noise vs the T4's) moves category CER a
lot: hindi 0.034 → 0.045, long 0.032 → 0.051, overall 0.2421 → 0.2429. Small differences on 24
items are mostly noise, so comparisons here keep the noise fixed.

## Files written outside `submission/`

`work/` (in the kit folder) holds scores, logs and scratch audio. None of it is needed to check the
submission.
