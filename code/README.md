# 24B4521 – TTS bootcamp (Hindi–Hinglish voice cloning with IndicF5)

Everything here runs from a copy of the kit (`tts_bootcamp_kit/`). This repository is meant to sit
**inside the kit folder** as `submission/` (or set `TTS_KIT=/path/to/kit`), so the scripts can
import `baseline/tts.py` and `metrics.py` and read `data/`. No kit data, model weights or caches
are included.

```
submission/
  code/            scripts (this README, normalize.py, translit.py, synth.py, fast.py, t4_noise.py, ...)
                   and code/timing/ (timing files of every synthesis run, incl. the Task 4 pair)
  predictions/     t1_val_errors.csv, t2_normalized.csv, t2_norm/ (40 wav), val_control/ (24 wav),
                   val_idea/ (24 wav), t3_idea/ (40 wav), t4_fast/ (40 wav)
  results.json     validation numbers from metrics.py
  idea.md          idea card
  report.md        answers to Q1–Q3
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
..\.venv\Scripts\python.exe code\synth.py --items ..\data\eval_items.csv --text-csv predictions\t2_normalized.csv --text-column normalized_text --out predictions\t2_norm --noise t4 --seed 0 --peak 0.95
# validation audio with the Task 2 text (= val_control for Task 3), and its scores
..\.venv\Scripts\python.exe code\synth.py --items ..\data\val_items.csv --text-csv predictions\t2_normalized.csv --text-column normalized_text --out predictions\val_control --noise t4 --seed 0 --peak 0.95
cd ..
.venv\Scripts\python.exe metrics.py submission/predictions/val_control --json work/val_control.json --csv work/val_control.csv
```

**Post-processing: one clipping guard.** IndicF5 copies the prompt's loudness, and the wrapper
rescales only quiet prompts (RMS < `target_rms`). Many outputs therefore reach full scale and are
clipped by `save_wav`. This happens in the kit's own `data/baseline_val` too: val_hd_02 has 0.06 %
of samples at |x| ≥ 0.999, just under the 0.1 % failure line. Without a guard, 3 of the 40 eval
outputs (ev_hi_01, ev_hi_05, ev_hd_05) fail the clipping check and would score as failed items.
`synth.py --peak 0.95` therefore scales an utterance down so its peak is 0.95 *only if it would
exceed that*. This is loudness normalization, which the rules allow; all other outputs are
bit-identical to the plain baseline. All submitted audio uses it.

Result (seed 0, `predictions/val_control`): overall CER 0.2421 → **0.1209**, hard 0.4583 →
**0.0946**, SIM 0.8303 → 0.8354. hindi, hinglish and long keep exactly the baseline's CER
(0.0342 / 0.302 / 0.0318). Their text is unchanged and the noise is the T4's, so the audio is the
baseline's; the guard only scales down the few that would clip. All 40 `t2_norm` files pass the
kit's sanity checks (`code/sanity_eval.py`), against 37/40 without the guard.

Robustness check with recognizers outside the kit (`code/eval_extra.py`, Meta MMS-1B-all with its
Hindi adapter). Latin letters are suppressed and digits are converted to words with `normalize()`,
because MMS writes numbers as digits. Hard CER goes 0.447 → 0.189 under MMS. Its remaining hard
errors are its own digit-grouping mistakes (it hears "दो हज़ार छब्बीस" as "2,26"). On hinglish, MMS
agrees with the kit's Whisper (0.296 vs 0.302).

```powershell
.venv\Scripts\python.exe submission\code\eval_extra.py data/baseline_val submission/predictions/val_control --json work/extra.json --csv work/extra.csv
```

### Four recognizers (all tasks)

Two more open Hindi recognizers were added later to `eval_extra.py` (`--asr whisper,mms,whisper_hi,vakyansh`):
`vasista22/whisper-hindi-large-v2` (Whisper large-v2 fine-tuned on Hindi) and
`Harveenchadha/vakyansh-wav2vec2-hindi-him-4200` (wav2vec2 CTC trained on 4200 h of Hindi). Both
are unreliable on the 25–45 s long items. Vakyansh decodes some whole files to a few words, yet
transcribes the same audio correctly in 20 s pieces, and the Hindi Whisper drops the end of val_lg_04
for every system. So they are only used on the hindi, hinglish and hard items. Hard CER, seed 0:

| hard CER | kit Whisper | MMS-1B | Whisper-Hindi | Vakyansh | mean |
|---|---|---|---|---|---|
| baseline (`data/baseline_val`) | 0.458 | 0.447 | 0.410 | 0.428 | 0.436 |
| Task 2 (`val_control`) | 0.095 | 0.184 | 0.094 | 0.044 | **0.104** |

```powershell
# every folder scored in this README, four recognizers (about 20 min)
.venv\Scripts\python.exe submission\code\eval_extra.py data\baseline_val submission\predictions\val_control work\val_control_s1 work\val_control_s2 submission\predictions\val_idea work\val_idea_s1 work\val_idea_s2 work\val_fast work\abl_naive_s0 work\abl_naive_s1 work\abl_naive_s2 work\abl_oracle_s0 work\abl_oracle_s1 work\abl_oracle_s2 --asr whisper,mms,whisper_hi,vakyansh --json work\robust4.json --csv work\robust4.csv
```

Without the T4 emulation, the official command is equivalent apart from the noise:
`python baseline/run_baseline.py --items data/eval_items.csv --out predictions/t2_norm --text-csv predictions/t2_normalized.csv --text-column normalized_text`.

## Task 3 – idea (see idea.md)

`code/translit.py` respells Latin-script English words in Devanagari with hand-written rules for
graphemes, Indian-English vowels and Hindi orthography. It has no word list; the rules were
developed on the validation words only. `python code/test_translit.py` checks the rules at text
level on the 17 English words of the validation items, plus the loanword in the V3 prompt
transcript: 13 of 18 are exact after the scorer's canonicalization. The audio experiment
(`val_idea/`, `t3_idea/`) was run only after the idea card (parts 1–3) was submitted on the form.

```powershell
# everything below: Task 3 text, val_idea/ + t3_idea/ (seed 0, T4 noise, baseline decoding),
# metrics.py on val_idea and val_control, MMS/WavLM check, results.json, zip + check_format
powershell -ExecutionPolicy Bypass -File submission\code\run_task3.ps1
# the same comparison at seeds 1 and 2 (control and idea), then the 3-seed summary
powershell -ExecutionPolicy Bypass -File submission\code\run_task3_seeds.ps1
```

Results on the 24 validation items (kit `metrics.py`):

| | val_control (Task 2 text) | val_idea (Task 2 + respelling) |
|---|---|---|
| hinglish CER, seed 0 | 0.302 | **0.047** |
| hinglish CER, mean of seeds 0/1/2 | 0.306 | **0.043** |
| overall CER, seed 0 (mean of 3 seeds) | 0.121 (0.128) | 0.057 (0.062) |
| SIM, seed 0 (mean of 3 seeds) | 0.835 (0.839) | 0.840 (0.839) |
| hinglish CER with MMS-1B (`code/eval_extra.py`) | 0.296 | 0.060 |

Hinglish CER, mean of seeds 0/1/2, with four recognizers (see "Four recognizers" above):

| | kit Whisper | MMS-1B | Whisper-Hindi | Vakyansh | mean |
|---|---|---|---|---|---|
| val_control (Task 2 text) | 0.306 | 0.313 | 0.318 | 0.304 | 0.310 |
| val_idea (Task 3) | 0.043 | 0.048 | 0.014 | 0.048 | **0.038 (−88 %)** |

All six hinglish items improve; hard, hindi and long items are unchanged, since their text is unchanged.
Per item (seed 0, kit Whisper):

| item | baseline = control | idea | what the recognizer heard for the English words (idea) |
|---|---|---|---|
| val_hg_01 | 0.452 | 0.032 | अफिस / इंटरनेट / स्लो |
| val_hg_02 | 0.216 | 0.135 | ने / अड्रेस / मेसेज (smallest gain: the य glide of न्यू is lost) |
| val_hg_03 | 0.280 | 0.020 | फ्लाइट / कैंसल / होटल |
| val_hg_04 | 0.263 | 0.026 | आनलाइन / क्लास |
| val_hg_05 | 0.283 | 0.043 | वाकिंग / प्रोपर / स्लीब |
| val_hg_06 | 0.318 | 0.023 | वीकेंड / मोवी / शॉपिंग |

`code/analyze_words.py` aligns each transcript to the reference word by word. The character error on
the 17 English words falls from 84 % (75/89 characters) with Latin input to 12 % (11/89) with the
respelling (9 % at seeds 1 and 2).

### Analysis: script alone, rules, and the best possible respelling

```powershell
powershell -ExecutionPolicy Bypass -File submission\code\run_ablation_t3.ps1
```

`code/run_ablation_t3.ps1` (not part of the submitted system) synthesizes the 6 hinglish validation
items, seeds 0–2, with two more versions of the English words: a **naive** mapping that replaces
each Latin letter by one fixed Devanagari letter with no English spelling knowledge (office →
ओफ्फिके, flight → फ्लिग्ह्त), and an **oracle** that uses the reference's own spelling of each word (the
best any respelling can do with this model). Hinglish CER, mean of 3 seeds:

| text given to the model | Whisper | MMS-1B | Whisper-Hindi | Vakyansh | mean of 4 | English-word char. error (seed 0, Whisper) |
|---|---|---|---|---|---|---|
| Latin English words (Task 2 text) | 0.306 | 0.313 | 0.318 | 0.304 | 0.310 | 84 % |
| naive letter mapping | 0.170 | 0.177 | 0.138 | 0.145 | 0.158 | 44 % |
| respelling rules (Task 3) | **0.043** | **0.048** | **0.014** | **0.048** | **0.038** | 12 % |
| oracle respelling | 0.026 | 0.032 | 0.006 | 0.059 | 0.031 | 10 % |

Moving to Devanagari letters alone removes about half of the error (0.310 → 0.158, mean of four
recognizers); the Indian-English pronunciation rules remove most of the rest and reach about 97 %
of the oracle's gain (0.310 → 0.038 against 0.310 → 0.031). The
remaining errors are shared by the oracle: न्यू loses its glide, ऑ is heard as आ (ऑनलाइन → आनलाइन,
वॉकिंग → वाकिंग), and स्लीप is sometimes heard with a final ब. These come from how the model renders those sounds (or how
the recognizers spell them), not from the respelling.

## Task 4 – faster

```powershell
# times baseline/run_baseline.py and code/fast.py on the 24 validation items in the same session
# (model loaded, one warm-up excluded), scores the fast system, writes t4_fast/, fills results.json
powershell -ExecutionPolicy Bypass -File submission\code\run_task4.ps1 -Nfe 8 -Schedule 0-2-4-6-8-12-16-24-32
```

The sweep and its comparison (about 1.5 h on this laptop):

```powershell
cd submission
..\.venv\Scripts\python.exe code\sweep_t4.py --out ..\work\t4sweep --seeds 0,1,2 --configs u12,e12,e10,e7,u7,e6,e5,e7c6,e7c5,p0-2-4-6-8-12-16-24-32,p0-2-4-8-12-16-24-32
# score the 3 Task 3 folders (seeds 0-2) and every sweep folder with Whisper + MMS, then compare
..\.venv\Scripts\python.exe code\eval_extra.py predictions\val_idea ..\work\val_idea_s1 ..\work\val_idea_s2 (Get-ChildItem ..\work\t4sweep -Directory).FullName --csv ..\work\t4sweep_all.csv
..\.venv\Scripts\python.exe code\compare_t4.py ..\work\t4sweep_all.csv --t3 val_idea,val_idea_s1,val_idea_s2 --timing-dir ..\work\t4sweep --baseline-s 408.6
```

**Official run** (Task 3 text, same machine and session): `baseline/run_baseline.py` 408.6 s, fast
system 37.4 s, so the speed-up is **10.9×**. Validation CER 0.0612 against 0.0571 for the Task 3
system (ratio 1.07; the limit is 1.2). With MMS-1B the fast system is lower: 0.083 against 0.091.
SIM 0.836 (Task 3: 0.840). All 40 `t4_fast/` files pass the kit's audio sanity checks. Timing files:
`code/timing/baseline_val_timing.json` and `code/timing/val_fast_timing.json`.

Profiling (`code/profile_baseline.py`): 99 % of synthesis time is `CFM.sample()`, i.e. 32 Euler
steps × 2 transformer passes for classifier-free guidance. Prompt preprocessing, the vocoder and
resampling together take under 1 %. `code/fast.py` re-implements `CFM.sample()`/`infer_process()`
with the same weights, duration rule, chunking, cross-fade and initial noise. It changes only how
the work is executed and the ODE time grid:

* the conditional and unconditional guidance passes run as one batch of 2
* the text embedding (4 ConvNeXt blocks) is computed once per item instead of at every step,
  since it does not depend on the noisy audio or the time
* the reference clip is preprocessed once per prompt
* bf16 autocast for the transformer. fp16 overflows to inf/NaN in this model, which is probably
  why IndicF5's loader keeps fp32; bf16 has fp32's exponent range
* the T4 noise (`t4_noise.py`) is drawn with torch ops on the GPU (`randn_torch`, ~1 ms) instead
  of numpy (~50 ms per item); it matches the numpy emulation to 5e-7
* **8 flow steps on a non-uniform time grid** (an allowed decoding setting, see below)

With fp32, 32 uniform steps and CPU noise, `fast.py --check` reproduces the baseline engine's
output (correlation 1.00000). After all of the above, 99 % of the remaining time is the
transformer itself (`work/profile_fast.py`): the GPU is compute-bound, so fewer transformer passes
are the only large lever. (cuDNN attention was tried and was slower than PyTorch's default
memory-efficient kernel on this Windows build; flash attention is not available in it.)

**Choosing the time grid.** F5-TTS samples with sway sampling, t' = t − (cos(πt/2) − 1 + t), which
packs steps near t = 0. With uniform t, cutting steps loses words quickly. EPSS (Empirically
Pruned Step Sampling; Zheng et al., 2025, and upstream F5-TTS `get_epss_timesteps`) keeps the fine
early steps of the 32-step grid and drops redundant late ones. `code/sweep_t4.py` ran each setting
on the 24 validation items with the Task 3 text at seeds 0, 1 and 2 (bf16, T4 noise). Each output
was scored with the kit's Whisper and with MMS-1B (`code/eval_extra.py`) and compared with the
Task 3 system (fp32, 32 steps) item by item at the same seeds (`code/compare_t4.py`). The last
column is the share of 20,000 resampled 40-item sets (pairs of item and seed) whose CER ratio
exceeds the 1.2 limit, an estimate of the risk on the 40 evaluation items.

| setting (time points in 1/32) | steps | CER Whisper | CER MMS | ratio to Task 3 | SIM | time | speed-up | P(ratio > 1.2) |
|---|---|---|---|---|---|---|---|---|
| Task 3 system (fp32, uniform) | 32 | 0.0619 | 0.0965 | 1.00 | 0.839 | 408.6 s* | 1.0× | – |
| uniform | 12 | 0.0606 | 0.0972 | 1.00 | 0.840 | 57.0 s | 7.2× | 1 % |
| EPSS | 12 | 0.0610 | 0.1006 | 1.02 | 0.836 | 56.8 s | 7.2× | 0.3 % |
| EPSS | 10 | 0.0606 | 0.1004 | 1.02 | 0.838 | 47.6 s | 8.6× | 1 % |
| **0-2-4-6-8-12-16-24-32 (chosen)** | **8** | **0.0603** | 0.1011 | **1.02** | 0.830 | 38.4 s | **10.6×** | 1.4 % |
| EPSS 0-2-4-6-8-16-24-32 | 7 | 0.0683 | 0.0987 | 1.05 | 0.833 | 33.8 s | 12.1× | 5 % |
| 0-2-4-8-12-16-24-32 | 7 | 0.0708 | 0.0911 | 1.02 | 0.831 | 33.9 s | 12.1× | 3 % |
| uniform | 7 | 0.0999 | 0.1310 | 1.46 | 0.822 | 33.9 s | 12.1× | 97 % |
| EPSS | 6 | 0.0668 | 0.1027 | 1.07 | 0.816 | 29.3 s | 13.9× | 7 % |
| EPSS | 5 | 0.0804 | 0.1084 | 1.19 | 0.818 | 24.7 s | 16.5× | 47 % |
| EPSS 7, guidance on first 6 steps | 7 | 0.0669 | 0.1001 | 1.05 | 0.817 | 31.7 s | 12.9× | 5 % |
| EPSS 7, guidance on first 5 steps | 7 | 0.0757 | 0.0964 | 1.09 | 0.817 | 29.5 s | 13.9× | 13 % |

CER, SIM (ECAPA) and time are means over the 3 seeds; ratio = (Whisper + MMS) CER over the Task 3
system's at the same seeds; speed-up against the official 408.6 s baseline (*the Task 3 row's time
is the kit baseline's). Sweep times used CPU noise; the official run draws it on the GPU (37.4 s).

At 7 steps the grid decides everything: uniform steps raise CER by 46 %, EPSS by 5 %. EPSS-7
leaves one large step between 8/32 and 16/32 (0.076 → 0.293 after sway), and Whisper's CER rises
10 % there. Adding the point 12/32 brings Whisper back to the full model's level for one more step,
so the chosen grid is EPSS-7 plus 12/32. The 7-step grids take 12 % less time, but one of the two
recognizers is 10–14 % worse with them, which is too close to the 1.2 limit for 40 items. The two
extra recognizers (hindi, hinglish and hard items, 3 seeds) agree: the chosen grid's CER is 0.94× the
Task 3 system's (Whisper 0.96, MMS 1.05, Whisper-Hindi 0.67, Vakyansh 0.83), EPSS-7's is 1.04×
(Whisper 1.10, Vakyansh 1.16). For the official run (seed 0) it is 0.83×. Skipping
guidance in the last steps saves little and costs speaker similarity.

An earlier step-count sweep with the Task 2 text and uniform steps (seed 0, `code/sweep_fast.ps1`)
gave CER 0.1209 (fp32, 32 steps), 0.1242 (16), 0.1200 (12) and 0.1434 (8 uniform steps).

The same seed with different noise (native RTX 4050 noise vs the T4's) moves category CER a
lot: hindi 0.034 → 0.045, long 0.032 → 0.051, overall 0.2421 → 0.2429. Small differences on 24
items are mostly noise, so comparisons here keep the noise fixed and use several seeds.

## Files written outside `submission/`

`work/` (in the kit folder) holds scores, logs and scratch audio. None of it is needed to check the
submission.
