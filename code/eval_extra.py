#!/usr/bin/env python3
"""
Robustness check: score validation folders with recognizers / speaker models besides the kit's.

The final scorer averages several recognizers that are not in the kit, so a change that only helps
Whisper-turbo is suspect. This script scores each folder with
  * CER (--asr, default "whisper,mms"):
      whisper     the kit's Whisper-large-v3-turbo (same suppression as metrics.py)
      mms         Meta MMS-1B-all with its Hindi adapter (CTC, a different architecture and data)
      whisper_hi  vasista22/whisper-hindi-large-v2 (Whisper large-v2 fine-tuned on Hindi speech)
      vakyansh    Harveenchadha/vakyansh-wav2vec2-hindi-him-4200 (wav2vec2 CTC, 4200 h of Hindi)
    all compared with metrics.cer (the kit's canonicalization). whisper_hi and vakyansh are
    unreliable on the long items (25-45 s): vakyansh decodes some whole files to a few words that
    it transcribes correctly in 20 s pieces, and whisper_hi drops the end of val_lg_04 for every
    system. Use them on the hindi / hinglish / hard items.
  * SIM: kit ECAPA-TDNN and WavLM-base-plus-sv (cosine to the prompt clip)
and prints per-category means for each folder side by side. Failed items (metrics.sanity) count as
CER 1 / SIM 0, as in metrics.py. Recognizers are loaded one at a time (they do not all fit in 6 GB).

    python code/eval_extra.py data/baseline_val predictions/val_control predictions/val_idea --json work/extra.json
    python code/eval_extra.py predictions/val_idea --asr whisper,mms,whisper_hi,vakyansh --csv work/extra4.csv
"""
import argparse, csv, gc, json, re, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from synth import KIT  # noqa: E402
sys.path.insert(0, str(KIT))
import metrics as M  # noqa: E402

LOCAL = {"mms": "facebook/mms-1b-all", "whisper_hi": "vasista22/whisper-hindi-large-v2",
         "vakyansh": "Harveenchadha/vakyansh-wav2vec2-hindi-him-4200", "wavlm": "microsoft/wavlm-base-plus-sv"}


def _local(repo):
    """Use a copy in <kit>/work/models/<name> if present (avoids HF cache symlinks on Windows)."""
    p = KIT / "work" / "models" / repo.split("/")[1]
    return str(p) if p.exists() else repo


class CTC:
    """Greedy CTC recognizer (wav2vec2 family). Latin letters and symbols are suppressed so English
    words come out in Devanagari; digits are allowed and turned into words afterwards (normalize()),
    matching the scorer's "numbers as words" rule."""

    def __init__(self, device, repo, target_lang=None, blank_id=None):
        import torch
        from transformers import AutoProcessor, Wav2Vec2ForCTC
        mid = _local(repo)
        kw = {"target_lang": target_lang, "ignore_mismatched_sizes": True} if target_lang else {}
        if target_lang:
            self.proc = AutoProcessor.from_pretrained(mid, target_lang=target_lang)
        else:                                                # greedy decoding: skip any LM processor
            from transformers import Wav2Vec2Processor
            self.proc = Wav2Vec2Processor.from_pretrained(mid)
        self.model = Wav2Vec2ForCTC.from_pretrained(mid, torch_dtype=torch.float16 if device == "cuda" else torch.float32, **kw)
        self.model.to(device).eval()
        self.device = device
        bad = re.compile(r"[A-Za-z$€£]")
        tok = self.proc.tokenizer
        keep = set(tok.all_special_tokens) | {"|", tok.word_delimiter_token}
        self.bad_ids = sorted(i for t, i in tok.get_vocab().items() if bad.search(t) and t not in keep)
        # fairseq-converted models (Vakyansh) use id 0 ("<s>") as the CTC blank, not the pad token
        self.blank_id = blank_id

    def transcribe(self, waves):
        import torch
        from normalize import normalize
        out = []
        for y in waves:
            inp = self.proc(y, sampling_rate=16000, return_tensors="pt")
            x = inp["input_values"].to(self.device, dtype=self.model.dtype)
            with torch.no_grad():
                logits = self.model(x).logits[0].float()
            if self.bad_ids:
                logits[:, self.bad_ids] = -float("inf")
            ids = logits.argmax(-1)
            if self.blank_id is not None:
                ids[ids == self.blank_id] = self.proc.tokenizer.pad_token_id
            out.append(normalize(self.proc.decode(ids)))
        return out


class WhisperASR:
    def __init__(self, repo=None, batch_size=8):
        if repo is None:
            self.asr = M.load_whisper()                      # the kit's recognizer, exactly as metrics.py
        else:
            # metrics.load_whisper first tries pipeline(dtype=...), which transformers 4.49 rejects only
            # after loading an fp32 copy onto the GPU; for a 1.5 B model that copy alone fills 6 GB.
            import torch
            from transformers import pipeline
            dev = M.device()
            self.asr = pipeline("automatic-speech-recognition", model=_local(repo), device=dev,
                                torch_dtype=torch.float16 if dev == "cuda" else torch.float32)
            self.asr._bootcamp_suppress = M.whisper_suppress_ids(self.asr.tokenizer, self.asr.model.generation_config)
        self.batch_size = batch_size

    def transcribe(self, waves):
        return M.transcribe_whisper(self.asr, waves, batch_size=self.batch_size)


def load_asr(name, dev):
    if name == "whisper":
        return WhisperASR()
    if name == "whisper_hi":
        # its own generation_config.json predates language selection; work/models/whisper-hindi-large-v2
        # uses openai/whisper-large-v2's generation_config.json instead (same tokenizer and vocabulary)
        return WhisperASR(LOCAL["whisper_hi"], batch_size=4)
    if name == "mms":
        return CTC(dev, LOCAL["mms"], target_lang="hin")
    if name == "vakyansh":
        return CTC(dev, LOCAL["vakyansh"], blank_id=0)
    raise ValueError(name)


class WavLMSV:
    def __init__(self, device):
        from transformers import AutoFeatureExtractor, WavLMForXVector
        mid = _local(LOCAL["wavlm"])
        self.fe = AutoFeatureExtractor.from_pretrained(mid)
        self.model = WavLMForXVector.from_pretrained(mid).to(device).eval()
        self.device = device

    def embed(self, y):
        import torch
        inp = self.fe(y, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            e = self.model(**{k: v.to(self.device) for k, v in inp.items()}).embeddings[0]
        e = torch.nn.functional.normalize(e, dim=-1)
        return e.cpu().numpy()


def _free():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--asr", default="whisper,mms", help="comma-separated: whisper, mms, whisper_hi, vakyansh")
    ap.add_argument("--json")
    ap.add_argument("--csv", help="per-item scores of all folders")
    ap.add_argument("--no-whisper", action="store_true", help="drop the kit Whisper from --asr")
    ap.add_argument("--no-sim", action="store_true", help="skip the speaker-similarity models")
    a = ap.parse_args()
    asr_names = [n for n in a.asr.split(",") if n and not (a.no_whisper and n == "whisper")]

    items = M.load_items(KIT / "data" / "val_items.csv")
    prompts = M.load_prompts(KIT / "data" / "prompts.csv")
    dev = M.device()

    # read every folder once
    rows, waves = [], []
    for folder in a.folders:
        for it in items:
            path = Path(folder) / f"{it['id']}.wav"
            r = {"folder": folder, "id": it["id"], "category": it["category"], "prompt_id": it["prompt_id"],
                 "spoken": it["spoken"], "failed": False}
            y = None
            if path.exists():
                y, sr = M.load_audio(path)
                ok, _, _ = M.sanity(y, 16000, M.ref_chars(it["spoken"]))
                r["failed"] = not ok or sr != 16000
            else:
                r["failed"] = True
            rows.append(r)
            waves.append(y)
    good = [k for k, y in enumerate(waves) if y is not None]

    for name in asr_names:                                   # one recognizer at a time
        asr = load_asr(name, dev)
        hyps = asr.transcribe([waves[k] for k in good])
        for k, h in zip(good, hyps):
            rows[k][f"hyp_{name}"] = h
            rows[k][f"cer_{name}"] = M.cer(rows[k]["spoken"], h)
        del asr
        _free()

    sim_keys = []
    if not a.no_sim:
        ecapa = M.load_ecapa()
        wavlm = WavLMSV(dev)
        pemb = {}
        for pid, p in prompts.items():
            y = M.load_audio(p["path"])[0]
            pemb[pid] = (M.ecapa_embed(ecapa, y), wavlm.embed(y))
        for k in good:
            e1, e2 = pemb[rows[k]["prompt_id"]]
            rows[k]["sim_ecapa"] = float(np.dot(M.ecapa_embed(ecapa, waves[k]), e1))
            rows[k]["sim_wavlm"] = float(np.dot(wavlm.embed(waves[k]), e2))
        sim_keys = ["sim_ecapa", "sim_wavlm"]
        del ecapa, wavlm
        _free()

    keys = [f"cer_{n}" for n in asr_names] + sim_keys
    for r in rows:
        if r["failed"]:
            r.update({k: (1.0 if k.startswith("cer") else 0.0) for k in keys})

    summary = {}
    mean = lambda xs: round(float(np.mean(xs)), 4)
    for folder in a.folders:
        fr = [r for r in rows if r["folder"] == folder]
        cats = sorted({r["category"] for r in fr})
        s = {k: {"all": mean([r[k] for r in fr]), **{c: mean([r[k] for r in fr if r["category"] == c]) for c in cats}}
             for k in keys}
        s["failed"] = sum(r["failed"] for r in fr)
        summary[folder] = s

    cols = ["all", "hindi", "hinglish", "hard", "long"]
    for key in keys:
        print(f"\n{key}")
        print(f"{'folder':45s}" + "".join(f"{c:>10s}" for c in cols))
        for folder, s in summary.items():
            print(f"{folder[-45:]:45s}" + "".join(f"{s[key].get(c, float('nan')):>10.4f}" for c in cols))
    if a.json:
        Path(a.json).write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    if a.csv:
        fields = ["folder", "id", "category", "failed"] + keys + [f"hyp_{n}" for n in asr_names]
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
