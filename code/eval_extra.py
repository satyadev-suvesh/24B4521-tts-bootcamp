#!/usr/bin/env python3
"""
Robustness check: score validation folders with recognizers / speaker models besides the kit's.

The final scorer averages several recognizers that are not in the kit, so a change that only helps
Whisper-turbo is suspect. This script scores each folder with
  * CER: kit Whisper-large-v3-turbo (same suppression as metrics.py) and Meta MMS-1B-all with its
    Hindi adapter (CTC, a different architecture and training set), using metrics.cer/canon
  * SIM: kit ECAPA-TDNN and WavLM-base-plus-sv (cosine to the prompt clip)
and prints per-category means for each folder side by side. Failed items (metrics.sanity) count as
CER 1 / SIM 0, as in metrics.py.

    python code/eval_extra.py data/baseline_val predictions/val_control predictions/val_idea --json work/extra.json
"""
import argparse, csv, json, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from synth import KIT  # noqa: E402
sys.path.insert(0, str(KIT))
import metrics as M  # noqa: E402


class MMS:
    def __init__(self, device):
        import torch
        from transformers import AutoProcessor, Wav2Vec2ForCTC
        mid = "facebook/mms-1b-all"
        self.proc = AutoProcessor.from_pretrained(mid, target_lang="hin")
        self.model = Wav2Vec2ForCTC.from_pretrained(mid, target_lang="hin", ignore_mismatched_sizes=True,
                                                    torch_dtype=torch.float16 if device == "cuda" else torch.float32)
        self.model.to(device).eval()
        self.device = device

    def transcribe(self, y):
        import torch
        inp = self.proc(y, sampling_rate=16000, return_tensors="pt")
        x = inp["input_values"].to(self.device, dtype=self.model.dtype)
        with torch.no_grad():
            ids = self.model(x).logits.argmax(-1)[0]
        return self.proc.decode(ids)


class WavLMSV:
    def __init__(self, device):
        import torch
        from transformers import AutoFeatureExtractor, WavLMForXVector
        mid = "microsoft/wavlm-base-plus-sv"
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--json")
    ap.add_argument("--csv", help="per-item scores of all folders")
    ap.add_argument("--no-whisper", action="store_true")
    a = ap.parse_args()

    items = M.load_items(KIT / "data" / "val_items.csv")
    prompts = M.load_prompts(KIT / "data" / "prompts.csv")
    dev = M.device()
    whisper = None if a.no_whisper else M.load_whisper()
    ecapa = M.load_ecapa()
    mms = MMS(dev)
    wavlm = WavLMSV(dev)
    pemb = {}
    for pid, p in prompts.items():
        y = M.load_audio(p["path"])[0]
        pemb[pid] = (M.ecapa_embed(ecapa, y), wavlm.embed(y))

    summary, per_item = {}, []
    for folder in a.folders:
        rows, waves = [], []
        for it in items:
            path = Path(folder) / f"{it['id']}.wav"
            r = {"folder": folder, "id": it["id"], "category": it["category"], "failed": False}
            if not path.exists():
                r["failed"] = True
                rows.append(r); waves.append(None); continue
            y, sr = M.load_audio(path)
            ok, reasons, _ = M.sanity(y, 16000, M.ref_chars(it["spoken"]))
            r["failed"] = not ok or sr != 16000
            rows.append(r); waves.append(y)
        good = [k for k, y in enumerate(waves) if y is not None]
        hyp_w = M.transcribe_whisper(whisper, [waves[k] for k in good]) if whisper else [""] * len(good)
        for k, hw in zip(good, hyp_w):
            r, it, y = rows[k], items[k], waves[k]
            hm = mms.transcribe(y)
            e1, e2 = pemb[it["prompt_id"]]
            r.update(hyp_whisper=hw, hyp_mms=hm,
                     cer_whisper=M.cer(it["spoken"], hw) if whisper else None,
                     cer_mms=M.cer(it["spoken"], hm),
                     sim_ecapa=float(np.dot(M.ecapa_embed(ecapa, y), e1)),
                     sim_wavlm=float(np.dot(wavlm.embed(y), e2)))
        for r in rows:
            if r["failed"]:
                r.update(cer_whisper=1.0, cer_mms=1.0, sim_ecapa=0.0, sim_wavlm=0.0)
        per_item += rows
        cats = sorted({r["category"] for r in rows})
        mean = lambda xs: round(float(np.mean(xs)), 4)
        s = {}
        for key in ["cer_whisper", "cer_mms", "sim_ecapa", "sim_wavlm"]:
            if rows[0].get(key) is None:
                continue
            s[key] = {"all": mean([r[key] for r in rows]), **{c: mean([r[key] for r in rows if r["category"] == c]) for c in cats}}
        s["failed"] = sum(r["failed"] for r in rows)
        summary[folder] = s

    keys = ["cer_whisper", "cer_mms", "sim_ecapa", "sim_wavlm"]
    for key in keys:
        if not any(key in s for s in summary.values()):
            continue
        print(f"\n{key}")
        cols = ["all", "hindi", "hinglish", "hard", "long"]
        print(f"{'folder':45s}" + "".join(f"{c:>10s}" for c in cols))
        for folder, s in summary.items():
            if key in s:
                print(f"{folder[-45:]:45s}" + "".join(f"{s[key].get(c, float('nan')):>10.4f}" for c in cols))
    if a.json:
        Path(a.json).write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    if a.csv:
        keys_all = ["folder", "id", "category", "failed", "cer_whisper", "cer_mms", "sim_ecapa", "sim_wavlm", "hyp_whisper", "hyp_mms"]
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys_all, extrasaction="ignore")
            w.writeheader()
            w.writerows(per_item)


if __name__ == "__main__":
    main()
