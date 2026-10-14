"""Time each stage of a baseline synthesis (native settings, as run_baseline.py).

    python code/profile_baseline.py
"""
import csv, sys, time
from pathlib import Path
import numpy as np
import torch, torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from synth import KIT  # noqa: E402  (finds the kit folder)
sys.path.insert(0, str(KIT))
from baseline.tts import get_engine, resample, to_float, set_seed

items = {r["id"]: r for r in csv.DictReader(open(KIT / "data/val_items.csv", encoding="utf-8"))}
prompts = {r["prompt_id"]: r for r in csv.DictReader(open(KIT / "data/prompts.csv", encoding="utf-8"))}
eng = get_engine()
U = eng.U


def sync():
    torch.cuda.synchronize()


def stage(name, t0, log):
    sync()
    log[name] = log.get(name, 0) + time.time() - t0
    return time.time()


# warm-up
it = items["val_hi_02"]; p = prompts[it["prompt_id"]]
eng(it["text"], str(KIT / "data" / p["file"]), p["transcript"], seed=0)

for iid in ["val_hi_02", "val_hg_01", "val_hd_06", "val_hi_06"]:
    it = items[iid]; p = prompts[it["prompt_id"]]
    log = {}
    set_seed(0)
    t = time.time()
    ref_audio, ref_text = U.preprocess_ref_audio_text(str(KIT / "data" / p["file"]), p["transcript"], show_info=lambda *a: None)
    t = stage("preprocess_ref (pydub)", t, log)
    audio, sr = torchaudio.load(ref_audio)
    max_chars = int(len("".join(ref_text.split())) / (audio.shape[-1] / sr) * (25 - audio.shape[-1] / sr))
    batches = U.chunk_text(it["text"], max_chars=max_chars)
    rms = torch.sqrt(torch.mean(torch.square(audio)))
    if rms < 0.1:
        audio = audio * 0.1 / rms
    audio = torchaudio.transforms.Resample(sr, 24000)(audio) if sr != 24000 else audio
    audio = audio.to("cuda")
    t = stage("load+resample ref", t, log)
    waves = []
    for gen_text in batches:
        rt = ref_text + " " if len(ref_text[-1].encode("utf-8")) == 1 else ref_text
        text_list = U.convert_char_to_pinyin([rt + gen_text])
        ref_len = audio.shape[-1] // 256
        dur = ref_len + int(ref_len / len("".join(rt.split())) * len("".join(gen_text.split())))
        t = stage("text prep", t, log)
        with torch.inference_mode():
            gen, _ = eng.ema.sample(cond=audio, text=text_list, duration=dur, steps=32, cfg_strength=2.0, sway_sampling_coef=-1.0)
            t = stage("CFM sample (32 NFE x2 CFG)", t, log)
            mel = gen.to(torch.float32)[:, ref_len:, :].permute(0, 2, 1)
            w = eng.vocoder.decode(mel)
            t = stage("vocoder", t, log)
        waves.append(w.squeeze().cpu().numpy())
    y = resample(to_float(np.concatenate(waves)), 24000)
    t = stage("resample 16k", t, log)
    tot = sum(log.values())
    print(f"{iid}: total {tot:.2f}s | " + " | ".join(f"{k} {v:.2f}" for k, v in log.items()), flush=True)
