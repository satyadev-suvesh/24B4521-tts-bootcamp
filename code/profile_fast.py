"""Per-stage time of code/fast.py (Task 3 text, bf16) on the 24 validation items.

    python code/profile_fast.py 8 0-2-4-6-8-12-16-24-32      (steps, schedule)
"""
import csv, sys, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fast, synth  # noqa: E402
from synth import KIT  # noqa: E402
from baseline.tts import set_seed, resample, to_float  # noqa: E402
import t4_noise  # noqa: E402

steps = int(sys.argv[1]) if len(sys.argv) > 1 else 7
schedule = sys.argv[2] if len(sys.argv) > 2 else "epss"
items = list(csv.DictReader(open(KIT / "data/val_items.csv", encoding="utf-8")))
txt = {r["id"]: r["t3_text"] for r in csv.DictReader(open(HERE / "t3_text.csv", encoding="utf-8"))}
prompts = {r["prompt_id"]: dict(r, path=str(KIT / "data" / r["file"])) for r in csv.DictReader(open(KIT / "data/prompts.csv", encoding="utf-8"))}
eng = fast.FastEngine(noise="t4", amp="bf16")
p0 = prompts[items[0]["prompt_id"]]
eng("नमस्ते, यह एक छोटा सा परीक्षण है।", p0["path"], p0["transcript"], seed=0, nfe_step=steps, schedule=schedule)

log = {}
def tick(name, t0):
    torch.cuda.synchronize()
    t1 = time.time()
    log[name] = log.get(name, 0.0) + t1 - t0
    return t1

# instrument the noise draw
orig = t4_noise._Gen.randn
def timed_randn(self, *shape):
    t = time.time()
    v = orig(self, *shape)
    log["noise draw (CPU side)"] = log.get("noise draw (CPU side)", 0.0) + time.time() - t
    return v
t4_noise._Gen.randn = timed_randn

total0 = time.time()
for it in items:
    p = prompts[it["prompt_id"]]
    t = time.time()
    set_seed(0); t4_noise.reset(0)
    r = eng.ref(p["path"], p["transcript"], 0.1)
    t = tick("ref (cached)", t)
    batches = eng.U.chunk_text(txt[it["id"]], max_chars=r["max_chars"])
    ref_chars = len("".join(r["ref_text"].split()))
    waves = []
    with torch.inference_mode():
        for gen_text in batches:
            text_list = eng.U.convert_char_to_pinyin([r["ref_text"] + gen_text])
            gen_chars = len("".join(gen_text.split()))
            duration = r["ref_len"] + int(r["ref_len"] / ref_chars * gen_chars)
            t = tick("text prep", t)
            mel = eng.sample(r["cond"], text_list, duration, steps, 2.0, -1.0, None, schedule)
            t = tick("sample (incl. noise)", t)
            mel = mel[:, r["ref_len"]:, :].permute(0, 2, 1)
            w = eng.vocoder.decode(mel)
            if r["rms"] < 0.1:
                w = w * r["rms"] / 0.1
            waves.append(w.squeeze().cpu().numpy())
            t = tick("vocoder", t)
    y = waves[0]
    for nxt in waves[1:]:
        k = min(int(0.15 * 24000), len(y), len(nxt))
        y = np.concatenate([y[:-k], y[-k:] * np.linspace(1, 0, k) + nxt[:k] * np.linspace(0, 1, k), nxt[k:]])
    y = resample(to_float(y), 24000)
    t = tick("crossfade + resample", t)
torch.cuda.synchronize()
tot = time.time() - total0
print(f"total {tot:.2f}s for 24 items ({steps} steps, {schedule})")
for k, v in log.items():
    print(f"  {k:24s} {v:7.2f}s  {100 * v / tot:5.1f}%")
