#!/usr/bin/env python3
"""
Task 4 sweep: faster settings of the Task 3 system on the 24 validation items, several seeds.

Loads the model once, then for every (setting, seed) writes <out>/<name>_s<seed>/ and a timing
file. Score the folders afterwards with eval_extra.py (Whisper + MMS, ECAPA + WavLM) and compare
with the Task 3 system at the same seeds (compare_t4.py).

    python code/sweep_t4.py --out ../work/t4sweep --seeds 0,1,2 --configs u12,e12,e10,e7,e6,e5
Config names: u<N> = uniform steps + sway (as the baseline), e<N> = EPSS steps + sway,
optional c<K> suffix = guidance only on the first K steps (e.g. e7c6), p<points> = custom time
points in 1/32 units + sway (e.g. p0-2-4-6-8-12-16-24-32).
"""
import argparse, json, re, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import synth  # noqa: E402
from synth import KIT, read_csv  # noqa: E402
from fast import FastEngine  # noqa: E402


def parse(name):
    m = re.fullmatch(r"p((?:\d+-)+\d+)", name)            # p0-2-4-8-16-32: custom time points
    if m:
        pts = m.group(1)
        return dict(nfe_step=len(pts.split("-")) - 1, schedule=pts)
    m = re.fullmatch(r"([ue])(\d+)(?:c(\d+))?", name)
    if not m:
        raise ValueError(f"bad config name {name}")
    s = dict(nfe_step=int(m.group(2)), schedule={"u": "uniform", "e": "epss"}[m.group(1)])
    if m.group(3):
        s["cfg_steps"] = int(m.group(3))
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", default=str(KIT / "data" / "val_items.csv"))
    ap.add_argument("--text-csv", default=str(HERE / "t3_text.csv"))
    ap.add_argument("--text-column", default="t3_text")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--configs", required=True)
    ap.add_argument("--amp", default="bf16")
    ap.add_argument("--peak", type=float, default=0.95)
    a = ap.parse_args()

    import torch
    from baseline.tts import save_wav
    items = read_csv(a.items)
    repl = {r["id"]: r[a.text_column] for r in read_csv(a.text_csv)}
    for it in items:
        it["text"] = repl[it["id"]]
    prompts = {r["prompt_id"]: dict(r, path=str(KIT / "data" / r["file"])) for r in read_csv(KIT / "data" / "prompts.csv")}
    eng = FastEngine(noise="t4", amp=a.amp)
    p0 = prompts[items[0]["prompt_id"]]
    eng("नमस्ते, यह एक छोटा सा परीक्षण है।", p0["path"], p0["transcript"], seed=0)   # warm-up

    for name in a.configs.split(","):
        settings = parse(name)
        for seed in [int(s) for s in a.seeds.split(",")]:
            out = Path(a.out) / f"{name}_s{seed}"
            if (out.parent / f"{out.name}_timing.json").exists():
                continue
            out.mkdir(parents=True, exist_ok=True)
            total, per = 0.0, {}
            for it in items:
                p = prompts[it["prompt_id"]]
                torch.cuda.synchronize()
                t = time.time()
                y = eng(it["text"], p["path"], p["transcript"], seed=seed, **settings)
                torch.cuda.synchronize()
                per[it["id"]] = round(time.time() - t, 3)
                total += per[it["id"]]
                save_wav(out / f"{it['id']}.wav", synth.peak_guard(y, a.peak))
            (out.parent / f"{out.name}_timing.json").write_text(json.dumps(
                {"total_synth_s": round(total, 3), "seed": seed, "overrides": settings, "amp": a.amp,
                 "per_item_s": per}, indent=1), encoding="utf-8")
            print(f"{name} seed {seed}: {total:.1f}s", flush=True)


if __name__ == "__main__":
    main()
