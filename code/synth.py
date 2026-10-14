#!/usr/bin/env python3
"""
Synthesize an item file with the kit's baseline wrapper (baseline/tts.py), like
baseline/run_baseline.py, with two additions for reproducibility on non-T4 GPUs:

  --noise t4   draw IndicF5's initial noise exactly as a T4 does (see t4_noise.py), so outputs
               match the kit's data/baseline_val/ (made on a T4) up to float rounding
  TF32 is disabled for cuDNN convolutions (a T4 has no TF32), again to match T4 numerics

Everything else is the baseline: same model revision, config.yaml decoding values, seed, text
handling, 16 kHz PCM_16 output. Timing excludes model loading and one warm-up item, as in
run_baseline.py.

    # Task 2: eval items, normalized text, baseline otherwise
    python code/synth.py --items data/eval_items.csv --text-csv predictions/t2_normalized.csv \
        --text-column normalized_text --out predictions/t2_norm --noise t4
"""
import argparse, csv, json, os, platform, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def kit_dir():
    env = os.environ.get("TTS_KIT")
    if env:
        return Path(env).resolve()
    for p in HERE.parents:
        if (p / "metrics.py").exists() and (p / "baseline" / "tts.py").exists():
            return p
    sys.exit("Cannot find the kit folder; set TTS_KIT=/path/to/tts_bootcamp_kit")


KIT = kit_dir()
sys.path.insert(0, str(KIT))


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_value(v):
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return {"true": True, "false": False}.get(v.lower(), v)


def load_engine(noise="t4"):
    import torch
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    from baseline.tts import get_engine
    engine = get_engine()
    if noise == "t4":
        import t4_noise
        t4_noise.install("T4")
    return engine


def synth_one(engine, text, prompt_path, prompt_text, seed=0, noise="t4", **overrides):
    if noise == "t4":
        import t4_noise
        t4_noise.reset(seed)
    return engine(text, prompt_path, prompt_text, seed=seed, **overrides)


def peak_guard(y, peak=0.95):
    """Loudness normalization (allowed post-processing): if the waveform would clip, scale the
    whole utterance so its peak is `peak`. Outputs that do not exceed `peak` are untouched.
    IndicF5 copies the prompt's loudness; with loud prompts (RMS > target_rms the wrapper does not
    rescale) some outputs exceed full scale and fail the kit's clipping check (> 0.1 % of samples
    at |x| >= 0.999) once save_wav clips them."""
    import numpy as np
    m = float(np.abs(y).max()) if len(y) else 0.0
    if peak and m > peak:
        y = y * (peak / m)
    return y


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", required=True)
    ap.add_argument("--prompts", default=str(KIT / "data" / "prompts.csv"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ids", help="comma-separated subset of item ids")
    ap.add_argument("--text-csv", help="CSV with an `id` column and replacement text")
    ap.add_argument("--text-column", default="normalized_text")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="decoding overrides")
    ap.add_argument("--noise", choices=["t4", "native"], default="t4")
    ap.add_argument("--peak", type=float, default=0.95, help="peak guard against clipping (0 = off)")
    ap.add_argument("--no-warmup", action="store_true")
    a = ap.parse_args()

    items_path = Path(a.items) if Path(a.items).exists() else KIT / a.items
    items = read_csv(items_path)
    if a.ids:
        keep = set(a.ids.split(","))
        items = [it for it in items if it["id"] in keep]
    if a.text_csv:
        repl = {r["id"]: r[a.text_column] for r in read_csv(a.text_csv)}
        missing = [it["id"] for it in items if it["id"] not in repl]
        if missing:
            sys.exit(f"{a.text_csv} lacks ids: {missing[:5]}")
        for it in items:
            it["text"] = repl[it["id"]]
    pbase = Path(a.prompts).resolve().parent
    prompts = {r["prompt_id"]: dict(r, path=str(pbase / r["file"])) for r in read_csv(a.prompts)}
    overrides = {k: parse_value(v) for k, v in (kv.split("=", 1) for kv in a.set)}

    import torch
    from baseline.tts import save_wav
    t0 = time.time()
    engine = load_engine(a.noise)
    load_s = time.time() - t0
    if not a.no_warmup and items:
        p = prompts[items[0]["prompt_id"]]
        synth_one(engine, "नमस्ते, यह एक छोटा सा परीक्षण है।", p["path"], p["transcript"], a.seed, a.noise, **overrides)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    per_item, total = {}, 0.0
    for i, it in enumerate(items, 1):
        p = prompts[it["prompt_id"]]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t = time.time()
        y = synth_one(engine, it["text"], p["path"], p["transcript"], a.seed, a.noise, **overrides)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.time() - t
        total += dt
        save_wav(out / f"{it['id']}.wav", peak_guard(y, a.peak))
        per_item[it["id"]] = round(dt, 3)
        print(f"[{i}/{len(items)}] {it['id']}: {dt:.2f}s synth, {len(y) / 16000:.2f}s audio", flush=True)

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    timing = {"total_synth_s": round(total, 3), "n_items": len(items), "model_load_s": round(load_s, 1),
              "seed": a.seed, "overrides": overrides, "noise": a.noise, "peak_guard": a.peak, "gpu": gpu,
              "python": platform.python_version(), "items_file": str(a.items),
              "text_csv": a.text_csv, "per_item_s": per_item}
    tpath = out.parent / f"{out.name}_timing.json"
    tpath.write_text(json.dumps(timing, indent=1), encoding="utf-8")
    print(f"\nDone: {len(items)} items in {total:.1f}s of synthesis on {gpu}. Timing in {tpath}")


if __name__ == "__main__":
    main()
