#!/usr/bin/env python3
"""
Task 4 - a faster way to run the same IndicF5 model (same weights, same inference maths).

Profiling the baseline (work/profile_baseline.py) shows 99 % of the time is CFM.sample(): 32
Euler steps x 2 transformer passes (conditional + unconditional for classifier-free guidance).
Prompt preprocessing, vocoder and resampling take < 1 %. This engine re-implements
CFM.sample() + infer_process() step by step and only changes how the work is executed:

  * both guidance branches run as ONE batch of 2 instead of two calls      (same maths)
  * the text embedding (text -> ConvNeXt) is computed once per item, not at every step,
    because it does not depend on the noisy audio or the time step        (same maths)
  * the reference clip is preprocessed once per prompt and cached          (same maths)
  * optional bf16 autocast for the transformer (fp16 overflows in this model; bf16 keeps the fp32 range)
  * the T4 initial noise is drawn on the GPU (t4_noise.randn_torch) instead of with numpy
  * optional fewer flow-matching steps (`nfe_step`) on a chosen time grid (`--schedule`: the
    baseline's uniform grid, EPSS, or custom points); both are allowed decoding settings

With fp32, 32 uniform steps and --cpu-noise it reproduces the baseline (checked with --check).
Duration estimate, text chunking, cross-fade, RMS handling and the initial noise are the baseline's.

    # Task 4 system (see run_task4.ps1)
    python code/fast.py --items ../data/val_items.csv --text-csv code/t3_text.csv --text-column t3_text \
        --out ../work/val_fast --nfe 8 --schedule 0-2-4-6-8-12-16-24-32 --amp bf16
"""
import argparse, csv, json, platform, sys, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import synth  # noqa: E402  (finds the kit, loads the baseline engine)
from synth import KIT, read_csv  # noqa: E402

TARGET_SR, HOP = 24000, 256

# Empirically Pruned Step Sampling (Zheng et al. 2025, "Accelerating Flow-Matching-Based TTS via
# Empirically Pruned Step Sampling"; the same table as upstream F5-TTS get_epss_timesteps): time
# points in 1/32 units. Few-step sampling keeps the fine early steps of the 32-step sway schedule
# and drops the redundant late ones; sway sampling is applied on top, as upstream.
EPSS = {5: [0, 2, 4, 8, 16, 32], 6: [0, 2, 4, 6, 8, 16, 32], 7: [0, 2, 4, 6, 8, 16, 24, 32],
        10: [0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32], 12: [0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32],
        16: [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32]}


class FastEngine:
    def __init__(self, noise="t4", amp="bf16", gpu_noise=True):
        import torch
        self.torch = torch
        self.base = synth.load_engine(noise)            # baseline weights, pinned revision
        self.noise = noise
        if noise == "t4" and gpu_noise:                  # same T4 noise, drawn on the GPU (~1 ms, not ~50 ms)
            import t4_noise
            t4_noise._GEN.device = self.base.device
        self.U = self.base.U
        cfm = self.base.ema
        self.cfm = getattr(cfm, "_orig_mod", cfm)        # unwrap torch.compile wrapper
        self.dit = self.cfm.transformer
        self.vocoder = getattr(self.base.vocoder, "_orig_mod", self.base.vocoder)
        self.vocab = self.cfm.vocab_char_map
        self.amp = {"none": None, "fp16": torch.float16, "bf16": torch.bfloat16}[amp]
        self.device = self.base.device
        self._ref = {}

    # -- reference clip: identical to preprocess_ref_audio_text + infer_batch_process prelude --
    def ref(self, prompt_wav, prompt_text, target_rms):
        key = (str(prompt_wav), prompt_text, target_rms)
        if key not in self._ref:
            import torchaudio
            from baseline.tts import _filter_kwargs, _quiet
            kw, _ = _filter_kwargs(self.U.preprocess_ref_audio_text, {"show_info": _quiet})
            ref_audio, ref_text = self.U.preprocess_ref_audio_text(str(prompt_wav), prompt_text, **kw)
            audio, sr = torchaudio.load(ref_audio)
            if audio.shape[0] > 1:
                audio = audio.mean(dim=0, keepdim=True)
            rms = self.torch.sqrt(self.torch.mean(self.torch.square(audio)))
            ref_dur = audio.shape[-1] / sr
            ref_chars = len("".join(ref_text.split()))           # the kit's length fix
            max_chars = int(ref_chars / ref_dur * (25 - ref_dur))
            if rms < target_rms:
                audio = audio * target_rms / rms
            if sr != TARGET_SR:
                audio = torchaudio.transforms.Resample(sr, TARGET_SR)(audio)
            audio = audio.to(self.device)
            if len(ref_text[-1].encode("utf-8")) == 1:
                ref_text = ref_text + " "
            with self.torch.inference_mode():
                cond = self.cfm.mel_spec(audio).permute(0, 2, 1)  # (1, n_ref, 100)
            self._ref[key] = dict(cond=cond, ref_text=ref_text, rms=rms, max_chars=max_chars,
                                  ref_len=audio.shape[-1] // HOP)
        return self._ref[key]

    # -- one transformer pass for both guidance branches --------------------------------------
    def _flow(self, x, cond2, text2, t):
        dit = self.dit
        batch, seq_len = x.shape[0], x.shape[1]
        tt = dit.time_embed(t.reshape(1).repeat(batch))
        h = dit.input_embed(x, cond2, text2, drop_audio_cond=False)
        rope = dit.rotary_embed.forward_from_seq_len(seq_len)
        for block in dit.transformer_blocks:
            h = block(h, tt, mask=None, rope=rope)
        h = dit.norm_out(h, tt)
        return dit.proj_out(h)

    def sample(self, cond, text_list, duration, steps, cfg_strength, sway, cfg_steps=None, schedule="uniform"):
        """Same as CFM.sample() for batch 1 (euler, sway sampling, CFG), batched CFG.
        cfg_steps: apply guidance only on the first cfg_steps steps (None = all, as the baseline).
        schedule: "uniform" (the baseline's time points) or "epss" (EPSS table for 5-7/10/12/16 steps)."""
        torch = self.torch
        from f5_tts.model.utils import list_str_to_idx, lens_to_mask
        cond = cond.to(torch.float32)
        cond_seq_len = cond.shape[1]
        text = list_str_to_idx(text_list, self.vocab).to(self.device)
        text_lens = (text != -1).sum(dim=-1)
        lens = torch.maximum(text_lens, torch.full((1,), cond_seq_len, device=self.device, dtype=torch.long))
        cond_mask = lens_to_mask(lens)
        dur = torch.full((1,), duration, device=self.device, dtype=torch.long)
        dur = torch.maximum(lens + 1, dur).clamp(max=4096)
        n = int(dur.amax())
        cond = torch.nn.functional.pad(cond, (0, 0, 0, n - cond_seq_len), value=0.0)
        cond_mask = torch.nn.functional.pad(cond_mask, (0, n - cond_mask.shape[-1]), value=False).unsqueeze(-1)
        step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))

        # initial noise: the same call (and so the same numbers) as CFM.sample
        import f5_tts.model.cfm as cfm_mod
        y = cfm_mod.torch.randn(n, self.cfm.num_channels, device=self.device, dtype=torch.float32)[None]

        ac = torch.autocast("cuda", dtype=self.amp or torch.float32, enabled=self.amp is not None)
        with ac:
            text_c = self.dit.text_embed(text, n, drop_text=False)
            text_u = self.dit.text_embed(text, n, drop_text=True)
        text2 = torch.cat([text_c, text_u], 0)
        cond2 = torch.cat([step_cond, torch.zeros_like(step_cond)], 0)

        if schedule == "epss" and steps in EPSS:
            t = torch.tensor(EPSS[steps], device=self.device, dtype=torch.float32) / 32
        elif schedule in ("uniform", "epss"):
            t = torch.linspace(0, 1, steps + 1, device=self.device, dtype=torch.float32)
        else:                                            # custom time points in 1/32, e.g. "0-2-4-8-16-32"
            pts = [int(v) for v in schedule.split("-")]
            if len(pts) != steps + 1 or pts[0] != 0 or pts[-1] != 32 or sorted(pts) != pts:
                raise ValueError(f"bad schedule {schedule} for {steps} steps")
            t = torch.tensor(pts, device=self.device, dtype=torch.float32) / 32
        if sway is not None:
            t = t + sway * (torch.cos(torch.pi / 2 * t) - 1 + t)
        for i in range(steps):
            if cfg_steps is None or i < cfg_steps:
                with ac:
                    out = self._flow(torch.cat([y, y], 0), cond2, text2, t[i])
                out = out.float()
                pred, null = out[:1], out[1:]
                v = pred + (pred - null) * cfg_strength
            else:
                with ac:
                    v = self._flow(y, cond2[:1], text2[:1], t[i]).float()
            y = y + (t[i + 1] - t[i]) * v
        return torch.where(cond_mask, cond, y)

    def __call__(self, text, prompt_wav, prompt_text, seed=0, nfe_step=32, cfg_strength=2.0,
                 sway_sampling_coef=-1.0, speed=1.0, cross_fade_duration=0.15, target_rms=0.1,
                 cfg_steps=None, schedule="uniform"):
        torch = self.torch
        from baseline.tts import set_seed, resample, to_float
        set_seed(seed)
        if self.noise == "t4":
            import t4_noise
            t4_noise.reset(seed)
        r = self.ref(prompt_wav, prompt_text, target_rms)
        batches = self.U.chunk_text(text, max_chars=r["max_chars"])
        ref_chars = len("".join(r["ref_text"].split()))
        waves = []
        with torch.inference_mode():
            for gen_text in batches:
                text_list = self.U.convert_char_to_pinyin([r["ref_text"] + gen_text])
                gen_chars = len("".join(gen_text.split()))
                duration = r["ref_len"] + int(r["ref_len"] / ref_chars * gen_chars / speed)
                mel = self.sample(r["cond"], text_list, duration, nfe_step, cfg_strength, sway_sampling_coef,
                                  cfg_steps, schedule)
                mel = mel[:, r["ref_len"]:, :].permute(0, 2, 1)
                w = self.vocoder.decode(mel)
                if r["rms"] < target_rms:
                    w = w * r["rms"] / target_rms
                waves.append(w.squeeze().cpu().numpy())
        y = waves[0]
        for nxt in waves[1:]:
            k = min(int(cross_fade_duration * TARGET_SR), len(y), len(nxt))
            if k <= 0:
                y = np.concatenate([y, nxt])
                continue
            y = np.concatenate([y[:-k], y[-k:] * np.linspace(1, 0, k) + nxt[:k] * np.linspace(0, 1, k), nxt[k:]])
        return resample(to_float(y), TARGET_SR)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", required=True)
    ap.add_argument("--prompts", default=str(KIT / "data" / "prompts.csv"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ids")
    ap.add_argument("--text-csv")
    ap.add_argument("--text-column", default="normalized_text")
    ap.add_argument("--nfe", type=int, default=16)
    ap.add_argument("--cfg-steps", type=int, default=None, help="guidance only on the first N steps")
    ap.add_argument("--schedule", default="uniform",
                    help="flow time points: 'uniform' (the baseline's grid), 'epss', or custom points in "
                         "1/32 units such as 0-2-4-6-8-12-16-24-32; sway sampling is applied on top")
    ap.add_argument("--amp", choices=["none", "fp16", "bf16"], default="none", help="autocast dtype for the transformer")
    ap.add_argument("--noise", choices=["t4", "native"], default="t4")
    ap.add_argument("--cpu-noise", action="store_true", help="draw the T4 noise with numpy (slower) instead of on the GPU")
    ap.add_argument("--peak", type=float, default=0.95, help="peak guard against clipping (0 = off)")
    ap.add_argument("--no-warmup", action="store_true")
    ap.add_argument("--check", action="store_true", help="compare with the baseline engine on the first item")
    a = ap.parse_args()

    items = read_csv(a.items if Path(a.items).exists() else KIT / a.items)
    if a.ids:
        keep = set(a.ids.split(","))
        items = [it for it in items if it["id"] in keep]
    if a.text_csv:
        repl = {r["id"]: r[a.text_column] for r in read_csv(a.text_csv)}
        for it in items:
            it["text"] = repl[it["id"]]
    pbase = Path(a.prompts).resolve().parent
    prompts = {r["prompt_id"]: dict(r, path=str(pbase / r["file"])) for r in read_csv(a.prompts)}

    import torch
    from baseline.tts import save_wav
    t0 = time.time()
    eng = FastEngine(noise=a.noise, amp=a.amp, gpu_noise=not a.cpu_noise)
    load_s = time.time() - t0
    settings = dict(nfe_step=a.nfe)
    if a.cfg_steps is not None:
        settings["cfg_steps"] = a.cfg_steps
    if a.schedule != "uniform":
        settings["schedule"] = a.schedule
    if a.check and items:
        it = items[0]; p = prompts[it["prompt_id"]]
        yb = synth.synth_one(eng.base, it["text"], p["path"], p["transcript"], a.seed, a.noise)
        yf = eng(it["text"], p["path"], p["transcript"], seed=a.seed, **settings)
        m = min(len(yb), len(yf))
        print(f"check {it['id']}: len {len(yb)} vs {len(yf)}, corr {np.corrcoef(yb[:m], yf[:m])[0, 1]:.5f}", flush=True)
    if not a.no_warmup and items:
        p = prompts[items[0]["prompt_id"]]
        eng("नमस्ते, यह एक छोटा सा परीक्षण है।", p["path"], p["transcript"], seed=a.seed, **settings)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    per_item, total = {}, 0.0
    for i, it in enumerate(items, 1):
        p = prompts[it["prompt_id"]]
        torch.cuda.synchronize()
        t = time.time()
        y = eng(it["text"], p["path"], p["transcript"], seed=a.seed, **settings)
        torch.cuda.synchronize()
        dt = time.time() - t
        total += dt
        save_wav(out / f"{it['id']}.wav", synth.peak_guard(y, a.peak))
        per_item[it["id"]] = round(dt, 3)
        print(f"[{i}/{len(items)}] {it['id']}: {dt:.2f}s synth, {len(y) / 16000:.2f}s audio", flush=True)

    gpu = torch.cuda.get_device_name(0)
    timing = {"total_synth_s": round(total, 3), "n_items": len(items), "model_load_s": round(load_s, 1),
              "seed": a.seed, "overrides": settings, "amp": a.amp, "batched_cfg": True, "peak_guard": a.peak,
              "noise": a.noise, "noise_on": "cpu" if a.cpu_noise else "gpu", "gpu": gpu, "python": platform.python_version(),
              "items_file": str(a.items), "text_csv": a.text_csv, "per_item_s": per_item}
    tpath = out.parent / f"{out.name}_timing.json"
    tpath.write_text(json.dumps(timing, indent=1), encoding="utf-8")
    print(f"\nDone: {len(items)} items in {total:.1f}s of synthesis on {gpu}. Timing in {tpath}")


if __name__ == "__main__":
    main()
