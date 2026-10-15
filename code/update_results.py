#!/usr/bin/env python3
"""
Fill results.json from metrics.py summaries (the numbers must match metrics.py within 0.01).

    python code/update_results.py --task2 work/val_control.json
    python code/update_results.py --task3 work/val_idea.json --control work/val_control.json \
        --method "..." --settings '{"nfe_step": 32}'
    python code/update_results.py --task4 work/val_fast.json --baseline-timing code/timing/baseline_val_timing.json \
        --fast-timing code/timing/val_fast_timing.json --method "..." --settings '{...}'
"""
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results.json"))
    ap.add_argument("--task2")
    ap.add_argument("--task3")
    ap.add_argument("--control")
    ap.add_argument("--task4")
    ap.add_argument("--baseline-timing")
    ap.add_argument("--fast-timing")
    ap.add_argument("--method")
    ap.add_argument("--settings")
    a = ap.parse_args()
    res = load(a.results)
    if a.task2:
        s = load(a.task2)
        res["task2"].update(val_cer=s["cer"], val_sim=s["sim"], val_cer_hard=s["cer_by_category"]["hard"])
    if a.task3:
        s = load(a.task3)
        t3 = res["task3"]
        cat = t3.get("targeted_category") or "hinglish"
        t3.update(val_cer=s["cer"], val_sim=s["sim"], val_cer_targeted=s["cer_by_category"][cat])
        if a.control:
            t3["val_control_cer"] = load(a.control)["cer"]
        if a.method:
            t3["method_name"] = a.method
        if a.settings:
            t3["settings"] = json.loads(a.settings)
        elif not t3.get("settings"):
            # Task 3 decodes exactly like the baseline; record those values and the text pipeline
            import yaml
            from synth import KIT
            cfg = yaml.safe_load((KIT / "baseline" / "config.yaml").read_text(encoding="utf-8"))
            t3["settings"] = {**cfg["decoding"], "seed": res.get("seed", 0), "noise": "T4 (code/t4_noise.py)",
                              "text": "normalize() (Task 2) then transliterate() (code/translit.py)"}
    if a.task4:
        s = load(a.task4)
        t4 = res["task4"]
        t4["val_cer"] = s["cer"]
        if a.baseline_timing and a.fast_timing:
            b = load(a.baseline_timing)["total_synth_s"]
            f = load(a.fast_timing)["total_synth_s"]
            t4.update(baseline_seconds=b, fast_seconds=f, speedup=round(b / f, 3))
        if a.method:
            t4["method_name"] = a.method
        if a.settings:
            t4["settings"] = json.loads(a.settings)
        elif a.fast_timing:
            ft = load(a.fast_timing)
            t4["settings"] = {**ft.get("overrides", {}), "amp": ft.get("amp"), "batched_cfg": ft.get("batched_cfg"),
                              "cached_text_embedding": True, "cached_prompt": True, "noise": ft.get("noise"),
                              "noise_on": ft.get("noise_on", "cpu"),
                              "peak_guard": ft.get("peak_guard"), "gpu": ft.get("gpu")}
    Path(a.results).write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
