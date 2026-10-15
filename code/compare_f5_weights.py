#!/usr/bin/env python3
"""
Task 3 analysis: how IndicF5's weights relate to the F5-TTS base model it was fine-tuned from.

IndicF5's vocabulary keeps F5-TTS's first 1700 entries (ASCII, Latin letters, digits, pinyin) at the
same indices and reuses entries 1700-2397 (Hebrew, Korean, ... in F5-TTS) for Indic characters. This
script compares, row by row, the character-embedding table of IndicF5 with F5-TTS Base
(SWivid/F5-TTS, F5TTS_Base/model_1200000.safetensors) and reports the relative change of every
other weight matrix, to see which parts fine-tuning moved.

    python code/compare_f5_weights.py --f5 ../work/models/f5base/F5TTS_Base_model_1200000.safetensors
"""
import argparse, re, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from synth import KIT  # noqa: E402  (finds the kit)


def indicf5_files():
    from huggingface_hub import snapshot_download
    d = Path(snapshot_download("ai4bharat/IndicF5", allow_patterns=["model.safetensors", "checkpoints/vocab.txt"]))
    return d / "model.safetensors", d / "checkpoints" / "vocab.txt"


def load(path, prefix):
    from safetensors import safe_open
    out = {}
    with safe_open(str(path), "pt") as f:
        for k in f.keys():
            if k.startswith(prefix):
                out[k[len(prefix):]] = f.get_tensor(k).float().numpy()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--f5", required=True, help="F5TTS_Base/model_1200000.safetensors")
    ap.add_argument("--f5-vocab", help="F5TTS_Base/vocab.txt (default: next to --f5)")
    a = ap.parse_args()
    ind_path, ind_vocab_path = indicf5_files()
    f5_vocab_path = a.f5_vocab or str(Path(a.f5).parent / "F5TTS_Base_vocab.txt")
    ind = load(ind_path, "ema_model._orig_mod.transformer.")
    f5 = load(a.f5, "ema_model.transformer.")
    v_ind = open(ind_vocab_path, encoding="utf-8").read().split("\n")
    v_f5 = open(f5_vocab_path, encoding="utf-8").read().split("\n")

    E_i, E_f = ind["text_embed.text_embed.weight"], f5["text_embed.text_embed.weight"]
    # row 0 is the filler/padding token; vocab line k is embedding row k + 1
    def row(E, k):
        return E[k + 1]

    def cos(x, y):
        return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))

    groups = {
        "Latin letters a-z": [k for k, t in enumerate(v_ind) if re.fullmatch(r"[a-z]", t)],
        "Latin letters A-Z": [k for k, t in enumerate(v_ind) if re.fullmatch(r"[A-Z]", t)],
        "digits 0-9": [k for k, t in enumerate(v_ind) if re.fullmatch(r"[0-9]", t)],
        "punctuation , . ? !": [k for k, t in enumerate(v_ind) if t in {",", ".", "?", "!"}],
        "pinyin syllables": [k for k, t in enumerate(v_ind) if re.fullmatch(r"[a-z]+[1-5]", t)],
        "Devanagari (rows reused from F5's other scripts)": [k for k, t in enumerate(v_ind) if t and all("ऀ" <= c <= "ॿ" for c in t)],
    }
    print(f"character embeddings, IndicF5 vs F5-TTS Base (same row index): cosine and norm ratio")
    print(f"{'group':48s} {'n':>4s} {'cos mean':>9s} {'cos min':>8s} {'|E_ind|/|E_f5|':>15s}")
    for name, ks in groups.items():
        ks = [k for k in ks if k + 1 < len(E_i)]
        same_tok = [k for k in ks if k < len(v_f5) and v_f5[k] == v_ind[k]]
        c = [cos(row(E_i, k), row(E_f, k)) for k in ks]
        r = [np.linalg.norm(row(E_i, k)) / (np.linalg.norm(row(E_f, k)) + 1e-12) for k in ks]
        print(f"{name:48s} {len(ks):4d} {np.mean(c):9.3f} {np.min(c):8.3f} {np.mean(r):15.2f}"
              + ("" if len(same_tok) == len(ks) else f"   ({len(ks) - len(same_tok)} rows hold a different token in F5)"))
    # nearest Devanagari neighbours of a few Latin letters in IndicF5's table vs F5's
    print("\nrelative change ||W_ind - W_f5|| / ||W_f5|| of the other weights (mean per group)")
    grp = {}
    for k, w in f5.items():
        if k not in ind or k == "text_embed.text_embed.weight" or w.ndim < 2 or ind[k].shape != w.shape:
            continue
        g = re.sub(r"\.\d+\.", ".N.", k).rsplit(".", 1)[0]
        g = "transformer_blocks" if k.startswith("transformer_blocks") else g.split(".")[0]
        grp.setdefault(g, []).append(np.linalg.norm(ind[k] - w) / (np.linalg.norm(w) + 1e-12))
    for g, v in sorted(grp.items()):
        print(f"  {g:24s} {len(v):4d} matrices  mean {np.mean(v):.3f}  max {np.max(v):.3f}")


if __name__ == "__main__":
    main()
