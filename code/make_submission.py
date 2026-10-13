#!/usr/bin/env python3
"""
Build <rollno>_tts_bootcamp.zip from this repository and run the kit's check_format.py on it.

    python code/make_submission.py --rollno 24B4521            # writes ../24B4521_tts_bootcamp.zip

Only the files the problem statement asks for go in: code/, predictions/, idea.md, results.json,
report.md (if present). Scratch files (NOTES_FOR_YOU.md, __pycache__, .git, timing JSONs left
next to audio folders) are skipped.
"""
import argparse, subprocess, sys, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ["code", "predictions", "idea.md", "results.json", "report.md"]
SKIP_PARTS = {"__pycache__", ".git", ".ipynb_checkpoints"}


def files():
    for name in INCLUDE:
        p = ROOT / name
        if p.is_file():
            yield p
        elif p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file() and not (SKIP_PARTS & set(f.relative_to(ROOT).parts)):
                    if f.parent.name == "predictions" and f.suffix == ".json":
                        continue                       # timing files belong in code/
                    yield f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollno", required=True)
    ap.add_argument("--out-dir", default=str(ROOT.parent))
    a = ap.parse_args()
    out = Path(a.out_dir) / f"{a.rollno}_tts_bootcamp.zip"
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files():
            z.write(f, f.relative_to(ROOT).as_posix())
            n += 1
    print(f"wrote {out} ({n} files, {out.stat().st_size / 1e6:.1f} MB)")
    kit = None
    for p in [ROOT.parent, *ROOT.parents]:
        if (p / "check_format.py").exists():
            kit = p
            break
    if kit:
        r = subprocess.run([sys.executable, str(kit / "check_format.py"), str(out)])
        sys.exit(r.returncode)
    print("check_format.py not found next to this repository; run it yourself.")


if __name__ == "__main__":
    main()
