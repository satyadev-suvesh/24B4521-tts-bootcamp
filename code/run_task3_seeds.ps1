# Robustness for Task 3: run control (Task 2 text) and idea (Task 3 text) at seeds 1 and 2 too,
# so the idea's effect is compared over 3 noise draws instead of one. Run after run_task3.ps1.
#
#   powershell -ExecutionPolicy Bypass -File submission\code\run_task3_seeds.ps1
$KIT = (Resolve-Path "$PSScriptRoot\..\..").Path
$PY = "$KIT\.venv\Scripts\python.exe"
$env:TTS_KIT = $KIT
$env:PYTHONIOENCODING = "utf-8"
$SUB = "$KIT\submission"

foreach ($seed in 1, 2) {
    foreach ($sys in @(@{name = "val_control"; csv = "predictions\t2_normalized.csv"; col = "normalized_text"},
                       @{name = "val_idea"; csv = "code\t3_text.csv"; col = "t3_text"})) {
        $out = "$KIT\work\$($sys.name)_s$seed"
        if (-not (Test-Path "$KIT\work\$($sys.name)_s$seed.json")) {
            Set-Location $SUB
            & $PY code\synth.py --items "$KIT\data\val_items.csv" --text-csv $sys.csv --text-column $sys.col --out $out --noise t4 --seed $seed --peak 0.95
            Set-Location $KIT
            & $PY metrics.py $out --json "$out.json" --csv "$out.csv"
        }
    }
}
Set-Location $KIT
& $PY - @'
import json
rows = []
for name in ["val_control", "val_idea"]:
    for s in [0, 1, 2]:
        p = f"work/{name}.json" if s == 0 else f"work/{name}_s{s}.json"
        d = json.load(open(p, encoding="utf-8"))
        rows.append((name, s, d["cer"], d["cer_by_category"]["hinglish"], d["sim"]))
for r in rows:
    print(f"{r[0]:12s} seed {r[1]}: CER {r[2]:.4f}  hinglish {r[3]:.4f}  SIM {r[4]:.4f}")
for name in ["val_control", "val_idea"]:
    h = [r[3] for r in rows if r[0] == name]
    print(f"{name:12s} mean hinglish CER {sum(h)/3:.4f}")
'@
