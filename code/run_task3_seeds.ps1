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
& $PY submission\code\seed_summary.py
