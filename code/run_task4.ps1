# Task 4: time the baseline and the fast Task 3 system on the same 24 validation items, in the
# same session on the same machine (model loaded, one warm-up excluded), then make t4_fast/.
# Run after run_task3.ps1 (needs code\t3_text.csv).
#
#   powershell -ExecutionPolicy Bypass -File submission\code\run_task4.ps1 [-Nfe 12]
param([int]$Nfe = 12)
$KIT = (Resolve-Path "$PSScriptRoot\..\..").Path
$PY = "$KIT\.venv\Scripts\python.exe"
$env:TTS_KIT = $KIT
$env:PYTHONIOENCODING = "utf-8"
$SUB = "$KIT\submission"

function Run($what, [scriptblock]$cmd) {
    Write-Host "`n=== $what" -ForegroundColor Cyan
    & $cmd
    if ($LASTEXITCODE -ne 0) { throw "failed: $what (exit $LASTEXITCODE)" }
}

Set-Location $KIT
# 1. official baseline timing (kit script, unchanged); its audio is not used
Run "baseline timing (baseline/run_baseline.py)" { & $PY baseline\run_baseline.py --items data\val_items.csv --out work\t4_baseline_val --seed 0 }
Copy-Item work\t4_baseline_val_timing.json "$SUB\code\timing\baseline_val_timing.json" -Force

# 2. fast Task 3 system on the same items, same session
Set-Location $SUB
Run "fast system timing (code/fast.py)" { & $PY code\fast.py --items "$KIT\data\val_items.csv" --text-csv code\t3_text.csv --text-column t3_text --out "$KIT\work\val_fast" --nfe $Nfe --amp bf16 --noise t4 --seed 0 --peak 0.95 }
Copy-Item "$KIT\work\val_fast_timing.json" code\timing\val_fast_timing.json -Force

# 3. quality of the fast system on validation
Set-Location $KIT
Run "metrics.py on val_fast" { & $PY metrics.py work\val_fast --json work\val_fast.json --csv work\val_fast.csv }

# 4. evaluation audio
Set-Location $SUB
Run "t4_fast (40 evaluation items)" { & $PY code\fast.py --items "$KIT\data\eval_items.csv" --text-csv code\t3_text.csv --text-column t3_text --out predictions\t4_fast --nfe $Nfe --amp bf16 --noise t4 --seed 0 --peak 0.95 }
Move-Item predictions\*_timing.json code\timing\ -Force
Run "sanity of t4_fast" { & $PY code\sanity_eval.py predictions\t4_fast --text-csv code\t3_text.csv --text-column t3_text }

# 5. results.json
Set-Location $KIT
Run "results.json" { & $PY submission\code\update_results.py --task4 work\val_fast.json --baseline-timing submission\code\timing\baseline_val_timing.json --fast-timing submission\code\timing\val_fast_timing.json --method "Same model, faster execution: batched CFG, cached text embedding, bf16 autocast, $Nfe flow steps" }
Write-Host "`nDone. Check that task4.val_cer <= 1.2 x task3.val_cer, then rebuild the zip." -ForegroundColor Green
