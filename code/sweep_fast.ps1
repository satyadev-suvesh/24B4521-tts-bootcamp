# Step-count sweep of code/fast.py on the 24 validation items (Task 2 text), scored with metrics.py.
#   powershell -ExecutionPolicy Bypass -File submission\code\sweep_fast.ps1
$KIT = (Resolve-Path "$PSScriptRoot\..\..").Path
$PY = "$KIT\.venv\Scripts\python.exe"
$env:TTS_KIT = $KIT
$env:PYTHONIOENCODING = "utf-8"
foreach ($nfe in 32, 16, 12, 8) {
    $name = "sweep_bf16_$nfe"
    Set-Location "$KIT\submission"
    & $PY code\fast.py --items "$KIT\data\val_items.csv" --text-csv predictions\t2_normalized.csv --text-column normalized_text --out "$KIT\work\$name" --nfe $nfe --amp bf16 2>$null | Select-String -Pattern "Done" | Out-File -Append -Encoding utf8 "$KIT\work\sweep.log"
    Set-Location $KIT
    & $PY metrics.py "work\$name" --json "work\$name.json" --csv "work\$name.csv" 2>$null | Out-Null
    $s = Get-Content "work\$name.json" -Raw | ConvertFrom-Json
    $t = Get-Content "work\${name}_timing.json" -Raw | ConvertFrom-Json
    "$name : time $($t.total_synth_s)s | CER $($s.cer) | SIM $($s.sim) | hard $($s.cer_by_category.hard) hindi $($s.cer_by_category.hindi) hinglish $($s.cer_by_category.hinglish) long $($s.cer_by_category.long) | failed $($s.failed)" | Out-File -Append -Encoding utf8 "$KIT\work\sweep.log"
}
"SWEEP DONE" | Out-File -Append -Encoding utf8 "$KIT\work\sweep.log"
