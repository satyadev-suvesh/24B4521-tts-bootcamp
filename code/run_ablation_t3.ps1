# Task 3 analysis (not part of the submitted system): script alone vs respelling rules vs the best
# possible respelling, on the 6 hinglish validation items, seeds 0-2, baseline decoding, T4 noise.
# Run after run_task3.ps1 and run_task3_seeds.ps1 (uses their val_control / val_idea folders).
#
#   powershell -ExecutionPolicy Bypass -File submission\code\run_ablation_t3.ps1
$KIT = (Resolve-Path "$PSScriptRoot\..\..").Path
$PY = "$KIT\.venv\Scripts\python.exe"
$env:TTS_KIT = $KIT
$env:PYTHONIOENCODING = "utf-8"
$SUB = "$KIT\submission"
$IDS = "val_hg_01,val_hg_02,val_hg_03,val_hg_04,val_hg_05,val_hg_06"

Set-Location $SUB
& $PY code\ablation_t3.py --out "$KIT\work\ablation_text.csv"
foreach ($col in "naive", "oracle") {
    foreach ($seed in 0, 1, 2) {
        $out = "$KIT\work\abl_${col}_s$seed"
        if (-not (Test-Path "$out\val_hg_06.wav")) {
            & $PY code\synth.py --items "$KIT\data\val_items.csv" --ids $IDS --text-csv "$KIT\work\ablation_text.csv" --text-column $col --out $out --noise t4 --seed $seed --peak 0.95
        }
        Set-Location $KIT
        & $PY metrics.py $out --json "$out.json" --csv "$out.csv" | Out-Null
        Set-Location $SUB
    }
}
# MMS as a second recognizer, item by item (folders hold only the 6 hinglish items; read those rows)
$folders = @("predictions\val_control", "$KIT\work\val_control_s1", "$KIT\work\val_control_s2",
             "$KIT\work\abl_naive_s0", "$KIT\work\abl_naive_s1", "$KIT\work\abl_naive_s2",
             "predictions\val_idea", "$KIT\work\val_idea_s1", "$KIT\work\val_idea_s2",
             "$KIT\work\abl_oracle_s0", "$KIT\work\abl_oracle_s1", "$KIT\work\abl_oracle_s2")
& $PY code\eval_extra.py @folders --csv "$KIT\work\ablation_scores.csv" | Out-Null
& $PY code\ablation_summary.py "$KIT\work\ablation_scores.csv"
Set-Location $KIT
& $PY submission\code\analyze_words.py latin=work\val_control.csv naive=work\abl_naive_s0.csv rules=work\val_idea.csv oracle=work\abl_oracle_s0.csv
