$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '../../../..')).Path
$formal = Join-Path $root 'tmp/issue187-formal'
$decisionPath = Join-Path $formal 'decisions.json'
$log = Join-Path $formal 'stage3-queue.log'
$py = Join-Path $env:USERPROFILE 'miniforge3/envs/py312/python.exe'
$selection = Join-Path $root 'research/training/reports/issue187_pretrained_vital_surrogates/STAGE1_SELECTION.json'
$qual = Join-Path $root 'tmp/issue187-selected-smoke/decisions.json'
$prior = Join-Path (Split-Path $root -Parent) 'issue187-pretrained-vital-surrogates/tmp/issue187-run'
$recovered = Join-Path $root 'tmp/issue187-audio-recovery'
$midi = Join-Path $root 'tmp/issue187-spansynth-midi-encoder.safetensors'
$runner = Join-Path $root 'research/training/reports/scripts/run_issue187_surrogates.py'
$upstream = (Resolve-Path (Join-Path (Split-Path $root -Parent) 'issue187-pretrained-vital-surrogates/tmp')).Path
$env:PYTHONPATH = "$upstream/stable-audio-tools-source;$upstream/stable-audio-3-source;$upstream/stable-audio-deps"
$env:HF_HUB_OFFLINE = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '4'

Add-Content -LiteralPath $log -Value 'Queued Stage 3 for every completed, qualified Stage 2 arm.'
while ($true) {
    if (Test-Path -LiteralPath $decisionPath) {
        $decision = Get-Content -LiteralPath $decisionPath -Raw | ConvertFrom-Json
        if ($decision.stage2 -eq 'analysis_required' -or
            ($decision.stage2 -eq 'complete' -and $decision.stage3 -eq 'running')) { break }
        if ($decision.stage2 -in @('incomplete', 'blocked_missing_verified_audio', 'failed')) {
            Add-Content -LiteralPath $log -Value "Stage 3 not started: Stage 2 ended as $($decision.stage2)."
            exit 0
        }
    }
    Start-Sleep -Seconds 10
}

$adapters = @($decision.completed_configurations | Sort-Object)
if ($adapters.Count -eq 0) {
    Add-Content -LiteralPath $log -Value 'Stage 3 not started: no candidate completed 8,000 updates.'
    exit 0
}
$stage3Selection = Join-Path $formal 'stage3-all-candidates.json'
$payload = @{ adapters = $adapters; selection = 'All completed, qualified Stage 2 configurations; no performance cutoff.' }
[System.IO.File]::WriteAllText($stage3Selection, ($payload | ConvertTo-Json -Depth 4), [System.Text.Encoding]::ASCII)
Add-Content -LiteralPath $log -Value "Starting Stage 3 for: $($adapters -join ', ')"
$shared = @('--output', $formal, '--selection', $selection, '--prior-run', $prior, '--recovered', $recovered,
            '--midi-checkpoint', $midi, '--qualification', $qual)
& $py -B $runner @shared --target-step 8000 --inversion-decision $stage3Selection *> (Join-Path $formal 'stage3-worker.log')
if ($LASTEXITCODE -ne 0) {
    Add-Content -LiteralPath $log -Value "Stage 3 runner failed with exit code $LASTEXITCODE."
    exit $LASTEXITCODE
}
Add-Content -LiteralPath $log -Value 'Stage 3 completed for all scheduled candidates.'
