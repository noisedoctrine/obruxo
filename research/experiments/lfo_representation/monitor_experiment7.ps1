param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("7A", "7B")]
    [string]$Experiment,
    [int]$RefreshSeconds = 10,
    [int]$LogTail = 12
)

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = "python"
if ($Experiment -eq "7A") {
    $Command = "experiment7a_status"
    $OutDir = Join-Path $Root "artifacts\additive_finalization_7a"
} else {
    $Command = "experiment7b_status"
    $OutDir = Join-Path $Root "artifacts\additive_finalization_7b"
}
$Stdout = Join-Path $OutDir "experiment$($Experiment)_background_stdout.log"
$Stderr = Join-Path $OutDir "experiment$($Experiment)_background_stderr.log"
$Done = Join-Path $OutDir "COMPLETED_EXPERIMENT_$($Experiment).txt"
$env:MPLCONFIGDIR = Join-Path $Root "artifacts\mpl"

$Host.UI.RawUI.WindowTitle = "Experiment $Experiment Monitor"

while ($true) {
    Clear-Host
    Write-Host "Experiment $Experiment Monitor" -ForegroundColor Cyan
    Write-Host ("Updated: " + (Get-Date).ToString("yyyy-MM-dd HH:mm:ss"))
    Write-Host ""

    Push-Location $Root
    try {
        & $Python run_experiment.py $Command
    } catch {
        Write-Host "Status command failed:" -ForegroundColor Red
        Write-Host $_
    } finally {
        Pop-Location
    }

    Write-Host ""
    Write-Host "Recent stdout" -ForegroundColor DarkCyan
    if (Test-Path $Stdout) {
        Get-Content $Stdout -Tail $LogTail
    } else {
        Write-Host "No stdout log yet."
    }

    Write-Host ""
    Write-Host "Recent stderr" -ForegroundColor DarkYellow
    if (Test-Path $Stderr) {
        $stderrLines = Get-Content $Stderr -Tail $LogTail
        if ($stderrLines) {
            $stderrLines
        } else {
            Write-Host "(empty)"
        }
    } else {
        Write-Host "No stderr log yet."
    }

    if (Test-Path $Done) {
        Write-Host ""
        Write-Host "Experiment $Experiment completed:" -ForegroundColor Green
        Get-Content $Done
        Write-Host ""
        Write-Host "Press Ctrl+C or close this window."
    }

    Start-Sleep -Seconds $RefreshSeconds
}
