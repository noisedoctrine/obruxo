param(
    [int]$RefreshSeconds = 10,
    [int]$LogTail = 12
)

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$OutDir = Join-Path $Root "artifacts\additive_finalization_9_screen"
$Stdout = Join-Path $OutDir "experiment9_background_stdout.log"
$Stderr = Join-Path $OutDir "experiment9_background_stderr.log"
$Done = Join-Path $OutDir "COMPLETED_EXPERIMENT_9.txt"
$env:MPLCONFIGDIR = Join-Path $Root "artifacts\mpl"

$Host.UI.RawUI.WindowTitle = "Experiment 9 Monitor"

while ($true) {
    Clear-Host
    Write-Host "Experiment 9 Monitor" -ForegroundColor Cyan
    Write-Host ("Updated: " + (Get-Date).ToString("yyyy-MM-dd HH:mm:ss"))
    Write-Host ""

    Push-Location $Root
    try {
        conda run -n py312 python .\experiment9.py status
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
        Write-Host "Experiment 9 completed:" -ForegroundColor Green
        Get-Content $Done
        Write-Host ""
        Write-Host "Press Ctrl+C or close this window."
    }

    Start-Sleep -Seconds $RefreshSeconds
}
