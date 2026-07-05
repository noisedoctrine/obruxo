param(
    [Parameter(Mandatory = $true)]
    [string]$RunDir,
    [int]$RefreshSeconds = 10,
    [int]$EventTail = 6,
    [string]$PythonExe = "python"
)

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$StatusScript = Join-Path $Root "run_era2.py"
$Status = Join-Path $RunDir "run_status.json"
$Events = Join-Path $RunDir "events.jsonl"
$Host.UI.RawUI.WindowTitle = "LFO Era 2 Monitor"

while ($true) {
    Clear-Host
    Write-Host "LFO Era 2 Monitor" -ForegroundColor Cyan
    Write-Host ("Updated: " + (Get-Date).ToString("yyyy-MM-dd HH:mm:ss"))
    Write-Host ("RunDir: " + $RunDir)
    Write-Host ("".PadLeft(72, "-")) -ForegroundColor DarkGray
    Write-Host ""

    if (Test-Path $Status) {
        try {
            & $PythonExe $StatusScript status --run-dir $RunDir
        } catch {
            Write-Host "Status command failed:" -ForegroundColor Red
            Write-Host $_
        }
    } elseif (Test-Path $RunDir) {
        Write-Host "Waiting for run status to be written..."
    } else {
        Write-Host "Waiting for run directory to be created..."
    }

    Write-Host ""
    Write-Host ("".PadLeft(72, "-")) -ForegroundColor DarkGray
    Write-Host "Recent events" -ForegroundColor DarkCyan
    if (Test-Path $Events) {
        Get-Content $Events -Tail $EventTail
    } else {
        Write-Host "No events yet."
    }

    Start-Sleep -Seconds ([Math]::Max(1, $RefreshSeconds))
}
