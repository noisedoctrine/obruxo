@echo off
setlocal

set "ROOT=%~dp0"
set "OUT=%ROOT%artifacts\additive_finalization_8_screen"
set "LOCK=%OUT%\RUNNING.lock"
if not exist "%OUT%" mkdir "%OUT%"
mkdir "%LOCK%" 2>nul
if errorlevel 1 (
  echo Experiment 8 background runner lock already exists at "%LOCK%".
  echo No new runner was started. Open the monitor to check whether the previous runner is active.
  echo If the monitor shows no active runner, remove that lock directory and retry.
  exit /b 3
)
if exist "%OUT%\COMPLETED_EXPERIMENT_8.txt" del /f /q "%OUT%\COMPLETED_EXPERIMENT_8.txt"

set "MKL_THREADING_LAYER=SEQUENTIAL"
set "MPLCONFIGDIR=%ROOT%artifacts\mpl"

cd /d "%ROOT%"
echo Experiment 8 background runner started at %DATE% %TIME% > "%OUT%\experiment8_background_stdout.log"
echo ROOT=%ROOT% >> "%OUT%\experiment8_background_stdout.log"
conda run -n py312 python -u .\experiment8.py run %* >> "%OUT%\experiment8_background_stdout.log" 2> "%OUT%\experiment8_background_stderr.log"
set "STATUS=%ERRORLEVEL%"
echo Experiment 8 background runner exited at %DATE% %TIME% with status %STATUS% >> "%OUT%\experiment8_background_stdout.log"
rmdir "%LOCK%" 2>nul
exit /b %STATUS%
