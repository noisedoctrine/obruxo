@echo off
setlocal

set "ROOT=%~dp0"
set "OUT=%ROOT%artifacts\additive_finalization_7a"
if not exist "%OUT%" mkdir "%OUT%"
if exist "%OUT%\COMPLETED_EXPERIMENT_7A.txt" del /f /q "%OUT%\COMPLETED_EXPERIMENT_7A.txt"

set "MKL_THREADING_LAYER=SEQUENTIAL"
set "MPLCONFIGDIR=%ROOT%artifacts\mpl"

cd /d "%ROOT%"
echo Experiment 7A background runner started at %DATE% %TIME% > "%OUT%\experiment7A_background_stdout.log"
echo ROOT=%ROOT% >> "%OUT%\experiment7A_background_stdout.log"
python -u -m lfo_experiment.experiment7_worker 7A "%ROOT%artifacts\lfo_catalog.csv" "%ROOT%artifacts\stock_codebook.json" "%OUT%" %* >> "%OUT%\experiment7A_background_stdout.log" 2> "%OUT%\experiment7A_background_stderr.log"
set "STATUS=%ERRORLEVEL%"
echo Experiment 7A background runner exited at %DATE% %TIME% with status %STATUS% >> "%OUT%\experiment7A_background_stdout.log"
exit /b %STATUS%
