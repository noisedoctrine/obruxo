@echo off
setlocal

set "ROOT=%~dp0"
set "OUT=%ROOT%artifacts\phase_alignment_oracle"
if not exist "%OUT%" mkdir "%OUT%"
if exist "%OUT%\COMPLETED_EXCPERIMENT_5.txt" del /f /q "%OUT%\COMPLETED_EXCPERIMENT_5.txt"

set "MKL_THREADING_LAYER=SEQUENTIAL"
set "MPLCONFIGDIR=%ROOT%artifacts\mpl"

cd /d "%ROOT%"
echo Experiment 5 background runner started at %DATE% %TIME% > "%OUT%\experiment5_background_stdout.log"
echo ROOT=%ROOT% >> "%OUT%\experiment5_background_stdout.log"
python -u -m lfo_experiment.experiment5_worker "%ROOT%artifacts\lfo_catalog.csv" "%ROOT%artifacts\phase_factorized_residual" "%OUT%" %* >> "%OUT%\experiment5_background_stdout.log" 2> "%OUT%\experiment5_background_stderr.log"
set "STATUS=%ERRORLEVEL%"
echo Experiment 5 background runner exited at %DATE% %TIME% with status %STATUS% >> "%OUT%\experiment5_background_stdout.log"
exit /b %STATUS%
