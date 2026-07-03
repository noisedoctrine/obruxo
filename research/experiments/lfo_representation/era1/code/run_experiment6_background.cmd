@echo off
setlocal

set "ROOT=%~dp0"
set "OUT=%ROOT%artifacts\codebook_selection"
if not exist "%OUT%" mkdir "%OUT%"
if exist "%OUT%\COMPLETED_EXCPERIMENT_6.txt" del /f /q "%OUT%\COMPLETED_EXCPERIMENT_6.txt"

set "MKL_THREADING_LAYER=SEQUENTIAL"
set "MPLCONFIGDIR=%ROOT%artifacts\mpl"

cd /d "%ROOT%"
echo Experiment 6 background runner started at %DATE% %TIME% > "%OUT%\experiment6_background_stdout.log"
echo ROOT=%ROOT% >> "%OUT%\experiment6_background_stdout.log"
python -u -m lfo_experiment.experiment6_worker "%ROOT%artifacts\lfo_catalog.csv" "%ROOT%artifacts\stock_codebook.json" "%ROOT%artifacts\phase_factorized_residual" "%OUT%" %* >> "%OUT%\experiment6_background_stdout.log" 2> "%OUT%\experiment6_background_stderr.log"
set "STATUS=%ERRORLEVEL%"
echo Experiment 6 background runner exited at %DATE% %TIME% with status %STATUS% >> "%OUT%\experiment6_background_stdout.log"
exit /b %STATUS%

