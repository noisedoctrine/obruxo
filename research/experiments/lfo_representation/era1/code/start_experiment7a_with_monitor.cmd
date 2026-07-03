@echo off
setlocal

set "ROOT=%~dp0"
set "BEAM=32"
set "REFRESH=30"
set "ALIGN=xpu"
set "QUICK="
set "MAXSHAPES="
set "BATCHSIZE="
set "TRAINSTAGEBATCH="
set "ALIGNBATCH="
set "CACHEEVERY="

:parse
if "%~1"=="" goto run
if /I "%~1"=="--beam-width" (
  set "BEAM=%~2"
  shift
  shift
  goto parse
)
if /I "%~1"=="--refresh-seconds" (
  set "REFRESH=%~2"
  shift
  shift
  goto parse
)
if /I "%~1"=="--align-device" (
  set "ALIGN=%~2"
  shift
  shift
  goto parse
)
if /I "%~1"=="--quick" (
  set "QUICK=--quick"
  shift
  goto parse
)
if /I "%~1"=="--max-shapes" (
  set "MAXSHAPES=--max-shapes %~2"
  shift
  shift
  goto parse
)
if /I "%~1"=="--batch-size" (
  set "BATCHSIZE=--batch-size %~2"
  shift
  shift
  goto parse
)
if /I "%~1"=="--train-stage-batch-size" (
  set "TRAINSTAGEBATCH=--train-stage-batch-size %~2"
  shift
  shift
  goto parse
)
if /I "%~1"=="--align-batch-size" (
  set "ALIGNBATCH=--align-batch-size %~2"
  shift
  shift
  goto parse
)
if /I "%~1"=="--cache-every" (
  set "CACHEEVERY=--cache-every %~2"
  shift
  shift
  goto parse
)
echo Unknown argument: %~1
exit /b 2

:run
start "Experiment 7A Runner" /min cmd.exe /c ""%ROOT%run_experiment7a_background.cmd" --beam-width %BEAM% --align-device %ALIGN% %QUICK% %MAXSHAPES% %BATCHSIZE% %TRAINSTAGEBATCH% %ALIGNBATCH% %CACHEEVERY%"
start "Experiment 7A Monitor" powershell.exe -NoExit -ExecutionPolicy Bypass -File "%ROOT%monitor_experiment7.ps1" -Experiment 7A -RefreshSeconds %REFRESH%

echo Started Experiment 7A with beam=%BEAM%, align-device=%ALIGN%, batch-size=%BATCHSIZE%, train-stage-batch-size=%TRAINSTAGEBATCH%, align-batch-size=%ALIGNBATCH%, cache-every=%CACHEEVERY%, monitor refresh=%REFRESH%s.
exit /b 0
