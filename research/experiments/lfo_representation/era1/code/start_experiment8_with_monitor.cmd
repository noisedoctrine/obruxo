@echo off
setlocal

set "ROOT=%~dp0"
set "BEAM=4"
set "REFRESH=30"
set "ALIGN=xpu"
set "CACHEEVERY=1"
set "BATCHSIZE="
set "TRAINSTAGEBATCHSIZE="
set "ALIGNBATCHSIZE="
set "MAXSHAPES="
set "SEED=7267"

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
if /I "%~1"=="--cache-every" (
  set "CACHEEVERY=%~2"
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
  set "TRAINSTAGEBATCHSIZE=--train-stage-batch-size %~2"
  shift
  shift
  goto parse
)
if /I "%~1"=="--align-batch-size" (
  set "ALIGNBATCHSIZE=--align-batch-size %~2"
  shift
  shift
  goto parse
)
if /I "%~1"=="--max-shapes" (
  set "MAXSHAPES=--max-shapes %~2"
  shift
  shift
  goto parse
)
if /I "%~1"=="--seed" (
  set "SEED=%~2"
  shift
  shift
  goto parse
)
echo Unknown argument: %~1
exit /b 2

:run
if exist "%ROOT%artifacts\additive_finalization_8_screen\RUNNING.lock" (
  echo Experiment 8 runner lock already exists. A previous runner may still be active.
  echo Opening the monitor without starting a second runner.
  start "Experiment 8 Monitor" powershell.exe -NoExit -ExecutionPolicy Bypass -File "%ROOT%monitor_experiment8.ps1" -RefreshSeconds %REFRESH%
  exit /b 0
)
start "Experiment 8 Runner" /min cmd.exe /c ""%ROOT%run_experiment8_background.cmd" --beam-width %BEAM% --align-device %ALIGN% --cache-every %CACHEEVERY% --seed %SEED% %BATCHSIZE% %TRAINSTAGEBATCHSIZE% %ALIGNBATCHSIZE% %MAXSHAPES%"
start "Experiment 8 Monitor" powershell.exe -NoExit -ExecutionPolicy Bypass -File "%ROOT%monitor_experiment8.ps1" -RefreshSeconds %REFRESH%

echo Started Experiment 8 screen with beam=%BEAM%, align-device=%ALIGN%, cache-every=%CACHEEVERY%, seed=%SEED%, monitor refresh=%REFRESH%s.
exit /b 0
