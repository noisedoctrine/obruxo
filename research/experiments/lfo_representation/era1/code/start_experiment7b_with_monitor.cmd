@echo off
setlocal

set "ROOT=%~dp0"
set "BEAM=4"
set "REFRESH=30"
set "ALIGN=xpu"
set "CONFIG="
set "QUICK="
set "MAXSHAPES="

:parse
if "%~1"=="" goto validate
if /I "%~1"=="--config" (
  set "CONFIG=%~2"
  shift
  shift
  goto parse
)
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
echo Unknown argument: %~1
exit /b 2

:validate
if "%CONFIG%"=="" (
  echo Experiment 7B requires --config path\to\experiment7b_config.json
  exit /b 2
)

:run
start "Experiment 7B Runner" /min cmd.exe /c ""%ROOT%run_experiment7b_background.cmd" --config "%CONFIG%" --beam-width %BEAM% --align-device %ALIGN% %QUICK% %MAXSHAPES%"
start "Experiment 7B Monitor" powershell.exe -NoExit -ExecutionPolicy Bypass -File "%ROOT%monitor_experiment7.ps1" -Experiment 7B -RefreshSeconds %REFRESH%

echo Started Experiment 7B with config=%CONFIG%, beam=%BEAM%, align-device=%ALIGN%, monitor refresh=%REFRESH%s.
exit /b 0
