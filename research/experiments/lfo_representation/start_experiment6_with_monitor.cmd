@echo off
setlocal

set "ROOT=%~dp0"
set "BEAM=32"
set "PARALLEL=2"
set "REFRESH=5"
set "ALIGN=xpu"

:parse
if "%~1"=="" goto run
if /I "%~1"=="--beam-width" (
  set "BEAM=%~2"
  shift
  shift
  goto parse
)
if /I "%~1"=="--parallel" (
  set "PARALLEL=%~2"
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
echo Unknown argument: %~1
exit /b 2

:run
start "Experiment 6 Runner" /min cmd.exe /c ""%ROOT%run_experiment6_background.cmd" --beam-width %BEAM% --parallel %PARALLEL% --align-device %ALIGN%"
start "Experiment 6 Monitor" cmd.exe /c ""%ROOT%open_experiment6_monitor.cmd" -RefreshSeconds %REFRESH%"

echo Started Experiment 6 with beam=%BEAM%, parallel=%PARALLEL%, align-device=%ALIGN%, monitor refresh=%REFRESH%s.
exit /b 0
