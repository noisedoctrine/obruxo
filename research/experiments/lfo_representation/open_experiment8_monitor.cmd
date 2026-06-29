@echo off
setlocal

set "ROOT=%~dp0"
powershell.exe -NoProfile -NoExit -ExecutionPolicy Bypass -File "%ROOT%monitor_experiment8.ps1" %*
