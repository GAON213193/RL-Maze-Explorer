@echo off
setlocal
cd /d "%~dp0"
if exist "%~dp0.tools\gh\bin\gh.exe" set "PATH=%~dp0.tools\gh\bin;%PATH%"
if not exist ".venv\Scripts\python.exe" goto setup
".venv\Scripts\python.exe" -c "import torch, stable_baselines3, gymnasium" >nul 2>&1
if errorlevel 1 goto setup
where gh >nul 2>&1
if errorlevel 1 (
 echo Run setup_windows.bat to prepare GitHub CLI, then reopen this file.
 pause
 exit /b 1
)
gh auth status --hostname github.com >nul 2>&1
if errorlevel 1 gh auth login --hostname github.com --git-protocol https --web
if errorlevel 1 (
 echo GitHub login failed.
 pause
 exit /b 1
)
".venv\Scripts\python.exe" -m training.cloud sync

if errorlevel 1 echo FAILED: See the error above. Local backups are retained.
pause
exit /b
:setup
echo Run setup_windows.bat first to install and verify the Python environment.
pause
exit /b 1
