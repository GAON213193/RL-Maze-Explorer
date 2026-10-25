@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" goto setup
".venv\Scripts\python.exe" -c "import torch, stable_baselines3, gymnasium" >nul 2>&1
if errorlevel 1 goto setup
echo [1] Download latest GitHub model and replay
echo [2] Replay local backup ppo_model.zip
choice /c 12 /n /m "Choose 1 or 2: "
if errorlevel 2 goto local
where gh >nul 2>&1
if errorlevel 1 (
 echo Install GitHub CLI from https://cli.github.com and reopen this file.
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
".venv\Scripts\python.exe" -m training.viewer
goto done
:local
".venv\Scripts\python.exe" -m training.viewer --local
:done

if errorlevel 1 echo FAILED: See the error above. Local backups are retained.
pause
exit /b
:setup
echo Run test_ppo.bat first to install and verify the Python environment.
pause
exit /b 1
