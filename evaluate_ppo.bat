@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" goto setup
".venv\Scripts\python.exe" -c "import torch, stable_baselines3, gymnasium" >nul 2>&1
if errorlevel 1 goto setup
echo [1] Evaluate normal escape (100 mazes per action mode)
echo [2] Evaluate return-and-charge practice (100 mazes per action mode)
choice /c 12 /n /m "Choose 1 or 2: "
set "EVAL_MODE="
if errorlevel 2 set "EVAL_MODE=--practice"
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
".venv\Scripts\python.exe" -m training.evaluate --cloud --episodes 100 --seed 20000 %EVAL_MODE%
:done

if errorlevel 1 echo FAILED: See the error above. Local backups are retained.
pause
exit /b
:setup
echo Run test_ppo.bat first to install and verify the Python environment.
pause
exit /b 1
