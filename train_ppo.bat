@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" goto setup
".venv\Scripts\python.exe" -c "import torch, stable_baselines3, gymnasium" >nul 2>&1
if errorlevel 1 goto setup
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
echo [1] Continue latest GitHub model (or start new if none exists)
echo [2] Start a NEW independent model
echo Models upload to GAON213193/RL-Maze-Explorer Releases.
echo This PC performs the training. Local backups stay in runs.
choice /c 12 /n /m "Choose 1 or 2: "
if errorlevel 2 goto new
".venv\Scripts\python.exe" -m training.train --cloud --auto-resume --steps 100000 --save-every 10000 --model-path ppo_model.zip
goto done
:new
".venv\Scripts\python.exe" -m training.train --cloud --steps 100000 --save-every 10000 --model-path ppo_model.zip
:done

if errorlevel 1 echo FAILED: See the error above. Local backups are retained.
pause
exit /b
:setup
echo Run test_ppo.bat first to install and verify the Python environment.
pause
exit /b 1
