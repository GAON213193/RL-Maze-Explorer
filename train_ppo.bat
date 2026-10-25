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
echo [1] Continue latest v3 GitHub model (or start new if none exists)
echo [2] Start a NEW independent v3 model (64x64)
echo [3] Continue v3 at a different maze size (after evaluation)
echo Models upload to GAON213193/RL-Maze-Explorer Releases.
echo This PC performs the training. Local backups stay in runs.
choice /c 123 /n /m "Choose 1, 2 or 3: "
if errorlevel 3 goto stage
if errorlevel 2 goto new
".venv\Scripts\python.exe" -m training.train --cloud --auto-resume --steps 100000 --save-every 10000 --model-path ppo_v3_model.zip
goto done
:new
".venv\Scripts\python.exe" -m training.train --cloud --steps 100000 --save-every 10000 --model-path ppo_v3_model.zip
goto done
:stage
echo [1] 7x7  [2] 11x11  [3] 17x17  [4] 33x33  [5] 64x64
choice /c 12345 /n /m "Select maze size: "
set "MAZE_SIZE=7"
if errorlevel 2 set "MAZE_SIZE=11"
if errorlevel 3 set "MAZE_SIZE=17"
if errorlevel 4 set "MAZE_SIZE=33"
if errorlevel 5 set "MAZE_SIZE=64"
".venv\Scripts\python.exe" -m training.train --cloud --auto-resume --size %MAZE_SIZE% --steps 100000 --save-every 10000 --model-path ppo_v3_model.zip
:done

if errorlevel 1 echo FAILED: See the error above. Local backups are retained.
pause
exit /b
:setup
echo Run test_ppo.bat first to install and verify the Python environment.
pause
exit /b 1
