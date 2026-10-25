@echo off
setlocal
title RL Maze Explorer - PPO Test
cd /d "%~dp0"
if errorlevel 1 goto failed

echo ============================================================
echo RL Maze Explorer - one-click PPO test (CPU)
echo First run downloads packages and may take several minutes.
echo Python 3.12 and internet access are required for initial setup.
echo ============================================================
echo.

if not exist "training\train.py" goto missing_project
if not exist "requirements-rl.txt" goto missing_project

if exist ".venv\Scripts\python.exe" goto check_python
py -3.12 -c "import sys; assert sys.version_info[:2] == (3, 12)" >nul 2>&1
if errorlevel 1 goto try_python
py -3.12 -m venv ".venv"
if errorlevel 1 goto failed
goto check_python

:try_python
python -c "import sys; assert sys.version_info[:2] == (3, 12)" >nul 2>&1
if errorlevel 1 goto missing_python
python -m venv ".venv"
if errorlevel 1 goto failed

:check_python
".venv\Scripts\python.exe" -c "import sys; assert sys.version_info[:2] == (3, 12), 'This test requires Python 3.12'"
if errorlevel 1 goto wrong_venv

echo [1/3] Checking dependencies...
".venv\Scripts\python.exe" -c "from importlib.metadata import version; import torch, gymnasium, stable_baselines3, numpy; assert version('stable-baselines3') == '2.9.0'; assert version('gymnasium') == '1.4.0'; assert version('numpy') == '2.5.4'; assert version('torch').split('+')[0] == '2.14.1'" >nul 2>&1
if not errorlevel 1 goto run_tests
".venv\Scripts\python.exe" -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install -r requirements-rl.txt
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip check
if errorlevel 1 goto failed

:run_tests
echo.
echo [2/3] Testing environment, learning, save/load and resume...
".venv\Scripts\python.exe" -m unittest discover -s tests -v
if errorlevel 1 goto failed

echo.
echo [3/3] Running a short 64x64 PPO training job...
".venv\Scripts\python.exe" -m training.train --size 64 --steps 256 --max-turns 128 --device cpu --model-path ppo_test_model.zip
if errorlevel 1 goto failed
echo.
echo SUCCESS: Tests and short training completed.
echo Model: %CD%\ppo_test_model.zip
echo Logs and resume configuration: see the Saved path above in runs.
echo This verifies training works; it does not prove maze-solving performance.
echo.
pause
exit /b 0

:missing_python
echo.
echo Python 3.12 was not found.
echo Install 64-bit Python 3.12 with Python Launcher from python.org,
echo then double-click this file again.
goto failed

:wrong_venv
echo.
echo Existing .venv is not a working Python 3.12 environment.
echo Rename the existing .venv folder to keep it, then run this file again.
goto failed

:missing_project
echo.
echo Project files are missing. Extract the FULL project ZIP first.
echo Keep this BAT file alongside requirements-rl.txt and training.
goto failed

:failed
echo.
echo FAILED: Read the error above. No existing results were deleted.
echo If a download failed, check your internet connection and run again.
echo.
pause
exit /b 1
