@echo off
setlocal
cd /d "%~dp0"
title RL Maze Explorer - First-time Setup
echo Install Python 3.12, CPU learning packages and GitHub CLI.
echo Internet is required. Existing compatible software is reused.
echo Extract the complete project ZIP before running this file.
echo.
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup_windows.ps1"
set "SETUP_RESULT=%ERRORLEVEL%"
echo.
if "%SETUP_RESULT%"=="0" echo Ready. Run train_ppo.bat and choose 1 to continue your GitHub model.
if "%SETUP_RESULT%"=="2" echo Software is ready. Finish GitHub login when you run train_ppo.bat.
if "%SETUP_RESULT%"=="1" echo Setup failed. See the message above and setup_windows.log.
pause
exit /b %SETUP_RESULT%
