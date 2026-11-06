@echo off
call "%~dp0train_ppo.bat" 15
exit /b %ERRORLEVEL%
