@echo off
cd /d "%~dp0"
py -3 -m environment.demo %*
if errorlevel 1 echo Python 3.10+ with Tcl/Tk is required. Try: python -m environment.demo
pause
