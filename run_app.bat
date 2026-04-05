@echo off
setlocal
cd /d "%~dp0"
echo [1] Directory: %cd%
if not exist "venv\Scripts\python.exe" (
    echo [ERROR] Cannot find: %~dp0venv\Scripts\python.exe
    pause
    exit /b 1
)
echo [2] Using venv: venv\Scripts\python.exe
echo [3] Running application...
venv\Scripts\python.exe src/app.py
pause
endlocal
