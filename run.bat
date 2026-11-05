@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (echo First run install.bat & pause & exit /b 1)
if not exist "main.py" (echo main.py has not been added to this repository yet. & pause & exit /b 1)
.venv\Scripts\python.exe main.py
if errorlevel 1 pause
