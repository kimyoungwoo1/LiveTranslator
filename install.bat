@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>&1
if errorlevel 1 goto no_python
py -3.13 --version >nul 2>&1
if not errorlevel 1 set "PY=-3.13" & goto setup
py -3.12 --version >nul 2>&1
if not errorlevel 1 set "PY=-3.12" & goto setup
:no_python
echo Python 3.12 or 3.13 is required.
pause
exit /b 1
:setup
if not exist ".venv\Scripts\python.exe" py %PY% -m venv .venv
if errorlevel 1 goto failed
.venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 goto failed
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto failed
echo [OK] Dependencies installed.
pause
exit /b 0
:failed
echo [ERROR] Installation failed. Review the messages above.
pause
exit /b 1
