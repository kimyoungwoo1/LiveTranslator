@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (echo Run install.bat first. & pause & exit /b 1)
.venv\Scripts\python.exe -m pip install pyinstaller
if errorlevel 1 (pause & exit /b 1)
.venv\Scripts\python.exe -m PyInstaller --noconfirm --clean --onedir --windowed --name LiveTranslator --collect-all faster_whisper --collect-all ctranslate2 --collect-all argostranslate main.py
if errorlevel 1 (pause & exit /b 1)
echo Built to dist\LiveTranslator
pause
