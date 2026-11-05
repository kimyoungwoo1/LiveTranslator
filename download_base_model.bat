@echo off
setlocal
set "MODEL_DIR=%LOCALAPPDATA%\LiveTranslator\models\base"
if not exist "%MODEL_DIR%" mkdir "%MODEL_DIR%"
echo Downloading Whisper base model to "%MODEL_DIR%"
for %%F in (config.json tokenizer.json vocabulary.txt model.bin) do (
  echo Downloading %%F ...
  curl.exe --fail --location --retry 3 --connect-timeout 20 --output "%MODEL_DIR%\%%F" "https://huggingface.co/Systran/faster-whisper-base/resolve/main/%%F"
  if errorlevel 1 goto fail
)
powershell -NoProfile -Command "$p=Join-Path $env:LOCALAPPDATA 'LiveTranslator\models\base\model.bin'; if ((Get-Item $p).Length -ne 145217532) { exit 1 }"
if errorlevel 1 goto fail
echo [OK] Whisper base model downloaded and size verified.
echo Restart LiveTranslator to load the local model.
pause
exit /b 0
:fail
echo [ERROR] Download failed or file size mismatch. Files were not deleted; retry is safe.
pause
exit /b 1
