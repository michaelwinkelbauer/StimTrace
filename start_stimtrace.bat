@echo off
setlocal
cd /d "%~dp0"

set "PYTHON=%USERPROFILE%\emt-env\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=%~dp0.venv\Scripts\python.exe"

if not exist "%PYTHON%" (
  echo StimTrace Python environment was not found.
  echo Create .venv or update PYTHON in start_stimtrace.bat.
  pause
  exit /b 1
)

"%PYTHON%" desktop_app.py
if errorlevel 1 pause
