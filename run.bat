@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
"C:\ProgramData\anaconda3\envs\ai_lab\python.exe" main.py
echo.
echo Done. Press any key to close this window.
pause >nul