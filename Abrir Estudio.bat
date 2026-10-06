@echo off
cd /d "%~dp0"
set CANCIONES_HOME=%~dp0
".venv\Scripts\python.exe" "app\server.py"
pause
