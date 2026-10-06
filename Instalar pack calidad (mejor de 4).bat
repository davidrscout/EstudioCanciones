@echo off
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "app\instalar.ps1" -calidad
pause
