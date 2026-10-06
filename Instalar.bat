@echo off
cd /d "%~dp0"
echo Instalando el Estudio de canciones...
powershell -NoProfile -ExecutionPolicy Bypass -File "app\instalar.ps1" %*
if errorlevel 1 (echo. & echo Algo ha fallado: mira el mensaje de arriba.)
pause
