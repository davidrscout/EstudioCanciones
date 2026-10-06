# Instalador del Estudio de canciones. Todo queda dentro de esta carpeta (no toca el sistema).
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Tools = Join-Path $Root "herramientas"
$Venv = Join-Path $Root ".venv"
$Models = Join-Path $Root "modelos"
$HfRepo = if ($env:CANCIONES_REPO) { $env:CANCIONES_REPO } else { "__REPO_HF__" }
New-Item -ItemType Directory -Force $Tools, $Models | Out-Null
$env:UV_PYTHON_INSTALL_DIR = Join-Path $Tools "python"
$env:UV_CACHE_DIR = Join-Path $Tools "cache"
$env:HF_HOME = Join-Path $Tools "hf"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

function Paso($t) { Write-Host ""; Write-Host "==> $t" -ForegroundColor Cyan }

Paso "Comprobando la tarjeta grafica"
$gpu = $null
try { $gpu = & nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>$null } catch {}
if ($gpu) { Write-Host "  $gpu" } else {
    Write-Host "No encuentro una tarjeta NVIDIA (o falta el driver)." -ForegroundColor Yellow
    Write-Host "Se instalara en MODO CPU: funciona, pero tarda unos 5 minutos por cada minuto de cancion y necesita 16 GB de RAM."
    Write-Host "Si tienes NVIDIA, actualiza el driver (https://www.nvidia.com/drivers) y vuelve a ejecutar el instalador."
}

Paso "Descargando uv (gestor de Python portable)"
$uv = Join-Path $Tools "uv.exe"
if (-not (Test-Path $uv)) {
    $zip = Join-Path $Tools "uv.zip"
    Invoke-WebRequest "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip" -OutFile $zip
    Expand-Archive $zip -DestinationPath $Tools -Force
    Remove-Item $zip
}

Paso "Preparando Python 3.12"
if (-not (Test-Path (Join-Path $Venv "Scripts\python.exe"))) { & $uv venv --python 3.12 $Venv }
$py = Join-Path $Venv "Scripts\python.exe"

if ($gpu) {
    Paso "Instalando PyTorch con CUDA (unos 3 GB, la parte mas larga)"
    & $uv pip install -p $py torch --index-url https://download.pytorch.org/whl/cu128
} else {
    Paso "Instalando PyTorch para CPU"
    & $uv pip install -p $py torch --index-url https://download.pytorch.org/whl/cpu
}
if ($LASTEXITCODE) { throw "Fallo instalando PyTorch" }

Paso "Instalando el resto"
& $uv pip install -p $py -r (Join-Path $Root "app\requirements.txt")
if ($LASTEXITCODE) { throw "Fallo instalando dependencias" }
if ($args -contains "-calidad") {
    & $uv pip install -p $py -r (Join-Path $Root "app\requirements-calidad.txt")
    if ($LASTEXITCODE) { throw "Fallo instalando el pack de calidad" }
}

Paso "Descargando los modelos (unos 7 GB; si se corta, vuelve a ejecutar y continua)"
if ($HfRepo -ne "__REPO_HF__") { $env:CANCIONES_REPO = $HfRepo }
$extra = @(); if ($args -contains "-calidad") { $extra += "--calidad" }
& $py (Join-Path $Root "app\descargar.py") $Models (Join-Path $Tools "descarga") @extra
if ($LASTEXITCODE) { throw "Fallo descargando los modelos" }

Paso "Limpiando temporales de la instalacion"
& $uv cache clean | Out-Null

Write-Host ""
Write-Host "Instalacion terminada. Abre 'Abrir Estudio.bat'." -ForegroundColor Green
