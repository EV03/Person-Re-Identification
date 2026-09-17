[CmdletBinding()]
param(
    [switch]$Rebuild
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$PythonInstallDir = Join-Path $ProjectRoot ".python"
$UvCacheDir = Join-Path $ProjectRoot ".uv-cache"
$VenvPath = Join-Path $ProjectRoot ".venv"
$VenvPython = Join-Path $VenvPath "Scripts\python.exe"

Set-Location -LiteralPath $ProjectRoot
$env:UV_PYTHON_INSTALL_DIR = $PythonInstallDir
$env:UV_PYTHON_NO_REGISTRY = "1"
$env:UV_CACHE_DIR = $UvCacheDir

$UvCommand = Get-Command uv -ErrorAction SilentlyContinue
if ($null -eq $UvCommand) {
    throw "uv wurde nicht gefunden. Installationsanleitung: https://docs.astral.sh/uv/getting-started/installation/"
}

Write-Host "[1/5] Installiere projektgebundenes Python 3.10.8 nach $PythonInstallDir"
& uv python install 3.10.8 --install-dir $PythonInstallDir --no-bin --no-registry
if ($LASTEXITCODE -ne 0) {
    throw "Python 3.10.8 konnte nicht installiert werden."
}

$NeedsRebuild = $Rebuild -or -not (Test-Path -LiteralPath $VenvPython)
if (-not $NeedsRebuild) {
    try {
        $CurrentVersion = & $VenvPython -c "import platform; print(platform.python_version())"
        $NeedsRebuild = $CurrentVersion.Trim() -ne "3.10.8"
    }
    catch {
        $NeedsRebuild = $true
    }
}

if ($NeedsRebuild) {
    $ResolvedProjectRoot = [System.IO.Path]::GetFullPath($ProjectRoot)
    $ResolvedVenvPath = [System.IO.Path]::GetFullPath($VenvPath)
    if (-not $ResolvedVenvPath.StartsWith($ResolvedProjectRoot + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Unsicheres Ziel für den Neuaufbau der virtuellen Umgebung: $ResolvedVenvPath"
    }
    Write-Host "[2/5] Erzeuge .venv mit Python 3.10.8 neu"
    & uv venv --clear --python 3.10.8 --managed-python $VenvPath
    if ($LASTEXITCODE -ne 0) {
        throw "Die virtuelle Umgebung konnte nicht neu erstellt werden."
    }
}
else {
    Write-Host "[2/5] Vorhandene .venv verwendet bereits Python 3.10.8"
}

Write-Host "[3/5] Synchronisiere exakt mit uv.lock"
& uv sync --locked --python 3.10.8 --managed-python
if ($LASTEXITCODE -ne 0) {
    throw "Die Python-Abhängigkeiten konnten nicht installiert werden."
}

Write-Host "[4/5] Installiere bzw. übernehme die Standardmodelle"
& $VenvPython scripts/manage_models.py install --yolo yolov8n.pt --osnet osnet_x1_0
if ($LASTEXITCODE -ne 0) {
    throw "Die Standardmodelle konnten nicht bereitgestellt werden."
}

Write-Host "[5/5] Prüfe Python, CUDA und Kernimporte"
& $VenvPython scripts/verify_environment.py
if ($LASTEXITCODE -ne 0) {
    throw "Der Umgebungs-Smoke-Test ist fehlgeschlagen."
}

Write-Host "Setup abgeschlossen. Start: .\scripts\start_windows.ps1"
