[CmdletBinding()]
param(
    [string]$Video = "Test-daten\Default\Tim_Allaround_Ohne_Details_Weises_Tshirt_Schwarze_hose.mp4",
    [ValidateSet("auto", "cpu", "cuda")]
    [string]$Device = "cuda"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $VenvPython)) {
    throw "Die Projektumgebung fehlt. Zuerst .\scripts\setup_windows.ps1 ausführen."
}

$VideoPath = if ([System.IO.Path]::IsPathRooted($Video)) { $Video } else { Join-Path $ProjectRoot $Video }
if (-not (Test-Path -LiteralPath $VideoPath)) {
    throw "Smoke-Test-Video nicht gefunden: $VideoPath"
}

$SmokeRunId = Get-Date -Format "yyyyMMdd_HHmmss"
$SmokeRoot = Join-Path $ProjectRoot ("data\smoke_test\" + $SmokeRunId)
$env:REID_DB_PATH = Join-Path $SmokeRoot "db\reid.sqlite3"
$env:REID_SNAPSHOT_DIR = Join-Path $SmokeRoot "snapshots"
$env:REID_OUTPUT_DIR = Join-Path $SmokeRoot "output"
$env:REID_MODE_DIR = Join-Path $SmokeRoot "modes"

Set-Location -LiteralPath $ProjectRoot
& $VenvPython scripts/verify_environment.py
if ($LASTEXITCODE -ne 0) {
    throw "Umgebungsprüfung fehlgeschlagen."
}

& $VenvPython -m app.main --source $VideoPath --mode default --device $Device --max-frames 0
if ($LASTEXITCODE -ne 0) {
    throw "Der vollständige Video-Smoke-Test ist fehlgeschlagen."
}

Write-Host "Smoke-Test erfolgreich. Isolierte Ergebnisse: $SmokeRoot"
