[CmdletBinding()]
param(
    [string[]]$Yolo = @(),
    [string[]]$Osnet = @(),
    [switch]$List
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $VenvPython)) {
    throw "Die Projektumgebung fehlt. Zuerst .\scripts\setup_windows.ps1 ausführen."
}

Set-Location -LiteralPath $ProjectRoot
if ($List) {
    & $VenvPython scripts/manage_models.py list
    exit $LASTEXITCODE
}

$Arguments = @("scripts/manage_models.py", "install")
foreach ($Model in $Yolo) {
    $Arguments += @("--yolo", $Model)
}
foreach ($Model in $Osnet) {
    $Arguments += @("--osnet", $Model)
}

if ($Yolo.Count -eq 0 -and $Osnet.Count -eq 0) {
    throw "Mindestens ein Modell mit -Yolo oder -Osnet angeben."
}

& $VenvPython @Arguments
exit $LASTEXITCODE
