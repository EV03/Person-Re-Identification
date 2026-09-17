[CmdletBinding()]
param(
    [ValidateRange(1, 65535)]
    [int]$Port = 8501
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $VenvPython)) {
    throw "Die Projektumgebung fehlt. Zuerst .\scripts\setup_windows.ps1 ausführen."
}

$ActivePorts = [System.Net.NetworkInformation.IPGlobalProperties]::GetIPGlobalProperties().GetActiveTcpListeners().Port
if ($Port -in $ActivePorts) {
    throw "Port $Port wird bereits verwendet. Vermutlich läuft noch eine ältere Streamlit-Instanz. Das zugehörige Terminal mit Strg+C beenden oder einen anderen Port über -Port angeben."
}

Set-Location -LiteralPath $ProjectRoot
& $VenvPython scripts/run_streamlit.py run app/ui/streamlit_app.py --server.port $Port
exit $LASTEXITCODE
