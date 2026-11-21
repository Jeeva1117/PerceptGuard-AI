# ==============================================================================
# PerceptGuard AI — Streamlit Dashboard Startup Script (Windows PowerShell)
# Target Platform: Windows 11 AMD64 / Streamlit / Intel Core i3 (CPU Execution)
# ==============================================================================

[CmdletBinding()]
param(
    [int]$Port = 8501,
    [string]$BackendUrl = "http://127.0.0.1:8000"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path "$PSScriptRoot\..").Path
Set-Location -Path $ProjectRoot

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  PerceptGuard AI — Streamlit Dashboard Launcher          " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Verify Virtual Environment
$StreamlitExe = Join-Path $ProjectRoot "venv\Scripts\streamlit.exe"
if (-not (Test-Path $StreamlitExe)) {
    Write-Error "Streamlit executable not found at $StreamlitExe. Please run: .\venv\Scripts\pip install -r requirements.txt"
    exit 1
}
Write-Host "[OK] Streamlit verified: $StreamlitExe" -ForegroundColor Green

# 2. Check Backend Connectivity
Write-Host "[INFO] Probing FastAPI backend at $BackendUrl..." -ForegroundColor Cyan
try {
    $HealthResp = Invoke-RestMethod -Uri ($BackendUrl + "/health/live") -TimeoutSec 2 -ErrorAction Stop
    Write-Host "[OK] Backend connection established. Operational Mode: REST_API" -ForegroundColor Green
} catch {
    Write-Warning "FastAPI backend at $BackendUrl is not responding. Streamlit will launch with IN_PROCESS fallback mode."
}

# 3. Check Dashboard Port
$PortInUse = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue | Where-Object { $_.State -eq "Listen" }
if ($PortInUse) {
    Write-Warning "Port $Port is already in use by PID $($PortInUse.OwningProcess). Streamlit will select the next available port."
}

# 4. Launch Streamlit
Write-Host ("[INFO] Starting Streamlit dashboard on http://localhost:" + $Port + "...") -ForegroundColor Cyan
& $StreamlitExe run dashboard/app.py --server.port $Port --browser.gatherUsageStats false
