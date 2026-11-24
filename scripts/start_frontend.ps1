# ==============================================================================
# PerceptGuard AI — Modern React Frontend Startup Script (Windows PowerShell)
# Target Platform: Windows 11 AMD64 / React + Vite + TypeScript / Port 3000
# ==============================================================================

[CmdletBinding()]
param(
    [int]$Port = 3000,
    [string]$BackendUrl = "http://127.0.0.1:8000"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path "$PSScriptRoot\..").Path
Set-Location -Path $ProjectRoot

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  PerceptGuard AI — React + TypeScript Frontend Launcher   " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Verify Node and Frontend directory
$FrontendDir = Join-Path $ProjectRoot "frontend"
if (-not (Test-Path $FrontendDir)) {
    Write-Error "Frontend directory not found at $FrontendDir."
    exit 1
}

$NodeCmd = Get-Command "npm" -ErrorAction SilentlyContinue
if (-not $NodeCmd) {
    Write-Error "npm was not found in PATH. Please install Node.js (v18+)."
    exit 1
}
Write-Host "[OK] Frontend verified: $FrontendDir (using npm)" -ForegroundColor Green

# 2. Check Backend Connectivity
Write-Host "[INFO] Probing FastAPI backend at $BackendUrl..." -ForegroundColor Cyan
try {
    $HealthResp = Invoke-RestMethod -Uri ($BackendUrl + "/health/live") -TimeoutSec 2 -ErrorAction Stop
    Write-Host "[OK] Backend connection established: $($HealthResp.status)" -ForegroundColor Green
} catch {
    Write-Warning "FastAPI backend at $BackendUrl is not responding yet. Ensure it is started on port 8000."
}

# 3. Check Frontend Port
$PortInUse = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue | Where-Object { $_.State -eq "Listen" }
if ($PortInUse) {
    Write-Warning "Port $Port is already in use by PID $($PortInUse.OwningProcess)."
}

# 4. Launch React Frontend
Write-Host ("[INFO] Starting React Frontend on http://localhost:" + $Port + "...") -ForegroundColor Cyan
npm run dev --prefix frontend -- --port $Port --host
