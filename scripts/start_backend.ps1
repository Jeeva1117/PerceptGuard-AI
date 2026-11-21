# ==============================================================================
# PerceptGuard AI — Backend Service Startup Script (Windows PowerShell)
# Target Platform: Windows 11 AMD64 / Python 3.13 / Intel Core i3 (CPU Execution)
# ==============================================================================

[CmdletBinding()]
param(
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 8000,
    [switch]$Reload
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path "$PSScriptRoot\..").Path
Set-Location -Path $ProjectRoot

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  PerceptGuard AI — FastAPI Backend Service Launcher      " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Verify Virtual Environment
$PythonExe = Join-Path $ProjectRoot "venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    Write-Error "Python virtual environment not found at $PythonExe. Please run: python -m venv venv && .\venv\Scripts\pip install -r requirements.txt"
    exit 1
}
Write-Host "[OK] Python environment verified: $PythonExe" -ForegroundColor Green

# 2. Verify Required Model Weights
$YoloWeights = Join-Path $ProjectRoot "models\yolo\yolo11n.pt"
$LstmWeights = Join-Path $ProjectRoot "models\forecasting\lstm\model.pt"

if (-not (Test-Path $YoloWeights)) {
    Write-Warning "YOLO weights missing at $YoloWeights. Video detection will run in degraded mode."
} else {
    Write-Host "[OK] YOLO11n detection weights verified: $YoloWeights" -ForegroundColor Green
}

if (-not (Test-Path $LstmWeights)) {
    Write-Warning "LSTM forecasting checkpoint missing at $LstmWeights. Forecasting will report UNAVAILABLE."
} else {
    Write-Host "[OK] PyTorch LSTM forecasting weights verified: $LstmWeights" -ForegroundColor Green
}

# 3. Check for Port Conflicts
$TargetUrl = "http://" + $HostAddress + ":" + $Port
$PortInUse = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue | Where-Object { $_.State -eq "Listen" }
if ($PortInUse) {
    Write-Host "[INFO] Port $Port is currently active. Probing backend health..." -ForegroundColor Yellow
    try {
        $HealthResp = Invoke-RestMethod -Uri ($TargetUrl + "/health/live") -TimeoutSec 2 -ErrorAction Stop
        if ($HealthResp.status -eq "alive" -or $HealthResp.status -eq "ALIVE" -or $HealthResp.status -eq "OK") {
            Write-Host ("[OK] PerceptGuard backend is already running on " + $TargetUrl) -ForegroundColor Green
            exit 0
        }
    } catch {
        Write-Error "Port $Port is in use by another process (PID: $($PortInUse.OwningProcess)). Please terminate it or run scripts/stop_project.ps1"
        exit 1
    }
}

# 4. Launch FastAPI Uvicorn Server
Write-Host ("[INFO] Starting FastAPI on " + $TargetUrl + "...") -ForegroundColor Cyan
$UvicornArgs = @("-m", "uvicorn", "src.api.app:app", "--host", $HostAddress, "--port", "$Port")
if ($Reload) {
    $UvicornArgs += "--reload"
}

& $PythonExe $UvicornArgs
