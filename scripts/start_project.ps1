# ==============================================================================
# PerceptGuard AI — Master One-Command Project Startup (Windows PowerShell)
# Target Platform: Windows 11 AMD64 / Python 3.13 / Pure CPU Execution
# ==============================================================================

[CmdletBinding()]
param(
    [int]$BackendPort = 8000,
    [int]$DashboardPort = 8501
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path "$PSScriptRoot\..").Path
Set-Location -Path $ProjectRoot

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  PerceptGuard AI — Master System Launcher (Windows 11)   " -ForegroundColor Cyan
Write-Host "  Anticipating Detection Failures Before They Happen     " -ForegroundColor DarkCyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Environment & Python Executable Verification
$PythonExe = Join-Path $ProjectRoot "venv\Scripts\python.exe"
$StreamlitExe = Join-Path $ProjectRoot "venv\Scripts\streamlit.exe"

if (-not (Test-Path $PythonExe) -or -not (Test-Path $StreamlitExe)) {
    Write-Error "Virtual environment missing or incomplete. Setup instructions:`n  python -m venv venv`n  .\venv\Scripts\pip install -r requirements.txt"
    exit 1
}
Write-Host "[1/5] Python virtual environment verified." -ForegroundColor Green

# 2. Model Checkpoints & Pretrained Weights Verification
$YoloWeights = Join-Path $ProjectRoot "models\detection\yolo11n.pt"
$LstmWeights = Join-Path $ProjectRoot "models\forecasting\model.pt"

if (Test-Path $YoloWeights) {
    Write-Host "[2/5] YOLO11n weights detected: $YoloWeights" -ForegroundColor Green
} else {
    Write-Warning "YOLO11n weights missing at $YoloWeights."
}

if (Test-Path $LstmWeights) {
    Write-Host "[2/5] PyTorch LSTM checkpoint detected: $LstmWeights" -ForegroundColor Green
} else {
    Write-Warning "LSTM checkpoint missing at $LstmWeights."
}

# 3. Check and Launch FastAPI Backend
$BackendUrl = "http://127.0.0.1:${BackendPort}"
$BackendLiveUrl = "${BackendUrl}/health/live"
$BackendRunning = $false

$PortConn = Get-NetTCPConnection -LocalPort $BackendPort -ErrorAction SilentlyContinue | Where-Object { $_.State -eq "Listen" }
if ($PortConn) {
    try {
        $Health = Invoke-RestMethod -Uri $BackendLiveUrl -TimeoutSec 2 -ErrorAction Stop
        if ($Health.status -eq "ALIVE" -or $Health.status -eq "OK") {
            Write-Host "[3/5] Backend already running and healthy at $BackendUrl" -ForegroundColor Green
            $BackendRunning = $true
        }
    } catch {
        Write-Error "Port $BackendPort is occupied by an external PID ($($PortConn.OwningProcess)). Stop it first using scripts\stop_project.ps1"
        exit 1
    }
}

if (-not $BackendRunning) {
    Write-Host "[3/5] Launching FastAPI backend on $BackendUrl..." -ForegroundColor Cyan
    $BackendProc = Start-Process -FilePath $PythonExe `
        -ArgumentList "-m uvicorn src.api.app:app --host 127.0.0.1 --port $BackendPort" `
        -WorkingDirectory $ProjectRoot `
        -PassThru

    Write-Host "Waiting for backend health confirmation (PID: $($BackendProc.Id))..." -ForegroundColor Yellow
    $MaxWaitSec = 20
    $Elapsed = 0
    while ($Elapsed -lt $MaxWaitSec) {
        Start-Sleep -Seconds 1
        $Elapsed++
        try {
            $Resp = Invoke-RestMethod -Uri $BackendLiveUrl -TimeoutSec 1 -ErrorAction Stop
            if ($Resp.status -eq "ALIVE" -or $Resp.status -eq "OK") {
                $BackendRunning = $true
                break
            }
        } catch {
            # Still spinning up
        }
    }

    if (-not $BackendRunning) {
        Write-Error "Backend failed to report healthy status within $MaxWaitSec seconds. Check logs."
        exit 1
    }
    Write-Host "[3/5] Backend is ONLINE and responding at $BackendUrl" -ForegroundColor Green
}

# 4. Service Endpoint Summary
Write-Host "[4/5] Active Service Endpoints:" -ForegroundColor Cyan
Write-Host "  * REST API Base:       $BackendUrl" -ForegroundColor White
Write-Host "  * Swagger API Docs:    ${BackendUrl}/docs" -ForegroundColor White
Write-Host "  * OpenAPI JSON:        ${BackendUrl}/openapi.json" -ForegroundColor White
Write-Host "  * System Health:       ${BackendUrl}/health" -ForegroundColor White
Write-Host "  * Streamlit Dashboard: http://localhost:${DashboardPort}" -ForegroundColor White

# 5. Launch Streamlit Operations Console
Write-Host "[5/5] Launching Streamlit dashboard..." -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop the dashboard." -ForegroundColor Yellow
Write-Host "To stop background backend: .\scripts\stop_project.ps1" -ForegroundColor Yellow
Write-Host "==========================================================" -ForegroundColor Cyan

& $StreamlitExe run dashboard/app.py --server.port $DashboardPort --browser.gatherUsageStats false
