# ==============================================================================
# PerceptGuard AI — Project Shutdown Script (Windows PowerShell)
# Target Platform: Windows 11 AMD64
# ==============================================================================

[CmdletBinding()]
param(
    [int[]]$Ports = @(8000, 3000, 8501)
)

$ErrorActionPreference = "Continue"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  PerceptGuard AI — Service Shutdown Utility               " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

$StoppedCount = 0

foreach ($Port in $Ports) {
    Write-Host "[INFO] Checking port $Port..." -ForegroundColor Cyan
    $Connections = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue | Where-Object { $_.State -eq "Listen" }

    if ($Connections) {
        foreach ($Conn in $Connections) {
            $PidToStop = $Conn.OwningProcess
            try {
                $Proc = Get-Process -Id $PidToStop -ErrorAction Stop
                Write-Host "  Stopping PID $PidToStop ($($Proc.ProcessName)) on port $Port..." -ForegroundColor Yellow
                Stop-Process -Id $PidToStop -Force -ErrorAction SilentlyContinue
                $StoppedCount++
                Write-Host "  [OK] PID $PidToStop terminated." -ForegroundColor Green
            } catch {
                Write-Warning ("  Could not stop PID " + $PidToStop + ": " + $_)
            }
        }
    } else {
        Write-Host "  Port $Port is free (no active listener)." -ForegroundColor Gray
    }
}

# Verify release
Start-Sleep -Seconds 1
$Remaining = @()
foreach ($Port in $Ports) {
    $StillInUse = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue | Where-Object { $_.State -eq "Listen" }
    if ($StillInUse) {
        $Remaining += $Port
    }
}

if ($Remaining.Count -eq 0) {
    Write-Host "==========================================================" -ForegroundColor Green
    Write-Host "  All PerceptGuard services stopped. Ports 8000, 3000, and 8501 released." -ForegroundColor Green
    Write-Host "==========================================================" -ForegroundColor Green
} else {
    $Occupied = $Remaining -join ", "
    Write-Warning "Ports still occupied: $Occupied"
}
