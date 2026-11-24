# ==============================================================================
# PerceptGuard AI — Dashboard Launcher (Redirects to Modern React Frontend)
# ==============================================================================

[CmdletBinding()]
param(
    [int]$Port = 3000,
    [string]$BackendUrl = "http://127.0.0.1:8000"
)

Write-Host "[INFO] PerceptGuard AI has migrated from Streamlit to the modern React + TypeScript frontend." -ForegroundColor Cyan
Write-Host "[INFO] Delegating to scripts/start_frontend.ps1 on port $Port..." -ForegroundColor Cyan

& "$PSScriptRoot\start_frontend.ps1" -Port $Port -BackendUrl $BackendUrl
