# start_all.ps1 — Windows equivalent of start_all.sh
# Starts all 5 services for the Legal Multi-Agent System (Stage 5)
#
# Usage:  .\start_all.ps1
# Stop:   Close this window, or press Ctrl+C then run .\stop_all.ps1

$ErrorActionPreference = "Stop"
$ROOT = $PSScriptRoot

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  Legal Multi-Agent System — Stage 5" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Track PIDs for cleanup
$pids = @{}

function Start-Agent {
    param(
        [string]$Name,
        [string]$Module,
        [int]$Port,
        [string]$Color = "Green"
    )
    Write-Host "Starting $Name on port $Port..." -ForegroundColor $Color
    $log = Join-Path $ROOT "logs\$Name.log"
    New-Item -ItemType Directory -Force -Path (Join-Path $ROOT "logs") | Out-Null
    $proc = Start-Process -NoNewWindow -FilePath "uv" `
        -ArgumentList "run", "python", "-m", $Module `
        -WorkingDirectory $ROOT `
        -RedirectStandardOutput $log `
        -RedirectStandardError "$log.err" `
        -PassThru
    $pids[$Name] = $proc.Id
    Write-Host "  PID $($proc.Id) | Log: logs\$Name.log" -ForegroundColor Gray
    return $proc
}

# ── 1. Registry (must start first) ───────────────────────────────────────────
$reg = Start-Agent -Name "registry" -Module "registry" -Port 10000 -Color "Cyan"
Write-Host "  Waiting for registry to be ready..." -ForegroundColor Gray
Start-Sleep -Seconds 3

# ── 2. Leaf agents (Tax + Compliance, no dependencies on each other) ─────────
$tax  = Start-Agent -Name "tax_agent"        -Module "tax_agent"        -Port 10102 -Color "Yellow"
$comp = Start-Agent -Name "compliance_agent" -Module "compliance_agent" -Port 10103 -Color "Yellow"
Start-Sleep -Seconds 3

# ── 3. Orchestrators ─────────────────────────────────────────────────────────
$law  = Start-Agent -Name "law_agent"      -Module "law_agent"      -Port 10101 -Color "Magenta"
Start-Sleep -Seconds 3
$cust = Start-Agent -Name "customer_agent" -Module "customer_agent" -Port 10100 -Color "Green"
Start-Sleep -Seconds 2

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  All services started!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Registry:         http://localhost:10000" -ForegroundColor White
Write-Host "  Customer Agent:   http://localhost:10100" -ForegroundColor White
Write-Host "  Law Agent:        http://localhost:10101" -ForegroundColor White
Write-Host "  Tax Agent:        http://localhost:10102" -ForegroundColor White
Write-Host "  Compliance Agent: http://localhost:10103" -ForegroundColor White
Write-Host ""
Write-Host "  Test the system:" -ForegroundColor Yellow
Write-Host "    uv run python test_client.py" -ForegroundColor Yellow
Write-Host ""
Write-Host "  Check registered agents:" -ForegroundColor Yellow
Write-Host "    Invoke-RestMethod http://localhost:10000/agents" -ForegroundColor Yellow
Write-Host ""
Write-Host "  Logs saved in: logs\" -ForegroundColor Gray
Write-Host ""
Write-Host "Press Ctrl+C to stop all services." -ForegroundColor Red
Write-Host ""

# Save PIDs to file for stop_all.ps1
$pids | ConvertTo-Json | Out-File (Join-Path $ROOT "logs\.pids.json") -Encoding utf8

# Wait for all processes
try {
    Wait-Process -Id $reg.Id, $tax.Id, $comp.Id, $law.Id, $cust.Id -ErrorAction SilentlyContinue
} catch {
    # Ctrl+C
}
