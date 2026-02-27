<#
.SYNOPSIS
    Start OpenStinger MCP server safely with startup verification.

.DESCRIPTION
    - Checks if server is already running on the configured port (default 8766)
    - Verifies FalkorDB Docker containers are healthy first
    - Starts the Tier 3 (all 24 tools) server as a background process
    - Writes PID to .openstinger/server.pid for stop_openstinger.ps1
    - Tails the log for 12 seconds to confirm successful startup
    - Exits with code 0 on success, 1 on failure

.EXAMPLE
    .\scripts\start_openstinger.ps1
    .\scripts\start_openstinger.ps1 -Port 8767
#>

param(
    [int]$Port = 8766,
    [string]$WorkDir = "C:\Users\bells\CLAUDE_CODE\openstinger"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$PidFile  = Join-Path $WorkDir ".openstinger\server.pid"
$LogFile  = Join-Path $WorkDir ".openstinger\openstinger.log"
$Venv     = Join-Path $WorkDir ".venv\Scripts\python.exe"
$Module   = "openstinger.gradient.mcp.server"

Write-Host ""
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "  OpenStinger - Start" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

# ── 1. Already running? ──────────────────────────────────────────────────────
$existing = netstat -ano 2>$null | Select-String ":$Port " | Select-String "LISTENING"
if ($existing) {
    Write-Host "✅ OpenStinger is already running on port $Port." -ForegroundColor Green
    if (Test-Path $PidFile) {
        $pid = Get-Content $PidFile -Raw
        Write-Host "   PID file: $pid"
    }
    exit 0
}

Write-Host "Port $Port is free — proceeding with startup." -ForegroundColor Gray

# ── 2. Check FalkorDB Docker containers ──────────────────────────────────────
Write-Host ""
Write-Host "Checking FalkorDB..." -ForegroundColor Gray
$falkordb = docker ps --filter "name=openstinger_falkordb" --format "{{.Status}}" 2>$null
if (-not $falkordb) {
    Write-Host "❌ FalkorDB container is not running. Starting Docker containers..." -ForegroundColor Yellow
    Push-Location $WorkDir
    docker compose up -d 2>&1 | Out-Null
    Pop-Location
    Start-Sleep -Seconds 5
    $falkordb = docker ps --filter "name=openstinger_falkordb" --format "{{.Status}}" 2>$null
    if (-not $falkordb) {
        Write-Host "❌ Failed to start FalkorDB. Is Docker Desktop running?" -ForegroundColor Red
        exit 1
    }
}
Write-Host "✅ FalkorDB: $falkordb" -ForegroundColor Green

# ── 3. Verify Python venv ─────────────────────────────────────────────────────
if (-not (Test-Path $Venv)) {
    Write-Host "❌ Python venv not found at $Venv" -ForegroundColor Red
    Write-Host "   Run: python -m venv .venv && .venv\Scripts\pip install -e '.[dev]'" -ForegroundColor Yellow
    exit 1
}

# ── 4. Launch server ──────────────────────────────────────────────────────────
Write-Host ""
Write-Host "Starting Tier 3 MCP server (all 24 tools)..." -ForegroundColor Gray
Push-Location $WorkDir

$proc = Start-Process `
    -FilePath $Venv `
    -ArgumentList "-m", $Module `
    -RedirectStandardOutput $LogFile `
    -RedirectStandardError $LogFile `
    -WorkingDirectory $WorkDir `
    -PassThru `
    -WindowStyle Hidden

Pop-Location

if (-not $proc) {
    Write-Host "❌ Failed to start process." -ForegroundColor Red
    exit 1
}

# Save PID
$proc.Id | Out-File $PidFile -Encoding utf8
Write-Host "   Started PID $($proc.Id) → $PidFile" -ForegroundColor Gray

# ── 5. Wait and verify ────────────────────────────────────────────────────────
Write-Host "   Waiting 12 seconds for startup..." -ForegroundColor Gray
Start-Sleep -Seconds 12

# Check port is now listening
$listening = netstat -ano 2>$null | Select-String ":$Port " | Select-String "LISTENING"
$logContent = if (Test-Path $LogFile) { Get-Content $LogFile -Tail 20 -Raw } else { "" }

$hasReady   = $logContent -match "Application startup complete"
$hasError   = $logContent -match "ERROR"

Write-Host ""
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "  Startup Log (last 20 lines):" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
if (Test-Path $LogFile) {
    Get-Content $LogFile -Tail 20 | ForEach-Object { Write-Host "  $_" -ForegroundColor Gray }
}
Write-Host ""

if ($listening -and $hasReady -and -not $hasError) {
    Write-Host "✅ OpenStinger is UP on port $Port (PID $($proc.Id))" -ForegroundColor Green
    Write-Host "   SSE endpoint: http://localhost:$Port/sse" -ForegroundColor Green
    exit 0
} elseif ($hasError) {
    Write-Host "❌ Server started but log contains ERROR — check $LogFile" -ForegroundColor Red
    exit 1
} else {
    Write-Host "⚠️  Server may still be starting. Check: .openstinger\openstinger.log" -ForegroundColor Yellow
    exit 0
}
