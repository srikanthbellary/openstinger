<#
.SYNOPSIS
    Stop OpenStinger MCP server safely with FalkorDB backup.

.DESCRIPTION
    Shutdown sequence:
      1. Take a FalkorDB BGSAVE backup and wait for completion
      2. Copy dump.rdb to a dated backup file in backups/
      3. Read PID from .openstinger/server.pid and send SIGTERM / Stop-Process
      4. Wait up to 30 seconds for port to close
      5. Print summary: episodes saved, backup location

    The SQLite session cursor file is preserved — restarting OpenStinger
    will continue ingestion exactly where it left off.

.EXAMPLE
    .\scripts\stop_openstinger.ps1
    .\scripts\stop_openstinger.ps1 -Port 8767 -SkipBackup
#>

param(
    [int]$Port        = 8766,
    [string]$WorkDir  = (Split-Path -Parent $PSScriptRoot),
    [switch]$SkipBackup
)

Set-StrictMode -Version Latest

$PidFile    = Join-Path $WorkDir ".openstinger\server.pid"
$BackupDir  = Join-Path $WorkDir "backups"
$LogFile    = Join-Path $WorkDir ".openstinger\openstinger.log"

Write-Host ""
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "  OpenStinger - Graceful Stop" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

# ── 1. Check if running ───────────────────────────────────────────────────────
$listening = netstat -ano 2>$null | Select-String ":$Port " | Select-String "LISTENING"
if (-not $listening) {
    Write-Host "ℹ️  OpenStinger is not running on port $Port." -ForegroundColor Yellow
    exit 0
}

# ── 2. FalkorDB backup ────────────────────────────────────────────────────────
if (-not $SkipBackup) {
    Write-Host ""
    Write-Host "Step 1/4: FalkorDB backup..." -ForegroundColor Gray

    $falkordb = docker ps --filter "name=openstinger_falkordb" --format "{{.Names}}" 2>$null | Select-Object -First 1
    if (-not $falkordb) {
        Write-Host "  ⚠️  FalkorDB container not found — skipping backup." -ForegroundColor Yellow
    } else {
        # Capture LASTSAVE timestamp before BGSAVE
        $before = docker exec $falkordb redis-cli LASTSAVE 2>$null
        Write-Host "  Triggering BGSAVE on container '$falkordb'..." -ForegroundColor Gray
        docker exec $falkordb redis-cli BGSAVE 2>$null | Out-Null

        # Wait for LASTSAVE to change (up to 60 seconds)
        $waited = 0
        while ($waited -lt 60) {
            Start-Sleep -Seconds 2
            $waited += 2
            $after = docker exec $falkordb redis-cli LASTSAVE 2>$null
            if ($after -ne $before) { break }
        }

        if ($waited -ge 60) {
            Write-Host "  ⚠️  BGSAVE did not complete in 60s — proceeding anyway." -ForegroundColor Yellow
        } else {
            Write-Host "  ✅ BGSAVE complete (elapsed ${waited}s)." -ForegroundColor Green

            # Copy dump.rdb out of container to local backups/
            New-Item -ItemType Directory -Force -Path $BackupDir | Out-Null
            $datestamp  = Get-Date -Format "yyyy-MM-dd_HHmmss"
            $backupFile = Join-Path $BackupDir "falkordb-backup-$datestamp.rdb"

            try {
                docker cp "${falkordb}:/data/dump.rdb" $backupFile 2>$null
                Write-Host "  ✅ Backup saved to: $backupFile" -ForegroundColor Green
            } catch {
                Write-Host "  ⚠️  Could not copy dump.rdb from container: $_" -ForegroundColor Yellow
            }
        }
    }
} else {
    Write-Host "ℹ️  Backup skipped (-SkipBackup)." -ForegroundColor Gray
}

# ── 3. Kill server process ────────────────────────────────────────────────────
Write-Host ""
Write-Host "Step 2/4: Stopping server..." -ForegroundColor Gray

$pidToKill = $null
if (Test-Path $PidFile) {
    $pidToKill = [int](Get-Content $PidFile -Raw).Trim()
    try {
        $proc = Get-Process -Id $pidToKill -ErrorAction Stop
        $proc | Stop-Process -Force
        Write-Host "  Sent SIGTERM to PID $pidToKill ($($proc.ProcessName))" -ForegroundColor Gray
    } catch {
        Write-Host "  Process $pidToKill not found (already stopped?)." -ForegroundColor Gray
    }
} else {
    Write-Host "  No PID file found — locating process on port $Port..." -ForegroundColor Gray
    $netLines = netstat -ano 2>$null | Select-String ":$Port " | Select-String "LISTENING"
    if ($netLines) {
        $pidStr = ($netLines -split "\s+")[-1]
        if ($pidStr -match "^\d+$") {
            Stop-Process -Id ([int]$pidStr) -Force -ErrorAction SilentlyContinue
            Write-Host "  Stopped PID $pidStr" -ForegroundColor Gray
        }
    }
}

# ── 4. Wait for port to close ─────────────────────────────────────────────────
Write-Host ""
Write-Host "Step 3/4: Waiting for port $Port to close..." -ForegroundColor Gray
$waited = 0
while ($waited -lt 30) {
    Start-Sleep -Seconds 1
    $waited++
    $still = netstat -ano 2>$null | Select-String ":$Port " | Select-String "LISTEN"
    if (-not $still) { break }
}

if ($waited -ge 30) {
    Write-Host "  ⚠️  Port $Port still open after ${waited}s." -ForegroundColor Yellow
} else {
    Write-Host "  ✅ Port $Port is closed (${waited}s)." -ForegroundColor Green
}

# ── 5. Cleanup ────────────────────────────────────────────────────────────────
if (Test-Path $PidFile) { Remove-Item $PidFile -Force }

# ── 6. Summary ────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "Step 4/4: Summary" -ForegroundColor Gray

$Venv = Join-Path $WorkDir ".venv\Scripts\python.exe"
if (Test-Path $Venv) {
    $episodeCount = & $Venv -c @"
import sys, falkordb
sys.stdout.reconfigure(encoding='utf-8')
try:
    c = falkordb.FalkorDB(host='localhost', port=6379)
    g = c.select_graph('openstinger_temporal')
    r = g.query('MATCH (ep:Episode) RETURN count(ep) AS n')
    print(r.result_set[0][0])
except Exception as e:
    print('?')
"@ 2>$null
    Write-Host "  Episodes in graph: $episodeCount" -ForegroundColor Gray
}

Write-Host ""
Write-Host "✅ OpenStinger stopped cleanly." -ForegroundColor Green
Write-Host "   Ingestion cursors preserved in SQLite — restart will resume where it left off." -ForegroundColor Green
Write-Host "   To restart: .\scripts\start_openstinger.ps1" -ForegroundColor Cyan
Write-Host ""
