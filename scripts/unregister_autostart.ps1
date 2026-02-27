<#
.SYNOPSIS
    Remove the OpenStinger auto-start Task Scheduler task.

.EXAMPLE
    .\scripts\unregister_autostart.ps1
#>

param(
    [string]$TaskName = "OpenStinger - Memory Harness"
)

Write-Host ""
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "  OpenStinger - Unregister Auto-Start" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "✅ Task '$TaskName' removed." -ForegroundColor Green
} else {
    Write-Host "ℹ️  Task '$TaskName' was not registered." -ForegroundColor Yellow
}
Write-Host ""
