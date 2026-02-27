<#
.SYNOPSIS
    Register OpenStinger as a Windows Task Scheduler auto-start task.

.DESCRIPTION
    Creates a Task Scheduler task that:
    - Triggers at logon AND at system startup
    - Waits 90 seconds post-trigger (gives Docker Desktop time to fully start)
    - Runs start_openstinger.ps1 in a minimized window
    - Runs as the current user with highest privileges
    - Task name: "OpenStinger - Memory Harness"

    Idempotent — safe to run multiple times (deletes existing task first).

.EXAMPLE
    .\scripts\register_autostart.ps1
    # Then verify: Get-ScheduledTask -TaskName "OpenStinger - Memory Harness"
#>

param(
    [string]$WorkDir   = "C:\Users\bells\CLAUDE_CODE\openstinger",
    [string]$TaskName  = "OpenStinger - Memory Harness",
    [int]$DelaySeconds = 90
)

$StartScript = Join-Path $WorkDir "scripts\start_openstinger.ps1"

Write-Host ""
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "  OpenStinger - Register Auto-Start" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

# Validate
if (-not (Test-Path $StartScript)) {
    Write-Host "❌ Start script not found: $StartScript" -ForegroundColor Red
    exit 1
}

# Remove existing task if present (idempotent)
$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "  Removing existing task '$TaskName'..." -ForegroundColor Gray
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

# Action: PowerShell running the start script
$action = New-ScheduledTaskAction `
    -Execute "PowerShell.exe" `
    -Argument "-NonInteractive -WindowStyle Minimized -ExecutionPolicy Bypass -File `"$StartScript`"" `
    -WorkingDirectory $WorkDir

# Triggers: logon + startup, each with a delay
$delaySpan = New-TimeSpan -Seconds $DelaySeconds
$triggerLogon  = New-ScheduledTaskTrigger -AtLogOn
$triggerStartup = New-ScheduledTaskTrigger -AtStartup

$triggerLogon.Delay   = "PT${DelaySeconds}S"
$triggerStartup.Delay = "PT${DelaySeconds}S"

# Settings
$settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 5) `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 2) `
    -MultipleInstances IgnoreNew `
    -StartWhenAvailable

# Principal: current user, highest privileges
$principal = New-ScheduledTaskPrincipal `
    -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) `
    -LogonType Interactive `
    -RunLevel Highest

# Register
Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger @($triggerLogon, $triggerStartup) `
    -Settings $settings `
    -Principal $principal `
    -Description "Starts the OpenStinger MCP memory harness server after user logon or system startup. Managed by openstinger/scripts/register_autostart.ps1." `
    -Force | Out-Null

Write-Host "✅ Task registered: '$TaskName'" -ForegroundColor Green
Write-Host "   Triggers: At logon + At startup (with ${DelaySeconds}s delay)" -ForegroundColor Green
Write-Host "   Action:   $StartScript" -ForegroundColor Green
Write-Host ""
Write-Host "To verify: Get-ScheduledTask -TaskName '$TaskName' | Format-List" -ForegroundColor Cyan
Write-Host "To remove: .\scripts\unregister_autostart.ps1" -ForegroundColor Cyan
Write-Host ""
