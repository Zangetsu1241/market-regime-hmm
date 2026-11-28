param(
    [string]$Repository = (Split-Path -Parent $PSScriptRoot),
    [string]$At = '04:30',
    [string]$TaskName = 'MarketRegimePaperOnly'
)
$ErrorActionPreference = 'Stop'
# At is LOCAL Windows time. 04:30 Asia/Kolkata is after US close in both DST seasons.
# This script is opt-in: registration never happens during package installation.
$scriptPath = Join-Path $Repository 'scripts/daily-paper.ps1'
$arguments = '-NoProfile -NonInteractive -File "{0}" -Repository "{1}"' -f $scriptPath, $Repository
$action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument $arguments -WorkingDirectory $Repository
$trigger = New-ScheduledTaskTrigger -Daily -At $At
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Hours 2)
Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings -Description 'Local PAPER ONLY market regime simulation; no brokerage orders'
