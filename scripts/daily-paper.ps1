param(
    [string]$Repository = (Split-Path -Parent $PSScriptRoot),
    [string]$Config = 'configs/research.json'
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $Repository
$pythonExecutable = Join-Path $Repository '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonExecutable)) { throw 'Create the project virtual environment first.' }
# Provider retries are bounded. A nonzero exit is visible to Task Scheduler.
# Calendar gates handle weekends, holidays, early closes and stale responses.
& $pythonExecutable -m regime_lab.system_cli paper --config $Config --refresh
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
exit 0
