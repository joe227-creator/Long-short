$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$LogFile = Join-Path $ProjectRoot "scheduled_trade.log"

function Write-LogLine {
    param([string]$Message)
    $Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss zzz"
    Add-Content -LiteralPath $LogFile -Value "[$Timestamp] $Message"
}

Write-LogLine "===== Scheduled trade and drift-monitor run started ====="

# Day-of-week guard: only execute on Sunday (defensive check against scheduler misconfiguration)
$Today = Get-Date
if ($Today.DayOfWeek -ne [System.DayOfWeek]::Sunday) {
    $ts = $Today.ToString("yyyy-MM-dd HH:mm:ss zzz")
    Add-Content -LiteralPath $LogFile -Value "[$ts] SKIPPED: not Sunday (today is $($Today.DayOfWeek)). Rescheduled to Sunday 21:00."
    exit 0
}

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python executable not found: $Python"
}

Push-Location $ProjectRoot
try {
    $env:HEDGE_PORTFOLIO_SCHEDULED = "1"
    & cmd.exe /d /c "`"$Python`" `"trade.py`" 2>&1" | ForEach-Object {
        $_
        Add-Content -LiteralPath $LogFile -Value $_
    }
    $ExitCode = $LASTEXITCODE
    if ($ExitCode -ne 0) {
        throw "trade.py exited with code $ExitCode"
    }
    Write-LogLine "===== Scheduled trade and drift-monitor run finished ====="
}
catch {
    Write-LogLine "ERROR: $($_.Exception.Message)"
    throw
}
finally {
    Remove-Item Env:\HEDGE_PORTFOLIO_SCHEDULED -ErrorAction SilentlyContinue
    Pop-Location
}
