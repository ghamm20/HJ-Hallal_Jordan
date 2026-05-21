param()

$ErrorActionPreference = "Stop"

$Root = $PSScriptRoot
Set-Location $Root
$StatePath = Join-Path $Root "runtime\halal-jordan-launch-state.json"
$LlamaStatePath = Join-Path $Root "runtime\halal-jordan-llama-state.json"

function Resolve-AppPidFromState {
    param(
        [psobject]$State
    )

    if ($null -eq $State) {
        return 0
    }

    $port = 0
    if ($null -ne $State.port) {
        $port = [int]$State.port
    }
    if ($port -le 0) {
        return 0
    }

    $listenHost = [string]($State.host)
    if ([string]::IsNullOrWhiteSpace($listenHost)) {
        $listenHost = "127.0.0.1"
    }

    $expectedPython = [string]($State.python)
    $connections = @()
    try {
        $connections = @(Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction Stop)
    }
    catch {
        return 0
    }

    foreach ($connection in $connections) {
        if ($listenHost -and $connection.LocalAddress -ne $listenHost -and $connection.LocalAddress -ne "0.0.0.0") {
            continue
        }
        $candidatePid = [int]$connection.OwningProcess
        if ($candidatePid -le 0) {
            continue
        }
        try {
            $proc = Get-CimInstance Win32_Process -Filter ("ProcessId = " + $candidatePid)
        }
        catch {
            continue
        }
        $candidatePath = [string]$proc.ExecutablePath
        $candidateCommandLine = [string]$proc.CommandLine
        $matchesExpectedPython = $expectedPython -and $candidatePath -and $candidatePath -ieq $expectedPython
        $matchesHalalJordanCommand = $candidateCommandLine -like "*app.backend.main:app*"
        if ($matchesExpectedPython -or $matchesHalalJordanCommand) {
            return $candidatePid
        }
    }

    return 0
}

function Write-StopLine {
    param(
        [string]$Message,
        [string]$Color = "Gray"
    )
    Write-Host $Message -ForegroundColor $Color
}

Write-StopLine "Halal Jordan stop helper" "Cyan"

if (-not (Test-Path $StatePath)) {
    Write-StopLine "No saved launcher state was found." "Yellow"
    Write-StopLine "If Halal Jordan is still running, close the server window manually." "Yellow"
    exit 0
}

try {
    $state = Get-Content -Raw $StatePath | ConvertFrom-Json
}
catch {
    Remove-Item $StatePath -ErrorAction SilentlyContinue
    Write-StopLine "Launcher state was unreadable and has been cleared." "Yellow"
    Write-StopLine "If a server window is still open, close it manually." "Yellow"
    exit 1
}

$serverPid = 0
if ($null -ne $state.pid) {
    $serverPid = [int]$state.pid
}
$appPid = 0
if ($null -ne $state.app_pid) {
    $appPid = [int]$state.app_pid
}
if ($appPid -le 0) {
    $appPid = Resolve-AppPidFromState -State $state
}
$appStopped = $false
if ($appPid -gt 0) {
    $appProcess = Get-Process -Id $appPid -ErrorAction SilentlyContinue
    if ($appProcess) {
        Stop-Process -Id $appPid -Force
        Start-Sleep -Milliseconds 600
        $appStopped = $true
        Write-StopLine ("Stopped Halal Jordan app process (PID " + $appPid + ").") "Green"
    }
}
$serverStopped = $false
if ($serverPid -gt 0 -and $serverPid -ne $appPid) {
    $process = Get-Process -Id $serverPid -ErrorAction SilentlyContinue
    if ($process) {
        Stop-Process -Id $serverPid -Force
        Start-Sleep -Milliseconds 600
        $serverStopped = $true
        Write-StopLine ("Stopped Halal Jordan server window (PID " + $serverPid + ").") "Green"
    }
    else {
        Write-StopLine ("No running Halal Jordan server process was found for saved PID " + $serverPid + ".") "Yellow"
    }
}

$llamaPid = 0
if ($null -ne $state.llama_pid) {
    $llamaPid = [int]$state.llama_pid
}
if ($llamaPid -gt 0) {
    $llamaProcess = Get-Process -Id $llamaPid -ErrorAction SilentlyContinue
    if ($llamaProcess) {
        Stop-Process -Id $llamaPid -Force
        Start-Sleep -Milliseconds 600
        Write-StopLine ("Stopped bundled llama.cpp process (PID " + $llamaPid + ").") "Green"
    }
    else {
        Write-StopLine ("No running bundled llama.cpp process was found for saved PID " + $llamaPid + ".") "Yellow"
    }
}

Remove-Item $StatePath -ErrorAction SilentlyContinue
Remove-Item $LlamaStatePath -ErrorAction SilentlyContinue
if (-not $appStopped -and -not $serverStopped -and -not ($llamaPid -gt 0)) {
    Write-StopLine "No launcher-managed processes needed stopping." "Yellow"
}
if ($state.url) {
    Write-StopLine ("Last URL was " + $state.url)
}
