param()

$ErrorActionPreference = "Stop"

function Test-TruthyEnv {
    param([string]$Name)

    $value = [string](Get-Item -Path ("Env:" + $Name) -ErrorAction SilentlyContinue).Value
    return $value -match '^(1|true|yes|on)$'
}

function Write-StartupProfileEvent {
    param(
        [string]$Path,
        [string]$Event,
        [hashtable]$Details = @{}
    )

    if ([string]::IsNullOrWhiteSpace($Path)) {
        return
    }

    $payload = @{
        timestamp = (Get-Date).ToString("o")
        event = $Event
    }
    foreach ($entry in $Details.GetEnumerator()) {
        $payload[$entry.Key] = $entry.Value
    }

    $logDir = Split-Path -Parent $Path
    if ($logDir) {
        New-Item -ItemType Directory -Force -Path $logDir | Out-Null
    }
    Add-Content -Path $Path -Value ($payload | ConvertTo-Json -Compress)
}

function Get-LogContents {
    param([string]$Path)

    if (-not (Test-Path $Path)) {
        return "[missing] $Path"
    }
    try {
        $content = Get-Content -Path $Path -Raw
        if ([string]::IsNullOrWhiteSpace($content)) {
            return "[empty] $Path"
        }
        return $content.TrimEnd()
    }
    catch {
        return "[unreadable] $Path`r`n$($_.Exception.Message)"
    }
}

function Get-BackendFailureDiagnostics {
    param([string]$Reason)

    $launchState = Read-LaunchState -Path $LaunchStatePath
    $details = New-Object System.Collections.Generic.List[string]
    $details.Add("Startup failure reason: $Reason")
    $details.Add("")
    $details.Add("Backend diagnostics:")
    $details.Add("python executable path = " + $(if ($launchState -and $launchState.python) { [string]$launchState.python } else { "unknown" }))
    $details.Add("working directory = " + $(if ($launchState -and $launchState.working_directory) { [string]$launchState.working_directory } else { $Root }))
    $details.Add("backend command = " + $(if ($launchState -and $launchState.backend_command_line) { [string]$launchState.backend_command_line } else { "unknown" }))
    $details.Add("selected port = " + $(if ($launchState -and $launchState.port) { [string]$launchState.port } else { "8000" }))
    $details.Add("health url = " + $(if ($launchState -and $launchState.url) { ([string]$launchState.url).TrimEnd("/") + "/health" } else { "http://127.0.0.1:8000/health" }))
    foreach ($entry in @(
        @{ label = "launcher stdout"; path = $LauncherOutLog },
        @{ label = "launcher stderr"; path = $LauncherErrLog },
        @{ label = "backend stdout"; path = (Join-Path $LogDir "halal-jordan-app.out.log") },
        @{ label = "backend stderr"; path = (Join-Path $LogDir "halal-jordan-app.err.log") }
    )) {
        $details.Add("")
        $details.Add($entry.label + " (" + $entry.path + "):")
        $details.Add((Get-LogContents -Path $entry.path))
    }
    return [string]::Join([Environment]::NewLine, $details)
}

$Root = $PSScriptRoot
Set-Location $Root
. (Join-Path $Root "runtime\launcher\HalalJordanExeCommon.ps1")

$LogDir = Join-Path $Root "logs"
$RuntimeDir = Join-Path $Root "runtime"
$StatusLog = Join-Path $LogDir "start-halal-jordan.log"
$LauncherOutLog = Join-Path $LogDir "start-launcher.out.log"
$LauncherErrLog = Join-Path $LogDir "start-launcher.err.log"
$StartupProfilePath = Join-Path $LogDir "startup-profile.jsonl"
$LaunchStatePath = Join-Path $RuntimeDir "halal-jordan-launch-state.json"
$LauncherPath = Join-Path $Root "LAUNCH_HALAL_JORDAN.ps1"
$StopPath = Join-Path $Root "STOP_HALAL_JORDAN.ps1"
$configPath = Join-Path $Root "config\\runtime_config.json"
$configObject = if (Test-Path $configPath) { Get-Content $configPath -Raw | ConvertFrom-Json } else { $null }
$configuredHealthTimeoutSeconds = 15
if ($configObject -and $null -ne $configObject.launcher_model_ready_timeout_seconds) {
    $timeoutCandidate = 0
    if ([int]::TryParse([string]$configObject.launcher_model_ready_timeout_seconds, [ref]$timeoutCandidate) -and $timeoutCandidate -gt 0) {
        $configuredHealthTimeoutSeconds = $timeoutCandidate
    }
}
$backendHealthTimeoutSeconds = Get-PositiveIntFromEnv -Name "HJ_MODEL_READY_TIMEOUT_SECONDS" -DefaultValue $configuredHealthTimeoutSeconds
$disableModelByEnv = Test-TruthyEnv -Name "HJ_DISABLE_MODEL"
$laptopBuildFromEnv = Test-TruthyEnv -Name "HJ_LAPTOP_BUILD"
$mainModelEnabled = if ($configObject -and $null -ne $configObject.main_model_enabled) { [bool]$configObject.main_model_enabled } else { $true }
$localModelDisabled = $disableModelByEnv -or (-not $mainModelEnabled)
$selectedModelRelativePath = if ($configObject -and $configObject.main_model_path) { [string]$configObject.main_model_path } else { "models/qwen3-4b-laptop.gguf" }
$selectedModelPath = if ([System.IO.Path]::IsPathRooted($selectedModelRelativePath)) { $selectedModelRelativePath } else { Join-Path $Root $selectedModelRelativePath }
$selectedModelGiB = if (Test-Path $selectedModelPath) { [math]::Round((Get-Item $selectedModelPath).Length / 1GB, 2) } else { 0 }
$expectedRamClass = if ($configObject -and $configObject.expected_ram_class) { [string]$configObject.expected_ram_class } else { "8-16GB" }
$laptopBuildFlag = if ($configObject -and $null -ne $configObject.laptop_build) { [bool]$configObject.laptop_build } else { $laptopBuildFromEnv }
if (-not $laptopBuildFlag -and $disableModelByEnv) {
    $laptopBuildFlag = $true
}

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
if (Test-Path $StartupProfilePath) {
    Remove-Item $StartupProfilePath -Force -ErrorAction SilentlyContinue
}
foreach ($stalePath in @(
    $LaunchStatePath,
    $LauncherOutLog,
    $LauncherErrLog,
    (Join-Path $LogDir "halal-jordan-app.out.log"),
    (Join-Path $LogDir "halal-jordan-app.err.log")
)) {
    if (Test-Path $stalePath) {
        Remove-Item $stalePath -Force -ErrorAction SilentlyContinue
    }
}
$startupStartedAt = Get-Date
$stopCompletedAt = $null
$launcherStartedAt = $null
$healthReadyAt = $null
$ui = $null

try {
    Write-StartupProfileEvent -Path $StartupProfilePath -Event "start_script_started" -Details @{
        root = $Root
        local_model_disabled = $localModelDisabled
        laptop_build = $laptopBuildFlag
    }
    $ui = New-StatusWindow -Title "Halal Jordan" -InitialMessage "Starting Halal Jordan..."
    Add-StatusLine -Ui $ui -LogPath $StatusLog -Message "Starting system..."

    if (Test-Path $StopPath) {
        try {
            & powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File $StopPath | Out-Null
        }
        catch {
        }
    }

    $stopCompletedAt = Get-Date
    Write-StartupProfileEvent -Path $StartupProfilePath -Event "startup_cleanup_complete" -Details @{
        elapsed_ms = [int](($stopCompletedAt - $startupStartedAt).TotalMilliseconds)
    }
    Stop-ProcessesOnPort -Port 8000 -Ui $ui -LogPath $StatusLog
    Add-StatusLine -Ui $ui -LogPath $StatusLog -Message "Loading knowledge..."
    Add-StatusLine -Ui $ui -LogPath $StatusLog -Message "Loading sources..."
    Add-StatusLine -Ui $ui -LogPath $StatusLog -Message "Loading retrieval index..."
    if ($localModelDisabled) {
        Add-StatusLine -Ui $ui -LogPath $StatusLog -Message "Local model disabled for laptop build."
        Add-StatusLine -Ui $ui -LogPath $StatusLog -Message "Running retrieval-first fast_mode only."
        Add-StatusLine -Ui $ui -LogPath $StatusLog -Message "final generation unavailable = true"
    }
    else {
        Add-StatusLine -Ui $ui -LogPath $StatusLog -Message "Loading local model..."
        Add-StatusLine -Ui $ui -LogPath $StatusLog -Message "This can take 1-2 minutes on first run or slower USB drives."
        Add-StatusLine -Ui $ui -LogPath $StatusLog -Message ("selected model path = " + $selectedModelPath)
        Add-StatusLine -Ui $ui -LogPath $StatusLog -Message ("selected model size = " + $selectedModelGiB + " GiB")
    }
    Add-StatusLine -Ui $ui -LogPath $StatusLog -Message ("laptop build = " + $laptopBuildFlag.ToString().ToLowerInvariant())
    Add-StatusLine -Ui $ui -LogPath $StatusLog -Message ("expected RAM class = " + $expectedRamClass)
    if ($localModelDisabled -and -not $disableModelByEnv) {
        $env:HJ_DISABLE_MODEL = "1"
    }
    if ($laptopBuildFlag) {
        $env:HJ_LAPTOP_BUILD = "1"
    }

    $env:HALAL_JORDAN_STATUS_LOG_PATH = $StatusLog
    $env:HALAL_JORDAN_STARTUP_PROFILE_PATH = $StartupProfilePath
    $launcherProcess = Start-HiddenPowerShellFile -FilePath $LauncherPath -Arguments @("-NoBrowser", "-SkipHealthWait") -WorkingDirectory $Root -StdOutLog $LauncherOutLog -StdErrLog $LauncherErrLog
    $launcherStartedAt = Get-Date
    Write-StartupProfileEvent -Path $StartupProfilePath -Event "launcher_process_started" -Details @{
        pid = $launcherProcess.Id
        elapsed_ms = [int](($launcherStartedAt - $startupStartedAt).TotalMilliseconds)
    }
    $backendProcessId = 0
    $launchState = $null
    $launchStateDeadline = (Get-Date).AddSeconds(5)
    while ((Get-Date) -lt $launchStateDeadline) {
        $launchState = Read-LaunchState -Path $LaunchStatePath
        if ($launchState -and $launchState.pid) {
            $backendProcessId = [int]$launchState.pid
            break
        }
        if (-not (Get-Process -Id $launcherProcess.Id -ErrorAction SilentlyContinue)) {
            break
        }
        Start-Sleep -Milliseconds 200
    }
    if ($launchState) {
        if ($launchState.python) {
            Add-StatusLine -Ui $ui -LogPath $StatusLog -Message ("Python executable path = " + [string]$launchState.python)
        }
        if ($launchState.working_directory) {
            Add-StatusLine -Ui $ui -LogPath $StatusLog -Message ("Working directory = " + [string]$launchState.working_directory)
        }
        if ($launchState.backend_command_line) {
            Add-StatusLine -Ui $ui -LogPath $StatusLog -Message ("Backend command = " + [string]$launchState.backend_command_line)
        }
        if ($launchState.port) {
            Add-StatusLine -Ui $ui -LogPath $StatusLog -Message ("Selected port = " + [string]$launchState.port)
        }
    }
    if (-not $backendProcessId -and -not (Get-Process -Id $launcherProcess.Id -ErrorAction SilentlyContinue)) {
        throw "Launcher exited before backend process details were written."
    }
    $selectedBaseUrl = if ($launchState -and $launchState.url) { ([string]$launchState.url).TrimEnd("/") } else { "http://127.0.0.1:8000" }
    $healthUri = $selectedBaseUrl + "/health"
    if ($localModelDisabled) {
        Add-StatusLine -Ui $ui -LogPath $StatusLog -Message "Initializing retrieval-first backend..."
        Add-StatusLine -Ui $ui -LogPath $StatusLog -Message "Opening browser when backend is ready..."
        Add-StatusLine -Ui $ui -LogPath $StatusLog -Message ("Waiting for backend health up to " + $backendHealthTimeoutSeconds + " seconds")
    }
    else {
        Add-StatusLine -Ui $ui -LogPath $StatusLog -Message "Initializing engine..."
    }

    $healthStatusPrefix = if ($localModelDisabled) { "Waiting for backend health:" } else { "Waiting for model server:" }
    $healthTimeoutMessage = if ($localModelDisabled) {
        "Backend did not become ready within " + $backendHealthTimeoutSeconds + " seconds."
    }
    else {
        "Local model did not become ready within " + $backendHealthTimeoutSeconds + " seconds. This may be due to slow hardware, insufficient RAM, antivirus scanning, or USB drive speed. Try again after closing other apps."
    }
    $healthPollIntervalMilliseconds = if ($localModelDisabled) { 2000 } else { 1000 }
    $healthWaitProcessId = if ($backendProcessId -gt 0) { $backendProcessId } else { 0 }
    $health = Wait-ForHttpJson -Uri $healthUri -TimeoutSeconds $backendHealthTimeoutSeconds -Ui $ui -LogPath $StatusLog -ProcessId $healthWaitProcessId -StatusMessagePrefix $healthStatusPrefix -PollIntervalMilliseconds $healthPollIntervalMilliseconds -TimeoutMessage $healthTimeoutMessage -ExitFailureLabel "Backend process"
    if (-not $health.status) {
        throw "Health endpoint did not return a status."
    }

    $healthReadyAt = Get-Date
    Write-StartupProfileEvent -Path $StartupProfilePath -Event "health_ready" -Details @{
        elapsed_ms = [int](($healthReadyAt - $startupStartedAt).TotalMilliseconds)
        status = [string]$health.status
        retrieval_warmup_status = [string]$health.runtime_health.retrieval_warmup_status
        prepared_search_loaded = [bool]$health.runtime_health.prepared_search_loaded
        retrieval_bootstrap_source = [string]$health.runtime_health.retrieval_bootstrap_source
    }
    Add-StatusLine -Ui $ui -LogPath $StatusLog -Message "Ready"
    if (-not (Test-AutomatedUiMode)) {
        Add-StatusLine -Ui $ui -LogPath $StatusLog -Message "Opening browser..."
        Start-Process ($selectedBaseUrl + "/")
        Write-StartupProfileEvent -Path $StartupProfilePath -Event "browser_opened" -Details @{
            elapsed_ms = [int](($((Get-Date)) - $startupStartedAt).TotalMilliseconds)
            url = $selectedBaseUrl + "/"
        }
    }
    Start-Sleep -Milliseconds 700
    Close-StatusWindow -Ui $ui
}
catch {
    $reason = $_.Exception.Message
    $reason = Get-BackendFailureDiagnostics -Reason $reason
    Write-StartupProfileEvent -Path $StartupProfilePath -Event "startup_failed" -Details @{
        reason = $reason
        elapsed_ms = [int](($((Get-Date)) - $startupStartedAt).TotalMilliseconds)
    }
    Add-StatusLine -Ui $ui -LogPath $StatusLog -Message $reason
    Show-FailureDialog -Ui $ui -Reason $reason
    if ($ui) {
        Close-StatusWindow -Ui $ui
    }
    exit 1
}
finally {
    Remove-Item Env:HALAL_JORDAN_STATUS_LOG_PATH -ErrorAction SilentlyContinue
    Remove-Item Env:HALAL_JORDAN_STARTUP_PROFILE_PATH -ErrorAction SilentlyContinue
}
