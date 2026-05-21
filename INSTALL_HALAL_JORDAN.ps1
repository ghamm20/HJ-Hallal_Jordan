param()

$ErrorActionPreference = "Stop"

function Test-TruthyEnv {
    param([string]$Name)

    $value = [string](Get-Item -Path ("Env:" + $Name) -ErrorAction SilentlyContinue).Value
    return $value -match '^(1|true|yes|on)$'
}

$Root = $PSScriptRoot
Set-Location $Root
. (Join-Path $Root "runtime\launcher\HalalJordanExeCommon.ps1")

$LogDir = Join-Path $Root "logs"
$RuntimeDir = Join-Path $Root "runtime"
$StatusLog = Join-Path $LogDir "install-halal-jordan.log"
$LauncherOutLog = Join-Path $LogDir "install-launcher.out.log"
$LauncherErrLog = Join-Path $LogDir "install-launcher.err.log"
$LauncherPath = Join-Path $Root "LAUNCH_HALAL_JORDAN.ps1"
$StopPath = Join-Path $Root "STOP_HALAL_JORDAN.ps1"
$PythonPath = Join-Path $Root "runtime\python\python.exe"
$LlamaServerPath = Join-Path $Root "runtime\llama\llama-server.exe"
$modelReadyTimeoutSeconds = Get-PositiveIntFromEnv -Name "HJ_MODEL_READY_TIMEOUT_SECONDS" -DefaultValue 120
$configPath = Join-Path $Root "config\\runtime_config.json"
$configObject = if (Test-Path $configPath) { Get-Content $configPath -Raw | ConvertFrom-Json } else { $null }
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
$ui = $null

try {
    $ui = New-StatusWindow -Title "Halal Jordan Installation" -InitialMessage "Installing Halal Jordan..."
    Add-StatusLine -Ui $ui -LogPath $StatusLog -Message "Installing Halal Jordan..."

    if ($env:OS -ne "Windows_NT") {
        throw "This portable installer supports Windows only."
    }
    Add-StatusLine -Ui $ui -LogPath $StatusLog -Message "Windows OS detected."

    $totalRamBytes = [int64](Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory
    $totalRamGb = [math]::Round($totalRamBytes / 1GB, 1)
    if ($totalRamBytes -lt 8GB) {
        throw ("At least 8 GB RAM is required. Detected: " + $totalRamGb + " GB.")
    }
    if ($totalRamBytes -lt 16GB) {
        Add-StatusLine -Ui $ui -LogPath $StatusLog -Message ("RAM warning: " + $totalRamGb + " GB detected. 16 GB or more is recommended.")
    }
    else {
        Add-StatusLine -Ui $ui -LogPath $StatusLog -Message ("RAM check passed: " + $totalRamGb + " GB detected.")
    }

    $driveRoot = [System.IO.Path]::GetPathRoot($Root)
    $driveInfo = New-Object System.IO.DriveInfo($driveRoot)
    if (-not $driveInfo.IsReady) {
        throw ("Portable media is not accessible: " + $driveRoot)
    }
    Add-StatusLine -Ui $ui -LogPath $StatusLog -Message ("Portable media accessible: " + $driveRoot + " (" + $driveInfo.DriveType + ")")

    $requiredPaths = @($PythonPath)
    if (-not $localModelDisabled) {
        $requiredPaths += @($selectedModelPath, $LlamaServerPath)
    }
    foreach ($requiredPath in $requiredPaths) {
        if (-not (Test-Path $requiredPath)) {
            throw ("Required file is missing: " + $requiredPath)
        }
        Add-StatusLine -Ui $ui -LogPath $StatusLog -Message ("Verified: " + $requiredPath)
    }

    if (Test-Path $StopPath) {
        try {
            & powershell.exe -NoLogo -ExecutionPolicy Bypass -File $StopPath | Out-Null
        }
        catch {
        }
    }
    Stop-ProcessesOnPort -Port 8000 -Ui $ui -LogPath $StatusLog

    Add-StatusLine -Ui $ui -LogPath $StatusLog -Message "Launching backend for installation check..."
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
    $launcherProcess = Start-HiddenPowerShellFile -FilePath $LauncherPath -Arguments @("-NoBrowser") -WorkingDirectory $Root -StdOutLog $LauncherOutLog -StdErrLog $LauncherErrLog
    $healthStatusPrefix = if ($localModelDisabled) { "Waiting for backend:" } else { "Waiting for model server:" }
    $healthTimeoutMessage = if ($localModelDisabled) {
        "Backend did not become ready within " + $modelReadyTimeoutSeconds + " seconds. Retrieval or runtime startup may be blocked."
    }
    else {
        "Local model did not become ready within " + $modelReadyTimeoutSeconds + " seconds. This may be due to slow hardware, insufficient RAM, antivirus scanning, or USB drive speed. Try again after closing other apps."
    }
    $health = Wait-ForHttpJson -Uri "http://127.0.0.1:8000/health" -TimeoutSeconds $modelReadyTimeoutSeconds -Ui $ui -LogPath $StatusLog -StatusMessagePrefix $healthStatusPrefix -TimeoutMessage $healthTimeoutMessage
    if (-not $health.status) {
        throw "Health endpoint did not return a status."
    }
    Add-StatusLine -Ui $ui -LogPath $StatusLog -Message ("Health status: " + $health.status)

    $smoke = Invoke-ChatSmokeQuery -BaseUrl "http://127.0.0.1:8000" -Question "What does the Quran say about the straight path?" -Ui $ui -LogPath $StatusLog
    Add-StatusLine -Ui $ui -LogPath $StatusLog -Message ("Smoke query status: " + [string]$smoke.status)
    Add-StatusLine -Ui $ui -LogPath $StatusLog -Message "Installation complete"

    if (Test-Path $StopPath) {
        & powershell.exe -NoLogo -ExecutionPolicy Bypass -File $StopPath | Out-Null
    }

    Close-StatusWindow -Ui $ui
    Show-SuccessDialog -Message "Installation complete"
}
catch {
    if (Test-Path $StopPath) {
        try {
            & powershell.exe -NoLogo -ExecutionPolicy Bypass -File $StopPath | Out-Null
        }
        catch {
        }
    }
    $reason = $_.Exception.Message
    $stderrTail = Get-LogTail -Path $LauncherErrLog -LineCount 10
    if ($stderrTail) {
        $reason = $reason + [Environment]::NewLine + $stderrTail
    }
    Show-FailureDialog -Ui $ui -Reason $reason
    if ($ui) {
        Close-StatusWindow -Ui $ui
    }
    exit 1
}
finally {
    Remove-Item Env:HALAL_JORDAN_STATUS_LOG_PATH -ErrorAction SilentlyContinue
}
