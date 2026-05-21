param(
    [int]$Port = 8000,
    [string]$BindHost = "127.0.0.1",
    [int]$HealthTimeoutSeconds = 15,
    [switch]$NoBrowser,
    [switch]$SkipHealthWait,
    [switch]$ServerOnly,
    [string]$LlamaHost = "127.0.0.1",
    [int]$LlamaPort = 11435
)

$ErrorActionPreference = "Stop"

$Root = $PSScriptRoot
Set-Location $Root
$RuntimeDir = Join-Path $Root "runtime"
$LogsDir = Join-Path $Root "logs"
$StatePath = Join-Path $RuntimeDir "halal-jordan-launch-state.json"
$LlamaStatePath = Join-Path $RuntimeDir "halal-jordan-llama-state.json"
$ConfigPath = Join-Path $Root "config\runtime_config.json"
$PythonPackagePath = Join-Path $Root "runtime\site-packages"
$LlamaRuntimeDir = Join-Path $Root "runtime\llama"
$LlamaServerPath = Join-Path $LlamaRuntimeDir "llama-server.exe"
$LlamaCliPath = Join-Path $LlamaRuntimeDir "llama-cli.exe"
$LlamaOutLog = Join-Path $LogsDir "halal-jordan-llama.out.log"
$LlamaErrLog = Join-Path $LogsDir "halal-jordan-llama.err.log"
$AppOutLog = Join-Path $LogsDir "halal-jordan-app.out.log"
$AppErrLog = Join-Path $LogsDir "halal-jordan-app.err.log"
$ServerLogPath = Join-Path $LogsDir "halal-jordan-launch.log"
$SharedStatusLogPath = [string]$env:HALAL_JORDAN_STATUS_LOG_PATH
$StartupProfilePath = [string]$env:HALAL_JORDAN_STARTUP_PROFILE_PATH
$SelectedPortSource = if ($PSBoundParameters.ContainsKey("Port")) { "parameter" } else { "default_8000" }
if (-not $PSBoundParameters.ContainsKey("Port")) {
    $configuredPortOverride = [string]$env:HJ_PORT
    $configuredPortOverride = $configuredPortOverride.Trim()
    if ($configuredPortOverride) {
        $parsedPort = 0
        if (-not [int]::TryParse($configuredPortOverride, [ref]$parsedPort) -or $parsedPort -le 0) {
            throw "HJ_PORT must be a positive integer. Current value: $configuredPortOverride"
        }
        $Port = $parsedPort
        $SelectedPortSource = "env:HJ_PORT"
    }
}

function Write-LaunchLine {
    param(
        [string]$Message,
        [string]$Color = "Gray"
    )
    Write-Host $Message -ForegroundColor $Color
}

function Write-LaunchStatusLine {
    param(
        [string]$Message,
        [string]$Color = "Gray"
    )

    Write-LaunchLine -Message $Message -Color $Color
    foreach ($path in @($ServerLogPath, $SharedStatusLogPath)) {
        if ([string]::IsNullOrWhiteSpace($path)) {
            continue
        }
        $logDir = Split-Path -Parent $path
        if ($logDir) {
            New-Item -ItemType Directory -Path $logDir -Force | Out-Null
        }
        Add-Content -Path $path -Value ("[{0}] {1}" -f (Get-Date).ToString("s"), $Message)
    }
}

function Write-StartupProfileEvent {
    param(
        [string]$Event,
        [hashtable]$Details = @{}
    )

    if ([string]::IsNullOrWhiteSpace($StartupProfilePath)) {
        return
    }

    $payload = @{
        timestamp = (Get-Date).ToString("o")
        event = $Event
    }
    foreach ($entry in $Details.GetEnumerator()) {
        $payload[$entry.Key] = $entry.Value
    }

    $profileDir = Split-Path -Parent $StartupProfilePath
    if ($profileDir) {
        New-Item -ItemType Directory -Path $profileDir -Force | Out-Null
    }
    Add-Content -Path $StartupProfilePath -Value ($payload | ConvertTo-Json -Compress)
}

function Format-CommandLine {
    param([string[]]$Tokens)

    return [string]::Join(" ", @($Tokens | ForEach-Object {
        $token = [string]$_
        if ($token -match '[\s"]') {
            '"' + $token.Replace('"', '\"') + '"'
        }
        else {
            $token
        }
    }))
}

function Get-PositiveIntFromEnv {
    param(
        [string]$Name,
        [int]$DefaultValue
    )

    $rawValue = [string](Get-Item -Path ("Env:" + $Name) -ErrorAction SilentlyContinue).Value
    if ([string]::IsNullOrWhiteSpace($rawValue)) {
        return $DefaultValue
    }

    $parsedValue = 0
    if (-not [int]::TryParse($rawValue.Trim(), [ref]$parsedValue) -or $parsedValue -le 0) {
        return $DefaultValue
    }
    return $parsedValue
}

function Test-EnvFlag {
    param([string]$Name)

    $rawValue = [string](Get-Item -Path ("Env:" + $Name) -ErrorAction SilentlyContinue).Value
    return $rawValue -match '^(1|true|yes|on)$'
}

function Get-ConfigObject {
    param([string]$Path)

    if (-not (Test-Path $Path)) {
        return [pscustomobject]@{}
    }
    $raw = Get-Content -Path $Path -Raw -Encoding UTF8
    if ([string]::IsNullOrWhiteSpace($raw)) {
        return [pscustomobject]@{}
    }
    return $raw | ConvertFrom-Json
}

function Get-PythonLaunchSpec {
    param([string]$ProjectRoot)

    $candidates = @(
        @{
            executable = (Join-Path $ProjectRoot "runtime\python\python.exe")
            prefix = @()
            label = "bundled runtime python"
            source = "bundled_runtime_python"
            host_fallback = $false
            bundled = $true
            package_path = (Join-Path $ProjectRoot "runtime\site-packages")
        },
        @{
            executable = (Join-Path $ProjectRoot "runtime\venv\Scripts\python.exe")
            prefix = @()
            label = "bundled runtime venv"
            source = "bundled_runtime_venv"
            host_fallback = $false
            bundled = $true
            package_path = ""
        },
        @{
            executable = (Join-Path $ProjectRoot ".venv\Scripts\python.exe")
            prefix = @()
            label = "project-local .venv"
            source = "project_local_dotvenv"
            host_fallback = $false
            bundled = $false
            package_path = ""
        }
    )

    foreach ($candidate in $candidates) {
        if (Test-Path $candidate.executable) {
            return $candidate
        }
    }

    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python) {
        return @{
            executable = $python.Source
            prefix = @()
            label = "python on PATH"
            source = "host_path_python"
            host_fallback = $true
            bundled = $false
            package_path = ""
        }
    }

    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) {
        return @{
            executable = $py.Source
            prefix = @("-3")
            label = "py launcher"
            source = "host_py_launcher"
            host_fallback = $true
            bundled = $false
            package_path = ""
        }
    }

    throw "Python was not found. Expected runtime\\python\\python.exe."
}

function Enable-BundledPackagePath {
    param([hashtable]$PythonSpec)

    if (-not $PythonSpec.package_path) {
        return $false
    }
    if (-not (Test-Path $PythonSpec.package_path)) {
        return $false
    }

    if ([string]::IsNullOrWhiteSpace($env:PYTHONPATH)) {
        $env:PYTHONPATH = $PythonSpec.package_path
    }
    else {
        $env:PYTHONPATH = $PythonSpec.package_path + [System.IO.Path]::PathSeparator + $env:PYTHONPATH
    }
    return $true
}

function Invoke-PythonText {
    param(
        [hashtable]$PythonSpec,
        [string[]]$Arguments
    )

    $output = & $PythonSpec.executable @($PythonSpec.prefix + $Arguments) 2>&1
    return [pscustomobject]@{
        Output = [string]::Join([Environment]::NewLine, @($output))
        ExitCode = $LASTEXITCODE
    }
}

function Get-ListeningProcessInfo {
    param([int]$Port)

    try {
        $connections = @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction Stop)
    }
    catch {
        return $null
    }
    if (-not $connections) {
        return $null
    }
    $connection = $connections | Select-Object -First 1
    $owningPid = [int]$connection.OwningProcess
    $processName = ""
    $path = ""
    try {
        $process = Get-Process -Id $owningPid -ErrorAction Stop
        $processName = [string]$process.ProcessName
    }
    catch {
    }
    try {
        $proc = Get-CimInstance Win32_Process -Filter ("ProcessId = " + $owningPid)
        $path = [string]$proc.ExecutablePath
    }
    catch {
    }
    return [pscustomobject]@{
        ProcessId = $owningPid
        ProcessName = $processName
        ExecutablePath = $path
    }
}

function Test-PortAvailable {
    param(
        [string]$BindHost,
        [int]$PortNumber
    )

    $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Parse($BindHost), $PortNumber)
    try {
        $listener.Start()
        return $true
    }
    catch {
        return $false
    }
    finally {
        try {
            $listener.Stop()
        }
        catch {
        }
    }
}

function Resolve-LaunchPort {
    param(
        [string]$BindHost,
        [int]$PreferredPort
    )

    if (Test-PortAvailable -BindHost $BindHost -PortNumber $PreferredPort) {
        return $PreferredPort
    }
    $occupant = Get-ListeningProcessInfo -Port $PreferredPort
    Write-LaunchStatusLine ("Port " + $PreferredPort + " is already in use. Stop the other app or set HJ_PORT.") "Red"
    if ($occupant) {
        Write-LaunchStatusLine ("Occupying PID: " + $occupant.ProcessId) "Yellow"
        if ($occupant.ProcessName) {
            Write-LaunchStatusLine ("Occupying process: " + $occupant.ProcessName) "Yellow"
        }
        if ($occupant.ExecutablePath) {
            Write-LaunchStatusLine ("Occupying executable: " + $occupant.ExecutablePath) "Yellow"
        }
        Write-LaunchStatusLine ("Stop command: Stop-Process -Id " + $occupant.ProcessId) "Yellow"
    }
    throw "Launch aborted because port $PreferredPort is already in use."
}

function Wait-ForHttpJson {
    param(
        [string]$Uri,
        [int]$TimeoutSeconds,
        [int]$ProcessId = 0
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        try {
            $response = Invoke-RestMethod -Uri $Uri -Method Get -TimeoutSec 5
            if ($null -ne $response) {
                return $response
            }
        }
        catch {
        }

        if ($ProcessId -gt 0 -and -not (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue)) {
            throw "Process $ProcessId exited before $Uri became ready."
        }
        Start-Sleep -Milliseconds 750
    }

    throw "Timed out waiting for $Uri."
}

function Wait-ForHealth {
    param(
        [string]$BaseUrl,
        [int]$TimeoutSeconds,
        [int]$ProcessId
    )

    $response = Wait-ForHttpJson -Uri ($BaseUrl.TrimEnd("/") + "/health") -TimeoutSeconds $TimeoutSeconds -ProcessId $ProcessId
    if (-not $response.status) {
        throw "Health endpoint did not return a status."
    }
    return $response
}

function Wait-ForLlamaServer {
    param(
        [string]$ServerUrl,
        [string]$ExpectedModel,
        [int]$TimeoutSeconds,
        [int]$ProcessId = 0,
        [string]$ModelPath = "",
        [string]$ServerPath = ""
    )

    $readyUri = $ServerUrl.TrimEnd("/") + "/v1/models"
    $startedAt = Get-Date
    $deadline = $startedAt.AddSeconds($TimeoutSeconds)
    $nextProgressAt = $startedAt
    $lastProbeError = ""

    Write-LaunchStatusLine ("Model start timestamp: " + $startedAt.ToString("o"))
    Write-LaunchStatusLine ("Model readiness timeout: " + $TimeoutSeconds + " seconds")
    if ($ModelPath) {
        Write-LaunchStatusLine ("Model path: " + $ModelPath)
    }
    if ($ServerPath) {
        Write-LaunchStatusLine ("llama server path: " + $ServerPath)
    }

    while ((Get-Date) -lt $deadline) {
        $now = Get-Date
        if ($now -ge $nextProgressAt) {
            $elapsedSeconds = [int][math]::Floor(($now - $startedAt).TotalSeconds)
            Write-LaunchStatusLine ("Waiting for model server: " + $elapsedSeconds + "/" + $TimeoutSeconds + " seconds")
            $nextProgressAt = $startedAt.AddSeconds($elapsedSeconds + 5)
        }

        try {
            $response = Invoke-RestMethod -Uri $readyUri -Method Get -TimeoutSec 5
            $modelIds = @($response.data | ForEach-Object { [string]$_.id })
            if (-not $modelIds) {
                throw "llama.cpp responded but did not expose any loaded models."
            }
            if ($ExpectedModel -and -not ($modelIds -contains $ExpectedModel)) {
                throw ("llama.cpp loaded an unexpected model. Visible models: " + ($modelIds -join ", "))
            }

            $readyAt = Get-Date
            $elapsedSeconds = [math]::Round(($readyAt - $startedAt).TotalSeconds, 1)
            Write-LaunchStatusLine ("Model ready timestamp: " + $readyAt.ToString("o"))
            Write-LaunchStatusLine ("Model ready after " + $elapsedSeconds + " seconds")
            return $response
        }
        catch {
            $lastProbeError = $_.Exception.Message
        }

        if ($ProcessId -gt 0 -and -not (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue)) {
            $elapsedSeconds = [math]::Round(((Get-Date) - $startedAt).TotalSeconds, 1)
            $crashMessage = "Local model server exited before " + $readyUri + " became ready after " + $elapsedSeconds + " seconds."
            if ($lastProbeError) {
                $crashMessage += " Last probe error: " + $lastProbeError
            }
            throw $crashMessage
        }

        Start-Sleep -Milliseconds 1000
    }

    $timeoutMessage = "Local model did not become ready within " + $TimeoutSeconds + " seconds. This may be due to slow hardware, insufficient RAM, antivirus scanning, or USB drive speed. Try again after closing other apps."
    Write-LaunchStatusLine $timeoutMessage "Yellow"
    throw $timeoutMessage
}

function Write-LauncherState {
    param(
        [string]$Path,
        [hashtable]$Payload
    )

    $Payload | ConvertTo-Json -Depth 8 | Set-Content -Path $Path -Encoding UTF8
}

function Start-BundledLlamaServer {
    param(
        [string]$ServerPath,
        [string]$WorkingDirectory,
        [string]$ModelPath,
        [string]$ModelAlias,
        [string]$BindHost,
        [int]$PortNumber
    )

    $arguments = @(
        "-m", $ModelPath,
        "--alias", $ModelAlias,
        "--host", $BindHost,
        "--port", [string]$PortNumber,
        "-c", "12288",
        "-t", "6",
        "-b", "256",
        "-ub", "256",
        "--reasoning", "off",
        "--no-webui",
        "--jinja",
        "--offline"
    )

    return Start-Process -FilePath $ServerPath -ArgumentList $arguments -WorkingDirectory $WorkingDirectory -RedirectStandardOutput $LlamaOutLog -RedirectStandardError $LlamaErrLog -WindowStyle Hidden -PassThru
}

function Get-LlamaRuntimeFacts {
    param(
        [string]$ProjectRoot,
        [object]$ConfigObject
    )

    $configuredModelPath = [string]$ConfigObject.main_model_path
    if ([string]::IsNullOrWhiteSpace($configuredModelPath)) {
        $configuredModelPath = "models/qwen3-4b-laptop.gguf"
    }
    $resolvedModelPath = if ([System.IO.Path]::IsPathRooted($configuredModelPath)) {
        $configuredModelPath
    }
    else {
        Join-Path $ProjectRoot $configuredModelPath
    }
    $modelName = [string]$ConfigObject.main_model_name
    if ([string]::IsNullOrWhiteSpace($modelName)) {
        $modelName = [System.IO.Path]::GetFileNameWithoutExtension($resolvedModelPath)
    }
    return @{
        model_path = $resolvedModelPath
        model_name = $modelName
        model_enabled = [bool]$ConfigObject.main_model_enabled
    }
}

function Set-BackendEnvironment {
    param(
        [string]$ProjectRoot,
        [string]$SelectedHost,
        [int]$SelectedPort,
        [string]$SelectedBaseUrl,
        [string]$SelectedLlamaHost,
        [int]$SelectedLlamaPort,
        [string]$SelectedLlamaUrl
    )

    $env:HALAL_JORDAN_PORTABLE_MODE = "1"
    $env:HALAL_JORDAN_PORTABLE_ROOT = $ProjectRoot
    $env:HALAL_JORDAN_SELECTED_HOST = $SelectedHost
    $env:HALAL_JORDAN_SELECTED_PORT = [string]$SelectedPort
    $env:HALAL_JORDAN_SELECTED_URL = $SelectedBaseUrl + "/"
    $env:HALAL_JORDAN_LLAMACPP_HOST = $SelectedLlamaHost
    $env:HALAL_JORDAN_LLAMACPP_PORT = [string]$SelectedLlamaPort
    $env:HALAL_JORDAN_LLAMACPP_URL = $SelectedLlamaUrl
    $env:LLAMA_OFFLINE = "1"
    $env:PYTHONUNBUFFERED = "1"
}

function Start-BackendPythonProcess {
    param(
        [hashtable]$PythonSpec,
        [string]$ProjectRoot,
        [string]$SelectedHost,
        [int]$SelectedPort
    )

    $arguments = @($PythonSpec.prefix + @(
        "-m",
        "uvicorn",
        "app.backend.main:app",
        "--host",
        $SelectedHost,
        "--port",
        [string]$SelectedPort
    ))
    $commandLine = Format-CommandLine -Tokens (@([string]$PythonSpec.executable) + @($arguments | ForEach-Object { [string]$_ }))
    Write-LaunchStatusLine ("python executable path = " + $PythonSpec.executable)
    Write-LaunchStatusLine ("working directory = " + $ProjectRoot)
    Write-LaunchStatusLine ("selected port = " + $SelectedPort)
    Write-LaunchStatusLine ("backend command = " + $commandLine)
    Write-LaunchStatusLine ("backend stdout log = " + $AppOutLog)
    Write-LaunchStatusLine ("backend stderr log = " + $AppErrLog)
    Add-Content -Path $ServerLogPath -Value ("[{0}] Backend python start requested on {1}:{2}" -f (Get-Date).ToString("s"), $SelectedHost, $SelectedPort)
    $process = Start-Process -FilePath $PythonSpec.executable -ArgumentList $arguments -WorkingDirectory $ProjectRoot -RedirectStandardOutput $AppOutLog -RedirectStandardError $AppErrLog -WindowStyle Hidden -PassThru
    return @{
        Process = $process
        CommandArguments = @($arguments | ForEach-Object { [string]$_ })
        CommandLine = $commandLine
    }
}

New-Item -ItemType Directory -Path $RuntimeDir -Force | Out-Null
New-Item -ItemType Directory -Path $LogsDir -Force | Out-Null
Set-Location $Root
Write-StartupProfileEvent -Event "launcher_invoked" -Details @{
    root = $Root
    skip_health_wait = [bool]$SkipHealthWait
    server_only = [bool]$ServerOnly
}

$configObject = Get-ConfigObject -Path $ConfigPath
$llamaFacts = Get-LlamaRuntimeFacts -ProjectRoot $Root -ConfigObject $configObject
$LlamaServerUrl = "http://{0}:{1}" -f $LlamaHost, $LlamaPort
$mainModelProvider = if ($configObject -and $configObject.main_model_provider) { [string]$configObject.main_model_provider } else { "local_gguf" }
$disableModelByEnv = Test-EnvFlag -Name "HJ_DISABLE_MODEL"
$configuredModelReadyTimeoutSeconds = 120
if ($null -ne $configObject.launcher_model_ready_timeout_seconds) {
    $configuredTimeoutCandidate = 0
    if ([int]::TryParse([string]$configObject.launcher_model_ready_timeout_seconds, [ref]$configuredTimeoutCandidate) -and $configuredTimeoutCandidate -gt 0) {
        $configuredModelReadyTimeoutSeconds = $configuredTimeoutCandidate
    }
}
$modelReadyTimeoutSeconds = Get-PositiveIntFromEnv -Name "HJ_MODEL_READY_TIMEOUT_SECONDS" -DefaultValue $configuredModelReadyTimeoutSeconds
$laptopBuildFlag = if ($null -ne $configObject.laptop_build) { [bool]$configObject.laptop_build } else { (Test-EnvFlag -Name "HJ_LAPTOP_BUILD") }
if (-not $laptopBuildFlag -and $disableModelByEnv) {
    $laptopBuildFlag = $true
}
$skipBundledLlamaStartup = $disableModelByEnv -or (-not $llamaFacts.model_enabled) -or ($mainModelProvider -ne "local_gguf")
$modelDisabled = $disableModelByEnv -or (-not $llamaFacts.model_enabled)
$modelSkipReason = if ($disableModelByEnv) {
    "HJ_DISABLE_MODEL"
}
elseif (-not $llamaFacts.model_enabled) {
    "config.main_model_enabled=false"
}
else {
    "main_model_provider=" + $mainModelProvider
}
$expectedRamClass = [string]$configObject.expected_ram_class
if ([string]::IsNullOrWhiteSpace($expectedRamClass)) {
    $expectedRamClass = "8-16GB"
}

if ($ServerOnly) {
    $pythonSpec = Get-PythonLaunchSpec -ProjectRoot $Root
    Enable-BundledPackagePath -PythonSpec $pythonSpec | Out-Null
    Set-BackendEnvironment -ProjectRoot $Root -SelectedHost $BindHost -SelectedPort $Port -SelectedBaseUrl ("http://{0}:{1}" -f $BindHost, $Port) -SelectedLlamaHost $LlamaHost -SelectedLlamaPort $LlamaPort -SelectedLlamaUrl $LlamaServerUrl
    Add-Content -Path $ServerLogPath -Value ("[{0}] Server start requested on {1}:{2}" -f (Get-Date).ToString("s"), $BindHost, $Port)
    Write-StartupProfileEvent -Event "backend_python_exec_started" -Details @{
        host = $BindHost
        port = $Port
        python = $pythonSpec.executable
        source = $pythonSpec.source
    }
    & $pythonSpec.executable @($pythonSpec.prefix + @(
        "-m",
        "uvicorn",
        "app.backend.main:app",
        "--host",
        $BindHost,
        "--port",
        [string]$Port
    ))
    exit $LASTEXITCODE
}

Write-LaunchLine "Starting system..." "Cyan"
Write-LaunchLine ("Project root: " + $Root)
Write-LaunchLine ("Requested launch port: " + $Port + " (" + $SelectedPortSource + ")")

$pythonSpec = Get-PythonLaunchSpec -ProjectRoot $Root
$bundledPackagePathEnabled = Enable-BundledPackagePath -PythonSpec $pythonSpec
$env:HALAL_JORDAN_PORTABLE_MODE = "1"
$env:HALAL_JORDAN_PORTABLE_ROOT = $Root

$packageProbe = Invoke-PythonText -PythonSpec $pythonSpec -Arguments @(
    "-c",
    "import importlib.util, json; required=('fastapi','uvicorn','pydantic','requests','pypdf'); missing=[name for name in required if importlib.util.find_spec(name) is None]; print(json.dumps({'missing': missing}))"
)
if ($packageProbe.ExitCode -ne 0) {
    throw ("Unable to verify Python package dependencies. " + $packageProbe.Output)
}
$packageStatus = $packageProbe.Output | ConvertFrom-Json
if ($packageStatus.missing.Count -gt 0) {
    throw ("Missing Python packages: " + ($packageStatus.missing -join ", "))
}
Write-StartupProfileEvent -Event "launcher_runtime_verified" -Details @{
    python = $pythonSpec.executable
    python_source = $pythonSpec.source
    host_fallback_used = [bool]$pythonSpec.host_fallback
    bundled_package_path_enabled = [bool]$bundledPackagePathEnabled
}
Write-LaunchLine "[OK] Runtime loaded" "Green"
Write-LaunchLine ("Python: " + $pythonSpec.executable + " (" + $pythonSpec.label + ")")
Write-LaunchLine ("Runtime source: " + $pythonSpec.source)
Write-LaunchLine ("Host fallback used: " + $(if ($pythonSpec.host_fallback) { "yes" } else { "no" }))
if ($bundledPackagePathEnabled) {
    Write-LaunchLine ("Bundled package path enabled: " + $pythonSpec.package_path)
}

Write-LaunchLine "Loading knowledge..." "Cyan"
Write-LaunchStatusLine ("laptop build = " + $laptopBuildFlag.ToString().ToLowerInvariant())
Write-LaunchStatusLine ("expected RAM class = " + $expectedRamClass)
$llamaProcess = $null
if ($skipBundledLlamaStartup) {
    $statusMessage = if ($modelDisabled) {
        "Local model disabled for laptop build."
    }
    else {
        "Skipping bundled local model startup because main model provider is " + $mainModelProvider + "."
    }
    Write-LaunchLine $statusMessage "Yellow"
    Write-LaunchLine "Running retrieval-first fast_mode only." "Yellow"
    Write-LaunchStatusLine $statusMessage "Yellow"
    Write-LaunchStatusLine "Running retrieval-first fast_mode only."
    Write-LaunchStatusLine "final generation unavailable = true"
    Write-LaunchStatusLine ("model disable reason = " + $modelSkipReason)
    Write-LauncherState -Path $LlamaStatePath -Payload @{
        pid = 0
        url = $LlamaServerUrl
        model_name = $llamaFacts.model_name
        model_path = $llamaFacts.model_path
        server_path = $LlamaServerPath
        started_at = (Get-Date).ToString("o")
        disabled = $true
        reason = $modelSkipReason
    }
}
else {
    if (-not (Test-Path $LlamaServerPath)) {
        throw ("Bundled llama.cpp server is missing at " + $LlamaServerPath)
    }
    if (-not (Test-Path $LlamaCliPath)) {
        throw ("Bundled llama.cpp CLI is missing at " + $LlamaCliPath)
    }
    if (-not (Test-Path $llamaFacts.model_path)) {
        throw ("Main model is missing at " + $llamaFacts.model_path)
    }
    Write-LaunchLine ("Main model path: " + $llamaFacts.model_path) "Green"
    Write-LaunchLine ("llama.cpp server: " + $LlamaServerPath) "Green"
    Write-LaunchLine ("llama.cpp URL: " + $LlamaServerUrl)
    Write-LaunchStatusLine "Loading local model..." "Cyan"
    Write-LaunchStatusLine "This can take 1-2 minutes on first run or slower USB drives."
    $selectedModelBytes = (Get-Item $llamaFacts.model_path).Length
    $selectedModelGiB = [math]::Round($selectedModelBytes / 1GB, 2)
    Write-LaunchStatusLine ("selected model path = " + $llamaFacts.model_path)
    Write-LaunchStatusLine ("selected model size = " + $selectedModelGiB + " GiB")

    $llamaListener = Get-ListeningProcessInfo -Port $LlamaPort
    if ($llamaListener) {
        try {
            $models = Wait-ForLlamaServer -ServerUrl $LlamaServerUrl -ExpectedModel $llamaFacts.model_name -TimeoutSeconds $modelReadyTimeoutSeconds -ProcessId $llamaListener.ProcessId -ModelPath $llamaFacts.model_path -ServerPath $LlamaServerPath
            $llamaProcess = Get-Process -Id $llamaListener.ProcessId -ErrorAction SilentlyContinue
            Write-LaunchLine "[OK] Local engine already running" "Green"
        }
        catch {
            throw ("Port " + $LlamaPort + " is occupied, but the local engine was not healthy: " + $_.Exception.Message)
        }
    }
    else {
        Write-LaunchLine "Initializing engine..." "Cyan"
        $llamaProcess = Start-BundledLlamaServer -ServerPath $LlamaServerPath -WorkingDirectory $LlamaRuntimeDir -ModelPath $llamaFacts.model_path -ModelAlias $llamaFacts.model_name -BindHost $LlamaHost -PortNumber $LlamaPort
        Wait-ForLlamaServer -ServerUrl $LlamaServerUrl -ExpectedModel $llamaFacts.model_name -TimeoutSeconds $modelReadyTimeoutSeconds -ProcessId $llamaProcess.Id -ModelPath $llamaFacts.model_path -ServerPath $LlamaServerPath | Out-Null
        Write-LaunchLine "[OK] Model ready" "Green"
    }

    Write-LauncherState -Path $LlamaStatePath -Payload @{
        pid = $(if ($llamaProcess) { $llamaProcess.Id } else { 0 })
        url = $LlamaServerUrl
        model_name = $llamaFacts.model_name
        model_path = $llamaFacts.model_path
        server_path = $LlamaServerPath
        started_at = (Get-Date).ToString("o")
        disabled = $false
        reason = ""
    }
}

$selectedPort = Resolve-LaunchPort -BindHost $BindHost -PreferredPort $Port
$baseUrl = "http://{0}:{1}" -f $BindHost, $selectedPort
Set-BackendEnvironment -ProjectRoot $Root -SelectedHost $BindHost -SelectedPort $selectedPort -SelectedBaseUrl $baseUrl -SelectedLlamaHost $LlamaHost -SelectedLlamaPort $LlamaPort -SelectedLlamaUrl $LlamaServerUrl
Write-StartupProfileEvent -Event "launcher_selected_port" -Details @{
    host = $BindHost
    port = $selectedPort
    port_source = $SelectedPortSource
}

$existingState = $null
if (Test-Path $StatePath) {
    try {
        $existingState = Get-Content -Raw $StatePath | ConvertFrom-Json
    }
    catch {
        Remove-Item $StatePath -ErrorAction SilentlyContinue
    }
}
if ($existingState -and $existingState.pid) {
    $running = Get-Process -Id ([int]$existingState.pid) -ErrorAction SilentlyContinue
    if ($running) {
        Write-LaunchLine ("A previous Halal Jordan server window is still running (PID " + $existingState.pid + ").") "Yellow"
    }
}

$backendLaunch = Start-BackendPythonProcess -PythonSpec $pythonSpec -ProjectRoot $Root -SelectedHost $BindHost -SelectedPort $selectedPort
$serverProcess = $backendLaunch.Process
Write-StartupProfileEvent -Event "backend_process_started" -Details @{
    pid = $serverProcess.Id
    python = $pythonSpec.executable
    host = $BindHost
    port = $selectedPort
    command_line = $backendLaunch.CommandLine
}

Write-LauncherState -Path $StatePath -Payload @{
    pid = $serverProcess.Id
    app_pid = $serverProcess.Id
    host = $BindHost
    port = $selectedPort
    url = $baseUrl + "/"
    admin_url = $baseUrl + "/admin"
    python = $pythonSpec.executable
    python_label = $pythonSpec.label
    python_source = $pythonSpec.source
    host_fallback_used = $pythonSpec.host_fallback
    bundled_package_path = $pythonSpec.package_path
    working_directory = $Root
    backend_command = $backendLaunch.CommandArguments
    backend_command_line = $backendLaunch.CommandLine
    backend_stdout_log = $AppOutLog
    backend_stderr_log = $AppErrLog
    started_at = (Get-Date).ToString("o")
    selected_port_source = $SelectedPortSource
    llama_pid = $(if ($llamaProcess) { $llamaProcess.Id } else { 0 })
    llama_path = $LlamaServerPath
    llama_url = $LlamaServerUrl
    llama_model_name = $llamaFacts.model_name
    llama_model_path = $llamaFacts.model_path
    llama_disabled = $skipBundledLlamaStartup
    llama_disable_reason = $modelSkipReason
}

if ($SkipHealthWait) {
    Write-LaunchStatusLine ("Backend process started on " + $baseUrl + "/") "Green"
    Write-LaunchStatusLine "Skipping internal health wait; outer launcher is managing readiness." "Gray"
    Write-StartupProfileEvent -Event "launcher_health_wait_skipped" -Details @{
        pid = $serverProcess.Id
        url = $baseUrl + "/"
    }
    exit 0
}

Write-LaunchLine ("Waiting for health at " + $baseUrl + "/health ...")
$health = Wait-ForHealth -BaseUrl $baseUrl -TimeoutSeconds $HealthTimeoutSeconds -ProcessId $serverProcess.Id
Write-StartupProfileEvent -Event "launcher_health_ready" -Details @{
    pid = $serverProcess.Id
    status = [string]$health.status
    url = $baseUrl + "/"
}
Write-LaunchLine ("Health status: " + $health.status) "Green"
Write-LaunchLine "[OK] Backend running" "Green"

$appListener = Get-ListeningProcessInfo -Port $selectedPort
if ($appListener) {
    $updatedState = Get-Content -Raw $StatePath | ConvertFrom-Json
    $updatedState.app_pid = [int]$appListener.ProcessId
    $updatedState | ConvertTo-Json -Depth 8 | Set-Content -Path $StatePath -Encoding UTF8
}

Write-LaunchLine ("Selected app URL: " + $baseUrl + "/") "Green"
Write-LaunchLine "Ready" "Green"
Write-LaunchLine ("[READY] " + $baseUrl + "/") "Green"

if (-not $NoBrowser) {
    Start-Process ($baseUrl + "/")
}
