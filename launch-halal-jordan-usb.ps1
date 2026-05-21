param(
    [int]$Port = 8000,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
Set-Location $Root

$launcherPath = Join-Path $Root "LAUNCH_HALAL_JORDAN.ps1"
$launcherArgs = @("-NoLogo", "-ExecutionPolicy", "Bypass", "-File", $launcherPath)
if ($PSBoundParameters.ContainsKey("Port")) {
    $launcherArgs += @("-Port", [string]$Port)
}
if ($NoBrowser) {
    $launcherArgs += "-NoBrowser"
    & powershell.exe @launcherArgs
}
else {
    & powershell.exe @launcherArgs
}
