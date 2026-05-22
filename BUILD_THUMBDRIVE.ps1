<#
.SYNOPSIS
    Verify the project is ready to copy onto a USB thumbdrive for handoff.

.DESCRIPTION
    Performs the four readiness checks a thumbdrive build needs:

      1. Bundled Python interpreter present
      2. Bundled site-packages present
      3. Bundled HuggingFace embedding model present
      4. Persisted retrieval index present (fast cold start)

    Reports total size and prints copy / share instructions.

    Optional: with -CopyTo <path>, robocopies the project to a target
    drive, skipping logs/, .git/, __pycache__/, and other ephemeral
    junk that bloats the bundle.

    Examples:
      .\BUILD_THUMBDRIVE.ps1
      .\BUILD_THUMBDRIVE.ps1 -CopyTo "F:\HalalJordan"
#>
param(
    [string]$CopyTo = "",
    [switch]$Quiet
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
Set-Location $Root

function Say {
    param([string]$msg, [string]$color = "Gray")
    if (-not $Quiet) { Write-Host $msg -ForegroundColor $color }
}

Say ""
Say "Halal Jordan thumbdrive readiness check" "Cyan"
Say ("Source: " + $Root) "Gray"
Say ""

$checks = @(
    @{ label = "Bundled Python interpreter"; path = "runtime\python\python.exe"; required = $true },
    @{ label = "Bundled site-packages"; path = "runtime\site-packages"; required = $true },
    @{ label = "Bundled embedding model"; path = "runtime\huggingface\models--sentence-transformers--all-MiniLM-L6-v2"; required = $true },
    @{ label = "Persisted retrieval index"; path = "data\index\chunks.jsonl"; required = $true },
    @{ label = "Bundled llama runtime (for /workspace chat)"; path = "runtime\llama\llama-server.exe"; required = $false },
    @{ label = "Bundled LLM model file"; path = "models\micro\qwen3-4b\Qwen3-4B-Q4_K_M.gguf"; required = $false }
)

$missing = New-Object System.Collections.Generic.List[string]
$warnings = New-Object System.Collections.Generic.List[string]
foreach ($check in $checks) {
    $full = Join-Path $Root $check.path
    if (Test-Path $full) {
        Say ("  [OK]      " + $check.label) "Green"
    }
    elseif ($check.required) {
        Say ("  [MISSING] " + $check.label + " - REQUIRED") "Red"
        $missing.Add($check.label) | Out-Null
    }
    else {
        Say ("  [WARN]    " + $check.label + " - optional (semantic chat synthesis disabled)") "Yellow"
        $warnings.Add($check.label) | Out-Null
    }
}
Say ""

if ($missing.Count -gt 0) {
    Say "Not ready to share. Missing required assets:" "Red"
    foreach ($m in $missing) {
        Say ("  - " + $m) "Red"
    }
    exit 1
}

# Compute folder size, excluding ephemeral subdirs
$skipPatterns = @("\.git", "__pycache__", "\\logs\\", "\\flight_test_results", "\\\.pytest_cache")
$totalBytes = 0
Get-ChildItem -Path $Root -Recurse -File -ErrorAction SilentlyContinue | ForEach-Object {
    $relative = $_.FullName.Substring($Root.Length).Replace("\", "/")
    $skip = $false
    foreach ($pat in $skipPatterns) {
        if ($relative -match $pat) { $skip = $true; break }
    }
    if (-not $skip) { $totalBytes += $_.Length }
}
$totalGB = [math]::Round($totalBytes / 1GB, 2)
Say ("Bundle size (excluding logs / .git / caches): " + $totalGB + " GB") "Cyan"
if ($totalGB -gt 16) {
    Say "  Warning: bundle larger than 16 GB. Consider a larger thumbdrive." "Yellow"
}
Say ""

if ($CopyTo) {
    if (-not (Test-Path $CopyTo)) {
        New-Item -ItemType Directory -Path $CopyTo -Force | Out-Null
    }
    Say ("Copying to: " + $CopyTo) "Cyan"
    Say "(robocopy excludes logs/, .git/, __pycache__/, .pytest_cache/, flight_test_results.txt)" "Gray"
    $excludeDirs = @("logs", ".git", "__pycache__", ".pytest_cache")
    $excludeFiles = @("flight_test_results.txt", "_preflight_test.ps1", "chunks.jsonl.prev")
    $robocopyArgs = @(
        $Root,
        $CopyTo,
        "/MIR",
        "/XJ",
        "/R:2",
        "/W:2",
        "/MT:8",
        "/NFL",
        "/NDL"
    )
    $robocopyArgs += "/XD"
    $robocopyArgs += $excludeDirs
    $robocopyArgs += "/XF"
    $robocopyArgs += $excludeFiles
    & robocopy.exe @robocopyArgs
    $rc = $LASTEXITCODE
    Say ""
    if ($rc -le 7) {
        Say ("Copy complete. Test by running: " + (Join-Path $CopyTo "START HALAL JORDAN.cmd")) "Green"
    }
    else {
        Say ("Copy reported errors (robocopy exit " + $rc + "). Review the output above.") "Red"
        exit 1
    }
}
else {
    Say "To copy onto a thumbdrive:" "Cyan"
    Say "  .\BUILD_THUMBDRIVE.ps1 -CopyTo 'F:\HalalJordan'" "Gray"
    Say ""
    Say "Requirements for the target drive:" "Cyan"
    Say "  - exFAT or NTFS filesystem (NOT FAT32)" "Gray"
    Say ("  - At least " + [math]::Ceiling($totalGB + 0.5) + " GB free") "Gray"
    Say "  - USB 3.0 strongly recommended" "Gray"
    Say ""
    Say "After copying, test by double-clicking 'START HALAL JORDAN.cmd' on the target drive." "Cyan"
}
Say ""
exit 0
