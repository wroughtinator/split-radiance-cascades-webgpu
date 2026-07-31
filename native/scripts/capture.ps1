param(
    [string]$TargetDir = (Join-Path $env:LOCALAPPDATA "split-rc-native-target"),
    [string]$AssetRoot = "..\public\models",
    [string]$OutputDir = (Join-Path (Split-Path $PSScriptRoot -Parent) "captures"),
    [string]$VcVars = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
)

$ErrorActionPreference = "Stop"
& "$PSScriptRoot\build.ps1" -TargetDir $TargetDir -VcVars $VcVars

$env:SPLIT_RC_ASSETS = (Resolve-Path -LiteralPath $AssetRoot).Path
$env:SPLIT_RC_CAPTURE_DIR = $OutputDir
$env:SPLIT_RC_RESOLUTION_SCALE = "1"
$env:SPLIT_RC_FIXED_SCALE = "1"
Remove-Item Env:SPLIT_RC_BENCH_FRAMES -ErrorAction SilentlyContinue
Remove-Item Env:SPLIT_RC_BENCH_OUT -ErrorAction SilentlyContinue
Remove-Item Env:SPLIT_RC_STABILITY_OUT -ErrorAction SilentlyContinue
Remove-Item Env:SPLIT_RC_ANIMATION_RESPONSE_OUT -ErrorAction SilentlyContinue
Remove-Item Env:SPLIT_RC_GPU_PROFILE -ErrorAction SilentlyContinue
Remove-Item Env:SPLIT_RC_STABILITY_SCENE -ErrorAction SilentlyContinue
Remove-Item Env:SPLIT_RC_STABILITY_REFERENCE_FRAME -ErrorAction SilentlyContinue
Remove-Item Env:SPLIT_RC_CACHE_MOTION_OUT -ErrorAction SilentlyContinue
Remove-Item Env:SPLIT_RC_INDIRECT_ONLY -ErrorAction SilentlyContinue
Remove-Item Env:SPLIT_RC_TEMPORAL_JITTER -ErrorAction SilentlyContinue
Remove-Item Env:SPLIT_RC_ROUGH_SPECULAR -ErrorAction SilentlyContinue
Remove-Item Env:SPLIT_RC_FREEZE_LIGHTS -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$executable = Join-Path $TargetDir "release\split-rc-native.exe"
& $executable
if ($LASTEXITCODE -ne 0) {
    throw "Framebuffer capture failed with exit code $LASTEXITCODE"
}
$sheet = Join-Path $OutputDir "contact-sheet.png"
if (-not (Test-Path -LiteralPath $sheet)) {
    throw "Capture completed without producing $sheet"
}
Write-Host "Captured 12 scenes: $sheet"
