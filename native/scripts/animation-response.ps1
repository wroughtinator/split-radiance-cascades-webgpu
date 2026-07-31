param(
    [string]$TargetDir = (Join-Path $env:LOCALAPPDATA "split-rc-native-target"),
    [string]$AssetRoot = "..\public\models",
    [string]$Output = (Join-Path (Split-Path $PSScriptRoot -Parent) "validation\sponza-animation-response.json"),
    [string]$VcVars = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
)

$ErrorActionPreference = "Stop"
& "$PSScriptRoot\build.ps1" -TargetDir $TargetDir -VcVars $VcVars

$env:SPLIT_RC_ASSETS = (Resolve-Path -LiteralPath $AssetRoot).Path
$env:SPLIT_RC_ANIMATION_RESPONSE_OUT = $Output
$env:SPLIT_RC_RESOLUTION_SCALE = "1"
$env:SPLIT_RC_FIXED_SCALE = "1"
Remove-Item Env:SPLIT_RC_CAPTURE_DIR -ErrorAction SilentlyContinue
Remove-Item Env:SPLIT_RC_BENCH_FRAMES -ErrorAction SilentlyContinue
Remove-Item Env:SPLIT_RC_BENCH_OUT -ErrorAction SilentlyContinue
Remove-Item Env:SPLIT_RC_STABILITY_OUT -ErrorAction SilentlyContinue
Remove-Item Env:SPLIT_RC_STABILITY_SCENE -ErrorAction SilentlyContinue
Remove-Item Env:SPLIT_RC_STABILITY_REFERENCE_FRAME -ErrorAction SilentlyContinue
Remove-Item Env:SPLIT_RC_TEMPORAL_JITTER -ErrorAction SilentlyContinue
Remove-Item Env:SPLIT_RC_ROUGH_SPECULAR -ErrorAction SilentlyContinue
Remove-Item Env:SPLIT_RC_FREEZE_LIGHTS -ErrorAction SilentlyContinue
Remove-Item Env:SPLIT_RC_GPU_PROFILE -ErrorAction SilentlyContinue

$parent = Split-Path $Output -Parent
if ($parent) {
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
}
$executable = Join-Path $TargetDir "release\split-rc-native.exe"
& $executable
if ($LASTEXITCODE -ne 0) {
    throw "Animated-light response gate failed with exit code $LASTEXITCODE"
}
if (-not (Test-Path -LiteralPath $Output)) {
    throw "Animated-light response run completed without producing $Output"
}
& "$PSScriptRoot\add-provenance.ps1" -Report $Output -Executable $executable
$report = Get-Content -LiteralPath $Output -Raw | ConvertFrom-Json
if (-not $report.gate.passes) {
    throw "Animated-light response was not visible or runtime diagnostics were nonzero."
}
Get-Content -LiteralPath $Output
