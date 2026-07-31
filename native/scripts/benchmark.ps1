param(
    [int]$Frames = 1200,
    [double]$Scale = 1.0,
    [string]$TargetDir = (Join-Path $env:LOCALAPPDATA "split-rc-native-target"),
    [string]$AssetRoot = "..\public\models",
    [string]$Output = (Join-Path (Split-Path $PSScriptRoot -Parent) "profile\native-benchmark.json"),
    [string]$VcVars = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
)

$ErrorActionPreference = "Stop"
if ($Frames -lt 120) {
    throw "Use at least 120 benchmark frames after warmup."
}
if ($Scale -lt 0.5 -or $Scale -gt 1.0) {
    throw "Scale must be between 0.5 and 1.0."
}
& "$PSScriptRoot\build.ps1" -TargetDir $TargetDir -VcVars $VcVars

$env:SPLIT_RC_ASSETS = (Resolve-Path -LiteralPath $AssetRoot).Path
$env:SPLIT_RC_BENCH_FRAMES = $Frames.ToString()
$env:SPLIT_RC_BENCH_OUT = $Output
$env:SPLIT_RC_RESOLUTION_SCALE = $Scale.ToString(
    [System.Globalization.CultureInfo]::InvariantCulture
)
$env:SPLIT_RC_FIXED_SCALE = "1"
$env:SPLIT_RC_GPU_PROFILE = "1"
Remove-Item Env:SPLIT_RC_CAPTURE_DIR -ErrorAction SilentlyContinue
Remove-Item Env:SPLIT_RC_STABILITY_OUT -ErrorAction SilentlyContinue
Remove-Item Env:SPLIT_RC_ANIMATION_RESPONSE_OUT -ErrorAction SilentlyContinue
Remove-Item Env:SPLIT_RC_STABILITY_SCENE -ErrorAction SilentlyContinue
Remove-Item Env:SPLIT_RC_STABILITY_REFERENCE_FRAME -ErrorAction SilentlyContinue
Remove-Item Env:SPLIT_RC_CACHE_MOTION_OUT -ErrorAction SilentlyContinue
Remove-Item Env:SPLIT_RC_INDIRECT_ONLY -ErrorAction SilentlyContinue
Remove-Item Env:SPLIT_RC_TEMPORAL_JITTER -ErrorAction SilentlyContinue
Remove-Item Env:SPLIT_RC_ROUGH_SPECULAR -ErrorAction SilentlyContinue
Remove-Item Env:SPLIT_RC_FREEZE_LIGHTS -ErrorAction SilentlyContinue

$parent = Split-Path $Output -Parent
if ($parent) {
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
}
$executable = Join-Path $TargetDir "release\split-rc-native.exe"
& $executable
if ($LASTEXITCODE -ne 0) {
    throw "Unlocked benchmark failed with exit code $LASTEXITCODE"
}
if (-not (Test-Path -LiteralPath $Output)) {
    throw "Benchmark completed without producing $Output"
}
& "$PSScriptRoot\add-provenance.ps1" -Report $Output -Executable $executable
Get-Content -LiteralPath $Output
