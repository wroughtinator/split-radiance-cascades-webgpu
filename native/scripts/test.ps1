param(
    [string]$TargetDir = (Join-Path $env:LOCALAPPDATA "split-rc-native-target"),
    [string]$VcVars = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
)

$ErrorActionPreference = "Stop"
if (-not (Test-Path -LiteralPath $VcVars)) {
    throw "Visual Studio Build Tools vcvars64.bat was not found at: $VcVars"
}
$env:CARGO_TARGET_DIR = $TargetDir
$command = "call `"$VcVars`" >nul && cargo fmt -- --check && cargo test --locked --all-targets"
& cmd.exe /d /s /c $command
if ($LASTEXITCODE -ne 0) {
    throw "Native validation failed with exit code $LASTEXITCODE"
}
