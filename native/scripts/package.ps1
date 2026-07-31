param(
    [string]$TargetDir = (Join-Path $env:LOCALAPPDATA "split-rc-native-target"),
    [string]$AssetRoot = "..\public\models",
    [string]$VcVars = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
)

$ErrorActionPreference = "Stop"
& "$PSScriptRoot\build.ps1" -TargetDir $TargetDir -VcVars $VcVars

$dist = Join-Path (Split-Path $PSScriptRoot -Parent) "dist"
$assets = Join-Path $dist "assets"
New-Item -ItemType Directory -Force -Path $assets | Out-Null

$executable = Join-Path $TargetDir "release\split-rc-native.exe"
$sponza = Join-Path $AssetRoot "sponza.rcb"
$atlas = Join-Path $AssetRoot "sponza-atlas.webp"
$license = Join-Path $AssetRoot "SPONZA-LICENSE.md"
foreach ($asset in @($sponza, $atlas, $license)) {
    if (-not (Test-Path -LiteralPath $asset)) {
        throw "Required Sponza asset was not found at: $asset"
    }
}
Copy-Item -LiteralPath $executable -Destination $dist -Force
Copy-Item -LiteralPath $sponza -Destination $assets -Force
Copy-Item -LiteralPath $atlas -Destination $assets -Force
Copy-Item -LiteralPath $license -Destination $assets -Force
Copy-Item -LiteralPath (Join-Path (Split-Path $PSScriptRoot -Parent) "README.md") -Destination $dist -Force
Write-Host "Packaged redistributable directory: $dist"
