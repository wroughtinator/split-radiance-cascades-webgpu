param(
    [Parameter(Mandatory = $true)]
    [string]$SokolShdc
)

$ErrorActionPreference = "Stop"
if (-not (Test-Path -LiteralPath $SokolShdc)) {
    throw "sokol-shdc was not found at: $SokolShdc"
}
& $SokolShdc `
    -i "shaders/split_rc.glsl" `
    -o "src/shader.rs" `
    -l "glsl430:metal_macos:hlsl5:spirv_vk" `
    -f "sokol_rust" `
    -b `
    -e "msvc"
if ($LASTEXITCODE -ne 0) {
    throw "Shader compilation failed with exit code $LASTEXITCODE"
}
cargo fmt
