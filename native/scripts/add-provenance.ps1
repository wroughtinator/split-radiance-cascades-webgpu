param(
    [Parameter(Mandatory = $true)]
    [string]$Report,
    [Parameter(Mandatory = $true)]
    [string]$Executable
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path $PSScriptRoot -Parent
$shaderSource = Join-Path $projectRoot "shaders\split_rc.glsl"
$generatedShader = Join-Path $projectRoot "src\shader.rs"
$gpuHarness = Join-Path $projectRoot "src\gpu.rs"
$cargoLock = Join-Path $projectRoot "Cargo.lock"
foreach ($path in @($Report, $Executable, $shaderSource, $generatedShader, $gpuHarness, $cargoLock)) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Provenance input was not found: $path"
    }
}

$document = Get-Content -LiteralPath $Report -Raw | ConvertFrom-Json
$provenance = [ordered]@{
    generated_at_utc = [DateTime]::UtcNow.ToString("o")
    executable_sha256 = (Get-FileHash -LiteralPath $Executable -Algorithm SHA256).Hash.ToLowerInvariant()
    shader_source_glsl_sha256 = (Get-FileHash -LiteralPath $shaderSource -Algorithm SHA256).Hash.ToLowerInvariant()
    generated_shader_rs_sha256 = (Get-FileHash -LiteralPath $generatedShader -Algorithm SHA256).Hash.ToLowerInvariant()
    gpu_harness_rs_sha256 = (Get-FileHash -LiteralPath $gpuHarness -Algorithm SHA256).Hash.ToLowerInvariant()
    cargo_lock_sha256 = (Get-FileHash -LiteralPath $cargoLock -Algorithm SHA256).Hash.ToLowerInvariant()
}
$document | Add-Member -NotePropertyName provenance -NotePropertyValue $provenance -Force
$document | ConvertTo-Json -Depth 64 | Set-Content -LiteralPath $Report -Encoding UTF8
