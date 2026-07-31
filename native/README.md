# Split Radiance Cascades — native Rust + Sokol

This directory is an independent native implementation of the same Split
Radiance Cascades renderer as the browser build. It uses the official
`floooh/sokol-rust` repository pinned in `Cargo.lock`, a software 16-bin SAH
BVH, and one generated shader source for D3D11, Metal, OpenGL 4.3, and Vulkan.

The renderer is not a raster fallback. Primary visibility, alpha-tested
materials, direct shadows, sparse-probe construction, hierarchical ray
assignment, split tracing, interval merging, temporal history, irradiance
integration, and final shading are explicit GPU compute passes.

## Fidelity-critical invariants

- Four cascades with `K=4`, `l=4`, and 32/128/512/2048 Morton-ordered
  equal-area directions.
- One base deterministic R2 owner per visible internal-resolution pixel.
  Fine/coarse LOD overlap and the previous-hit secondary stream are separate
  quality streams, not hidden spatial decimation.
- Exact 61-bit logical probe keys use two GPU words: three signed 19-bit cell
  coordinates, 3-bit LOD, and a primary/secondary tag. Each five-word GPU hash
  slot has a reservation signature, collision-resistant token, both key words,
  and a state/index word published last with release semantics. Lookup verifies
  the full key after acquire, and invalid ranges are rejected and counted.
- Probe construction is split into collect, compact retry, canonicalize, and
  map dispatches. A same-signature reservation defers instead of probing past
  an unpublished slot; tombstone cleanup eliminates rare backend-timing
  duplicates. No consumer can observe a half-published hash entry.
- Allocation order cannot change hierarchical ray rank or parent offsets.
- Temporal history is double-buffered and accepted only after exact
  `(cell, LOD, primary/secondary)` key matching.
- Temporal filtering applies to the raw `(J, beta)` interval. Fixed-lighting
  validation uses exact sample-count accumulation; animated lighting uses a
  0.92 exact-key EMA. An already composed cone is never fed back into a near
  interval.
- The current tone-mapped Split RC field is presented directly. There is no
  recursive screen-space display history.
- The secondary surface cache feeds the next bounce through the previous
  irradiance atlas.
- The diffuse cache is a real filterable RGBA16F texture atlas. Every probe
  owns an 8×8 octahedral tile with a one-texel seam border.
- The paper's optional rough-specular extension starts from a coarse cone and
  composes near raw intervals front-to-back. It is disabled by default because
  the bundled material data has no roughness channel; set
  `SPLIT_RC_ROUGH_SPECULAR=1` to opt in.
- The optional directional screen-space `C(-1)` experiment is implemented and
  can be toggled at runtime.
- Performance scaling changes internal resolution (maximum 960×540), never the
  per-pixel ray-owner stride.

See `PAPER-COVERAGE.md` for the pass-by-pass implementation map.

## Windows

Requirements:

- Rust 1.77 or newer
- Visual Studio 2022 Build Tools with the C++ workload
- A D3D11 compute-capable GPU

From this directory:

```powershell
.\scripts\test.ps1
.\scripts\build.ps1
$env:SPLIT_RC_ASSETS = "..\public\models"
& "$env:LOCALAPPDATA\split-rc-native-target\release\split-rc-native.exe"
```

The app searches `SPLIT_RC_ASSETS`, `../public/models`, `public/models`, and
`assets` for `sponza.rcb` and `sponza-atlas.webp`, in that order.

`.\scripts\package.ps1` creates a redistributable `dist/` directory containing
the executable, packed Sponza geometry, material atlas, license, and README.

## macOS

Install Rust and the Xcode command-line tools, then run:

```sh
cargo test --all-targets
cargo build --release
SPLIT_RC_ASSETS=../public/models cargo run --release --bin split-rc-native
```

Sokol selects Metal. The committed shader binding contains a `metal_macos`
variant generated from the same source as the other native backends.

## Controls

- Right mouse: capture/release look
- `W/A/S/D`, `Q/E`: move
- `1`–`0`: laboratory through stress scene
- `C`: Cornell box
- `H`: 128×128 concave heightmap
- `Space`: pause/resume animated lighting
- `M`: toggle directional screen-space `C(-1)`

Set `SPLIT_RC_C_MINUS_ONE=1` to start with the optional experiment enabled.
The production sampler advances its R2 rotation every frame by default. Set
`SPLIT_RC_TEMPORAL_JITTER=0` only for a diagnostic comparison.

## Automated validation, captures, and benchmark

The deterministic audit checks all 12 scenes, all cascade direction centers,
unbounded R2 temporal rotation, allocation-order invariance, finite animated
lights, packed Sponza UV/material/alpha metadata, and CPU BVH traversal against
brute force:

```powershell
cargo run --release --bin split-rc-audit -- ..\public\models
```

Generate D3D11 framebuffer captures for all 12 scenes and a contact sheet:

```powershell
.\scripts\capture.ps1
```

Run the repeated-camera-loop temporal gate. Lighting is frozen while the
production R2 sequence continues to advance. The laboratory and Sponza are
both tested at full 960×536 internal resolution by comparing the identical
camera pose at frames 1200 and 1320:

```powershell
.\scripts\stability.ps1
```

Reproduce the reported low-camera Sponza translation in indirect-only mode,
hold the final pose, rebuild temporal resources without moving, and require
identical sparse populations plus a low full-resolution image delta:

```powershell
.\scripts\cache-motion.ps1
```

Run the complementary fixed-camera Sponza response gate. It leaves animated
lighting enabled, records the actual sun and point-light trajectories, and
requires a visible framebuffer response plus clean GPU/hash diagnostics:

```powershell
.\scripts\animation-response.ps1
```

Run an unlocked, fixed-resolution Sponza benchmark after a warmup and write a
machine-readable report:

```powershell
.\scripts\benchmark.ps1 -Frames 1200
```

`-Scale` accepts `0.5` through `1.0` and defaults to full internal resolution.
Interactive mode retains dynamic resolution unless `SPLIT_RC_FIXED_SCALE` is
set.

The current full-resolution D3D11 Sponza report covers 262,269 triangles and
records 60.41 callback FPS on the validation machine (display/driver capped).
Thirty D3D11 timestamp samples are distributed every ten frames across the
300-frame measured interval after a separate 60-frame warmup. The sum of
per-stage medians is 10.685 ms, equivalent to about 93.6 GPU-limited FPS; the
sum of per-stage p95 values is 11.719 ms. The largest median groups are split
tracing plus indirect shadows at 4.176 ms, sparse probe construction at
2.432 ms, presentation at 2.340 ms, and primary visibility plus direct shadows
at 1.521 ms. The JSON
retains every stage's sample count, minimum, median, p95, and maximum.

The advancing-R2 repeated-camera gate passes for both scenes on the raw current
field. The laboratory reports mean/RMS/p95/p99/p99.9 deltas of
0.018/0.166/0/1/1 RGB code values; Sponza reports
0.252/0.836/1/3/11. Both runs have zero hash overflow, full-key collision,
key-range rejection, BVH-stack overflow, duplicate slot, reserved slot,
or unpublished slot, with occupied and canonical counts equal. The exact
6.48-unit Sponza translation/rebuild gate reports identical
`[862, 277, 99, 27]` probe populations, p95 1/255 and p99 3/255, and zero
runtime/publication diagnostics. See
`profile/native-benchmark.json`, `validation/*-camera-loop-stability.json`,
`validation/sponza-cache-motion-recovery.json`, and `captures/contact-sheet.png`.

With the Sponza camera fixed, the animated-light response gate changes 93.26%
of RGB channels with a 17.55/255 mean delta between 2.08 s and 6.08 s. The
recorded point light moves from `(3.75, 5.20, 4.68)` to
`(-5.18, 5.90, 3.02)`, the sun direction changes concurrently, and all
runtime/publication diagnostics remain zero. The two source frames and delta
heatmap are stored beside `validation/sponza-animation-response.json`.

Benchmark, stability, cache-motion, and animation-response reports include SHA-256
provenance for the exact executable, GLSL source, generated multi-backend
`shader.rs`, GPU harness source, and `Cargo.lock`, plus the UTC generation
time. Matching hashes make stale evidence detectable.

Runtime diagnostics expose active probes, hash/capacity overflow, full-key
hash collisions, key-range rejection, and BVH stack overflow. D3D11 reports
real GPU timestamps for seven pass groups. Current Sokol does not expose the
same portable timestamp-query API for every selected backend, so other
backends continue to label callback throughput separately.

## Shader regeneration

The generated `src/shader.rs` is committed, so normal builds do not require a
shader compiler:

```powershell
.\scripts\compile-shaders.ps1 -SokolShdc C:\tools\sokol-shdc.exe
```

The source cross-compiles without warnings for GL 4.3, Metal, HLSL 5, and
SPIR-V/Vulkan. The current `sokol-shdc` WGSL translator rejects storage-address
space pointer parameters used by this native compute source; the browser
implementation has its own WGSL shaders and is not generated from this file.
