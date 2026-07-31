# Paper implementation coverage

This matrix maps the implementation to Freeman and Sannikov, *Split Radiance
Cascades: Real-Time Global Illumination via Sparse Radiance Probes*
([arXiv:2607.20384](https://arxiv.org/abs/2607.20384)).

| Paper item | WebGPU implementation | Validation |
| --- | --- | --- |
| Eq. 1-2, Lambertian directional irradiance | 32 c0 equal-area directions are cosine-integrated into a bordered 6x6 octahedral irradiance field. | 512-spp, 64x36 cosine-weighted BVH references, robust/frozen regression gates, and RGB error image. |
| Eq. 4-5, cascade scaling | Four cascades use `K=4`, `l=4`, `Theta0=4`, spacings `ds0*2^n`, 32/128/512/2048 directions, and `t0=1.6*ds0`. | Math unit tests and shader constants. |
| Eq. 6, interval composition | Rays deposit `(J,beta)` into every crossed interval; merge evaluates `J + beta*I_parent` from c3 to c0. | Shader regression tests and reference audit. |
| Algorithm 1, sparse RC | Every visible internal-screen pixel inserts its half-cell c0 probe; parent cascades are sparse hash maps; gathers renormalize only existing trilinear neighbors. | Twelve-scene capacity/finite-value tests and live overflow counters. |
| Section 4.1, LOD | Chebyshev-distance LODs are independent, scale spacing and cutoff together, overlap at 0.9, and cross-fade. | Shader tests and continuous camera test. |
| Section 5, ray splitting | Rays originate on visible surfaces, face outward, and split at the cascade cutoffs instead of originating at probes. | Path-reference audit and Sponza comparison. |
| Algorithm 2, equal-area map | Encode/decode use azimuth and uniform sphere height. | Round-trip and distribution unit tests. |
| Algorithm 3, hierarchical R2 | One ray is assigned to every visible internal-screen pixel. Counts propagate c0-to-c3, reverse offsets assign contiguous parent segments, and each ray receives an exact deterministic local rank plus one global temporal rotation. | Byte-identical independent trajectory replay plus path-reference coverage gate. |
| Section 5.2, temporal accumulation | Exact world/LOD/cache keys retrieve the previous directional `(J,beta)` interval, which is blended before recursive merge; a bounded guard retains the complete prior sparse ancestry during camera motion. | Baseline and multibounce replay plus uninterrupted motion gates. |
| Section 6, irradiance optimization | A 6x6 octahedral field with an evaluated one-texel border is written to a double-buffered RGBA16F storage atlas, copied at the WebGPU usage boundary, and consumed through a filterable atlas; final and secondary-cache gathers use one bilinear lookup for each of at most eight sparse neighbors per LOD. | Shader contract tests, four-scene reference comparison, and GPU profiling. |
| Section 6, multibounce | Previous primary hit points seed a cache two LODs coarser; the cache samples its prior irradiance and can feed itself. | Dedicated repeatability and 32-frame motion gates. |
| Section 7.1, rough specular | The reflection direction selects a broad c2 cone and composes c1/c0 intervals in front. | Optional live view and shader regression test. |
| Section 7.1, directional c(-1) | 32 screen-space near intervals are composed in front of world-space cones; c0 rays begin at `ds0` when enabled. | Optional live view and shader regression test. |
| Section 7.1, denoising interpretation | Split RC itself is the spatiotemporal radiance-interval filter. OIDN is not shipped because the paper uses it only as an offline comparison baseline, not as part of Split RC. | Raw/reference/Split-RC error triptych. |

## Backend substitutions

The authors' unreleased implementation uses LuisaCompute, CUDA, OptiX,
64-bit hash keys, and warp-level sharing. Portable browser WebGPU exposes none
of OptiX, CUDA warps, hardware ray queries, or atomic 64-bit integers.
Accordingly:

- a 16-bin SAH BVH is built offline and traversed exactly in WGSL;
- hash keys use a bounded 32-bit scene-local packing appropriate to the
  published scenes; out-of-range coordinates increment a visible diagnostic
  instead of aliasing through clamping;
- subgroup sharing is replaced by coherent storage-buffer reads and explicit
  WebGPU pass boundaries;
- atomic interval accumulation uses non-negative fixed point.

These substitutions change storage and traversal mechanics, not the cascade,
screen-ray assignment, interval, merge, LOD, temporal, or shading equations. BVH profiling
on the packed Sponza asset measures 50 median / 94 p95 node visits and 12
median / 24 p95 triangle tests across 1,024 representative rays.

The deployment additionally applies a world-position/normal-validated
presentation resolve after Split RC shading. It is isolated from the paper's
interval math and rejects disocclusions; its purpose is to remove
sparse-to-screen reconstruction shimmer in an interactive browser. The audit
captures world position and surface normal alongside the image and tests the
paper field (exact-key probes), the deterministic renderer, and this final
presented image independently.

## Stated method limits

The implementation does not claim equality with a path tracer. Section 8.1 of
the paper states that Split RC is biased by interpolation, can leak light,
overblur details below the base spacing, and cannot represent sharp mirror
reflection. The production audit therefore reports those errors rather than
hiding them: NRMSE and energy bias, p95/p99 absolute error, severe under-light
outliers, and bright-leak candidates. Raw pixel error remains reported; a 3x3
low-frequency scale-invariant NRMSE is gated because irradiance is explicitly
a low-frequency field.
