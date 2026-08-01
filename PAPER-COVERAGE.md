# Paper implementation coverage

This matrix maps the implementation to Freeman and Sannikov, *Split Radiance
Cascades: Real-Time Global Illumination via Sparse Radiance Probes*
([arXiv:2607.20384](https://arxiv.org/abs/2607.20384)).

| Paper item | WebGPU implementation | Validation |
| --- | --- | --- |
| Eq. 1-2, Lambertian directional irradiance | 32 c0 equal-area directions are cosine-integrated into a bordered 6x6 octahedral irradiance field. | 512-spp, 64x36 cosine-weighted BVH references, robust/frozen regression gates, and RGB error image. |
| Eq. 4-5, cascade scaling | Four cascades use `K=4`, `l=4`, `Theta0=4`, spacings `ds0*2^n`, 32/128/512/2048 directions, and `t0=1.6*ds0`. | Math unit tests and shader constants. |
| Eq. 6, interval composition | Rays deposit `(J,beta)` into every crossed interval; merge evaluates `J + beta*I_parent` from c3 to c0. | Shader regression tests and reference audit. |
| Algorithm 1, sparse RC | Every visible internal-screen pixel assigns its ray to the nearest half-cell c0 probe. For the higher-quality trilinear option discussed in Section 4, allocation also inserts the four neighbors in the dominant surface tangent plane; reconstruction uses the identical normalized footprint. Every cascade key carries a dominant-normal surface sheet so opposite sides of thin geometry never share a cache entry. | Byte-identical Sponza probe-coverage motion loop, twelve-scene capacity/finite-value tests, and live overflow counters. |
| Section 4.1, LOD | Chebyshev-distance LODs are independent, scale spacing and cutoff together, overlap at 0.9, and cross-fade. | Shader tests and continuous camera test. |
| Section 5, ray splitting | Rays originate on visible surfaces, face outward, and split at the cascade cutoffs instead of originating at probes. | Path-reference audit and Sponza comparison. |
| Algorithm 2, equal-area map | Encode/decode use azimuth and uniform sphere height. | Round-trip and distribution unit tests. |
| Algorithm 3, hierarchical R2 | One ray is assigned to every visible internal-screen pixel. Counts propagate c0-to-c3, reverse offsets assign contiguous parent segments, and each ray receives an exact deterministic local rank plus one global temporal rotation. | Byte-identical independent trajectory replay plus path-reference coverage gate. |
| Section 5.2, temporal accumulation | Exact world/LOD keys retrieve the previous directional `(J,beta)` interval, which is blended before recursive merge. The current sparse hierarchy is recreated from current visible surfaces and parents; previous-only probes are removed exactly as described in Section 6. | Deterministic replay, uninterrupted motion, the low-camera Sponza translation/clean-rebuild gate, and a strict forward/backward floor loop. |
| Section 6, irradiance optimization | A 6x6 octahedral field with an evaluated one-texel border is written to a double-buffered RGBA16F storage atlas, copied at the WebGPU usage boundary, and consumed through a filterable atlas; final gathers use one bilinear lookup for each of at most four surface-tangent neighbors per LOD. | Shader contract tests, four-scene reference comparison, and GPU profiling. |
| Section 7.1, ambient-form `C(-1)` | The shipped production extension resolves the paper's missing sub-c0 visibility interval with exact software-BVH rays. A fixed world-space radius avoids camera-relative LOD discontinuities; a six-axis enclosure precheck keeps open surfaces on the smooth Split RC field, while every locally closed surface uses the same 16 deterministic R2 cosine rays. The robust path uses watertight shear-space triangle tests. | Closed-box 512-spp oracle, every-pixel displayed-leak scan, six-step camera loop, same-pose closure, and Cornell motion/reference gates. |
| Section 7.1, denoising interpretation | Split RC itself is the spatiotemporal radiance-interval filter. OIDN is not shipped because the paper uses it only as an offline comparison baseline, not as part of Split RC. | Raw/reference/Split-RC error triptych. |

The production demo intentionally excludes the paper's optional multibounce
and rough-specular experiments. The ambient-form `C(-1)` direction proposed
in Section 7.1 is shipped as a production extension because the paper itself
identifies sub-c0 visibility and interpolation leakage as unresolved limits.

## Backend substitutions

The authors' unreleased implementation uses LuisaCompute, CUDA, OptiX,
64-bit hash keys, and warp-level sharing. Portable browser WebGPU exposes none
of OptiX, CUDA warps, hardware ray queries, or atomic 64-bit integers.
Accordingly:

- a 16-bin SAH BVH is built offline and traversed exactly in WGSL;
- hash keys use a bounded 32-bit scene-local packing of signed cell, LOD, and
  dominant-normal surface sheet; out-of-range coordinates increment a visible
  diagnostic instead of aliasing through clamping;
- subgroup sharing is replaced by coherent storage-buffer reads and explicit
  WebGPU pass boundaries;
- atomic interval accumulation uses non-negative fixed point;
- the high-throughput cascade tracer uses conservative shared-edge tolerances,
  while the closed-volume `C(-1)` path uses a watertight shear-space triangle
  test so rays cannot escape through shared mesh edges.

These substitutions change storage and traversal mechanics, not the cascade,
screen-ray assignment, interval, merge, LOD, temporal, or shading equations. BVH profiling
on the packed Sponza asset measures 50 median / 94 p95 node visits and 12
median / 24 p95 triangle tests across 1,024 representative rays.

Probe base spacing is not authored per scene. A single asset-driven formula
uses only the scene bounding radius and triangle count, with no scene identity
or preset table; all other cascade, cache, temporal, and `C(-1)` parameters are
identical across the twelve scenes.

The deployment applies current-frame-only FXAA to the Split RC composite. It
does not add a recursive screen-space presentation cache. World position and surface
normal are captured only by the validation harness to establish
surface-to-surface correspondence between moving frames.

## Stated method limits

The implementation does not claim equality with a path tracer. Section 8.1 of
the paper states that baseline Split RC is biased by interpolation, can leak
light, overblur details below the base spacing, and cannot represent sharp
mirror reflection. Surface-sheet keys and the exact enclosed-volume `C(-1)`
extension address the demonstrated cross-wall case, but do not turn the sparse
estimator into unbiased path tracing. The audit therefore reports NRMSE and
energy bias, p95/p99 absolute error, severe under-light outliers, and
bright-leak candidates. Raw pixel error remains reported; a 3x3 low-frequency
scale-invariant NRMSE is gated because irradiance is explicitly a
low-frequency field.
