# Split Radiance Cascades - WebGPU

A production browser implementation of Freeman and Sannikov's
[Split Radiance Cascades](https://arxiv.org/abs/2607.20384).

The renderer implements the paper's sparse world-space diffuse GI path and its
demonstrated extensions. It runs entirely in WebGPU with a GPU-resident
software BVH, so no native ray-tracing extension, server, or runtime download
is required.

## Paper coverage

- four radiance cascades
- branching factor `K = 4` and length scaling `l = 4`
- 32 base directions (`Theta_0 = 4`)
- equal-area spherical direction mapping
- ray splitting into `(J, beta)` intervals
- sparse half-cell world probes and renormalized trilinear interpolation
- the hierarchical R2 prefix allocator from Algorithm 3
- 90%-boundary LOD overlap and cross-fade
- back-to-front cascade merge
- 6x6 octahedral irradiance prefilter
- exact-key world-space temporal accumulation
- two-LOD-coarser secondary cache for multibounce illumination
- rough-specular cone reconstruction
- directional screen-space `C(-1)` interval merging (optional extension)

WebGPU does not expose a portable hardware ray-tracing pipeline. Rays and
shadow queries therefore traverse a binned-SAH triangle BVH in WGSL. This
changes the traversal backend, not the Split RC interval or merge math.
The equation-by-equation audit is in
[PAPER-COVERAGE.md](./PAPER-COVERAGE.md).

## Stability

The ten-scene baseline is single-bounce Split RC. Sponza opens in the paper's
multibounce showcase configuration for presentation quality; its checkbox can
return immediately to the single-bounce comparison. Rough-specular and
directional `C(-1)` experiments remain explicit opt-in extensions.

Stable mode advances a deterministic global low-discrepancy temporal rotation,
uses exact canonical screen-sample ranks, canonicalizes sparse probe indices
by key, and uses the paper's hierarchical key-ordered Algorithm 3 offsets.
Directional `(J, beta)` intervals accumulate only on exact world/LOD/cache key
matches. Adjacent LODs cross-fade across the paper's 0.9 overlap. There is no
stochastic full-resolution GI trace: the GI grid is 192x108 in Balanced mode.

The sparse hierarchy retains prior probes inside a bounded view guard so
trilinear neighborhoods and parent ancestry do not pop as the camera moves.
After the paper's world-space interval accumulation, a production presentation
resolve reprojects the tone-mapped result by world position and surface normal.
It rejects disocclusions and normal changes, so it stabilizes sparse-to-screen
reconstruction without smearing newly visible geometry. This final resolve is
an explicitly documented browser-production extension, not a paper equation.

Stable mode explores 32 temporal rotations, then freezes the converged global
rotation. With paused lighting, exact-key intervals also lock as suggested by
the paper's semi-static temporal path; animated lights continue adapting at a
0.96 history blend.

The built-in audit has four independent gates:

- matched world-probe irradiance while the camera moves
- two separately initialized runs of the same camera trajectory, compared from
  the final 8-bit framebuffer
- uninterrupted adjacent moving-camera frames compared after world-position
  reprojection (baseline and multibounce)
- a deterministic 128-spp cosine-weighted BVH path-traced reference for Sponza

On the development NVIDIA RTX 5080, independent Sponza trajectory replays
measured 0/255 at p95 in both modes (p99 0/255 baseline and 1/255
multibounce). Across 31 consecutive moving-camera comparisons, baseline and
multibounce both measured 2/255 at p95 and 8/255 at p99 after world
reprojection, with 99.5%-trimmed RMSE of 1.27/255 and 1.24/255. This directly
covers the final image rather than inferring stability from probe values.

## Ten validation scenes

1. Color bleed laboratory - near-field transfer and emissive geometry.
2. Sponza atrium - the official Crytek/Khronos model used in the paper,
   prepacked with the paper-style neutral/cyan material preset and a real red
   area-emitter quad as 262,269 triangles, a 158,359-node SAH BVH, and a
   4096px atlas made from all 25 official base-color materials.
3. Concave canyon heightfield - a 72x72 terrain with craters and ravines.
4. Dense lantern forest - thousands of thin foliage triangles.
5. Multi-level atrium - stairs, balconies, skylight, and curved sculptures.
6. Industrial pipe maze - curved pipework, narrow gaps, and emitters.
7. Sun temple - layered portals and moving sunlight.
8. Orbital sculpture field - open-sky misses and high-curvature meshes.
9. Night market - many colored emitters and dark-region stability.
10. Megacity stress grid - 13,496 triangles and high probe pressure.

Every scene has a deterministic camera path, moving sun, and moving colored
point light. The controls expose final, indirect, direct, normal, coverage, and
albedo views, four quality tiers, and independent toggles for multibounce,
rough specular, `C(-1)`, and stable history.

## Measured result

Balanced mode was validated at a 1152x648 render resolution with a 192x108 GI
grid and 12 primary rays per visible GI sample:

- ordinary Sponza profiling measured 10.42 ms single-bounce and 11.73 ms
  multibounce at 60 FPS; the exhaustive audit's worst Sponza sample was
  14.02 ms
- both baseline and multibounce Sponza trajectory replays measured 0/255 at
  p95, and both uninterrupted 32-frame camera-motion tests measured 2/255 at
  p95 and 8/255 at p99
- exact matched Sponza world probes changed by 0.83% at p95 during camera
  motion, with zero sparse/BVH overflows or WebGPU errors
- the independent reference gate measured 48.25% raw NRMSE, 47.21% raw
  scale-invariant NRMSE, 27.21% low-frequency scale-invariant NRMSE, 0.198
  scene-linear p95 error, 1.00% severe under-light outliers, and 8.41%
  bright-leak candidates
- all ten production scenes passed final-frame replay, continuous-camera,
  world-probe, overflow, and device-error gates; the audit's minimum sampled
  rate was 55.4 FPS

The reference numbers are not a claim of exact path-traced equality: Split RC
is a biased sparse estimator, and the paper documents interpolation leaks,
overblurring, and base-direction bias. They are regression gates that fail when
the implementation materially diverges. Performance is device-specific; run
the in-app audit on another machine for its actual result.

## Development and deployable bundle

```powershell
npm install
npm test
npm run build
```

`npm run build` validates the Vinext/Sites production output and refreshes the
standalone `netlify-dist` folder. That folder is a complete static fallback
bundle, although the canonical deployment is published through OpenAI Sites.

The Sponza geometry and texture atlas are same-origin and retain the official
Khronos sample asset's
[Cryengine Limited License attribution](./public/models/SPONZA-LICENSE.md).
The runtime sends no user data and makes no third-party requests.

## Known physical limits

Like the paper, the technique is biased by probe interpolation, can overblur
hard shadows, and cannot represent sharp mirror reflections. The rough
specular path intentionally targets broad lobes. Static geometry is supported;
changing meshes requires rebuilding their BVH. These are method limits, not
silent feature omissions.

This is a from-paper WebGPU implementation, not the authors' unreleased source
code. WebGPU's portable software BVH, deterministic probe canonicalization, and
manual storage-buffer filtering replace native ray-tracing and CUDA-specific
implementation details without changing the Split RC interval/merge model.
