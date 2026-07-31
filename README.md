# Split Radiance Cascades - WebGPU

A production browser implementation of Freeman and Sannikov's
[Split Radiance Cascades](https://arxiv.org/abs/2607.20384).

The renderer implements the paper's primary sparse world-space diffuse GI
path. It runs entirely in WebGPU with a GPU-resident
software BVH, so no native ray-tracing extension, server, or runtime download
is required.

## Paper coverage

- four radiance cascades
- branching factor `K = 4` and length scaling `l = 4`
- 32 base directions (`Theta_0 = 4`)
- equal-area spherical direction mapping
- ray splitting into `(J, beta)` intervals
- sparse half-cell world probes and surface-tangent trilinear interpolation
- the hierarchical R2 prefix allocator from Algorithm 3
- 90%-boundary LOD overlap and cross-fade
- back-to-front cascade merge
- 6x6 octahedral irradiance prefilter
- exact-key world-space temporal accumulation
- four camera-fitted, texel-stabilized directional shadow cascades with
  cross-faded boundaries and 5x5 tent PCF

WebGPU does not expose a portable hardware ray-tracing pipeline. Rays and
shadow queries therefore traverse a binned-SAH triangle BVH in WGSL. This
changes the traversal backend, not the Split RC interval or merge math.
The equation-by-equation audit is in
[PAPER-COVERAGE.md](./PAPER-COVERAGE.md).

## Stability

The shipped renderer has one diffuse Split RC configuration. The paper's
optional multibounce, rough-specular, and directional `C(-1)` experiments are
not compiled into the production shader and have no UI controls.

Stable mode advances a deterministic global low-discrepancy temporal rotation,
uses exact canonical screen-sample ranks, orders every allocation decision by
probe key, and uses the paper's hierarchical key-ordered Algorithm 3 offsets.
Directional `(J, beta)` intervals accumulate only on exact world/LOD key
matches. Adjacent LODs cross-fade across the paper's 0.9 overlap. There is no
separate low-resolution ray lattice: Algorithm 3 assigns one ray to every pixel
of the bounded internal render. Performance tiers vary that internal screen
resolution, so thin and distant geometry cannot fall between ray samples and
Retina/4K windows cannot multiply work without limit. Balanced and Performance
also adapt that bounded pixel budget to sustained frame time; they always keep
one owner per resulting internal pixel and never switch to a sparse ray grid.
GPU timestamps drive recovery when available. A browser without timestamp
queries makes a conservative five-second upscale probe after sustaining a
vsync-limited 60 Hz cadence, so a transient slowdown cannot leave the renderer
permanently downscaled.

The sparse hierarchy is recreated from current visible surfaces every frame,
as specified by Algorithms 1 and 3. Only probes that exist in both frames may
reuse an exact-key world-space interval. Old off-screen probes are not retained
as interpolation neighbors. To remove the paper's documented nearest-only
quality tradeoff, c0 initialization allocates the four interpolation neighbors
in the dominant surface tangent plane while retaining one Algorithm 3 ray per
visible pixel. The same four-neighbor footprint is used for reconstruction, so
camera motion cannot toggle floor or wall coverage as pixels cross a nearest
probe boundary. The tone-mapped current Split RC result receives only
current-frame FXAA; there is no recursive screen-space presentation history.
These two invariants prevent a long camera translation from accumulating stale
block-shaped light and ensure that rebuilding history at a fixed pose produces
the same sparse population.

Stable mode advances the deterministic R2/Cranley-Patterson sequence
continuously. An odd 32-bit Weyl permutation supplies the global rotation, so
the 2D rotation pair cannot repeat before the full `2^32`-frame cycle and never
loses precision through a large float conversion. Paused lighting accumulates
an exact-key sample-count-weighted running average, capped at 16,384 effective
samples per interval; animated lighting uses a bounded 0.965 EMA and is
separately checked for both smooth motion and step-response lag. Toggling the
lighting clock invalidates history
instead of retaining radiance from a discontinuous sun pose.

The built-in audit has twelve independent gates:

- matched world-probe irradiance while the camera moves
- two separately initialized runs of the same camera trajectory, compared from
  the final 8-bit framebuffer
- uninterrupted adjacent moving-camera frames compared after world-position
  reprojection
- the same uninterrupted motion with animated lights enabled
- the reported low-camera Sponza translation, followed by a same-pose history
  rebuild; it requires identical per-cascade probe counts, zero diagnostics,
  clean motion checkpoints, and a low image delta
- a low-floor Sponza forward/backward loop with strict adjacent-frame and
  same-pose clean-rebuild image gates; Probe coverage must be byte-identical
  at every matched pixel throughout the loop
- a fixed-camera moving-light step response compared with a freshly converged
  target, which detects excessive temporal lag
- a deliberate near/far Sponza camera dolly aimed at view-sized distant meshes
- deterministic 512-spp, 64x36 cosine-weighted BVH references for the
  color-bleed lab, Sponza, Cornell, and the large concave heightmap
- raster sun and six-face point-shadow visibility compared sample-for-sample
  with exact software-BVH visibility, including an eight-position full sun
  sweep and mandatory coverage of all six point-shadow layers in Cornell
- sparse-hash, key-range, capacity, BVH-stack, and WebGPU error diagnostics

On the development NVIDIA RTX 5080, the full low-floor Sponza forward/backward
coverage loop measures exactly 0/255 maximum byte delta at every adjacent
matched surface and at same-pose closure. The corresponding indirect-only loop
measures 2/255 at p95, 7/255 at p99, and 1.14/255 trimmed RMSE. The same path
with moving lighting remains 2/255 at p95 and 7/255 at p99. Every per-capture
sparse diagnostic is zero. This directly covers the final image rather than
inferring stability from probe values.

## Twelve validation scenes

1. Color bleed laboratory - near-field transfer and emissive geometry.
2. Sponza atrium - the official Crytek/Khronos geometry reconstructed with the
   paper's neutral/cyan presentation and a real red area-emitter quad as
   262,269 triangles, a 158,359-node SAH BVH, and a 25-layer 811px sRGB texture
   array with complete mip chains made from all official base-color materials.
   The paper does not distribute its exact
   prepared asset, camera, or exposure, so visual matching is not mislabeled
   as byte-identical scene provenance.
3. Concave canyon heightfield - a 72x72 terrain with craters and ravines.
4. Dense lantern forest - thousands of thin foliage triangles.
5. Multi-level atrium - stairs, balconies, skylight, and curved sculptures.
6. Industrial pipe maze - curved pipework, narrow gaps, and emitters.
7. Sun temple - layered portals and moving sunlight.
8. Orbital sculpture field - open-sky misses and high-curvature meshes.
9. Night market - many colored emitters and dark-region stability.
10. Megacity stress grid - 13,496 triangles and high probe pressure.
11. Cornell box reference - canonical red/green enclosure, two occluders,
    ceiling area emitter, and a moving comparison light.
12. Grand concave heightmap - a 128x128 terrain with nested bowls, crater,
    ravine, terraces, shelves, and moving sun/fill light.

Every scene has a deterministic camera path, moving sun, and moving colored
point light. The controls expose final, indirect, direct, normal, coverage, and
albedo views, four quality tiers, and stable-history diagnostics. There are no
optional shading-mode knobs.

## Measured result

Balanced mode bounds the internal screen to 360,000 pixels; the development
viewport resolves to 800x450 and traces one primary R2 ray per internal-screen
pixel:

- the final all-scene audit measured Sponza at 6.49 ms GPU and 60 FPS; the
  slowest callback rate across all 12 scenes was 55.38 FPS
- baseline and moving-light 32-frame motion gates measure 3/255 at p95
- the 512-spp Sponza reference gate measures 37.12% raw NRMSE, 29.91%
  99%-trimmed NRMSE, 20.64% low-frequency scale-invariant NRMSE, 0.137
  scene-linear p95 error, 0.35% severe under-light outliers, and 3.21%
  bright-leak candidates
- the point-shadow/BVH classification mismatch is 0% in the laboratory and
  Cornell scenes and 1.19% on the large heightmap; corresponding sun-shadow
  mismatch is 0%, 1.52%, and 1.04%
- no sparse-hash, capacity, or BVH-stack overflow was observed

The final machine-readable summary is
[`docs/validation/RTX-5080-balanced.json`](docs/validation/RTX-5080-balanced.json).

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
hard indirect detail, and cannot represent sharp mirror reflections. Static
geometry is supported; changing meshes requires rebuilding their BVH. These
are method limits, not silent feature omissions.

This is a from-paper WebGPU implementation, not the authors' unreleased source
code. WebGPU's portable software BVH and deterministic key-ordered allocation
replace native ray-tracing and CUDA-specific implementation details. The
paper's bordered irradiance field is copied from its storage write target into
a filterable RGBA16F texture atlas, so final diffuse reconstruction uses one
bilinear texture sample for each of at most four surface-tangent probe
neighbors per LOD. Moving point shadows are rendered into
six alpha-tested depth layers and sampled with explicit face/UV mapping; this
avoids backend-dependent cube-face orientation assumptions.
