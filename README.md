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

The twelve-scene baseline is single-bounce Split RC. Sponza opens in that
paper-comparison configuration; its checkbox enables the paper's multibounce
showcase path. Rough-specular and directional `C(-1)` experiments remain
explicit opt-in extensions.

Stable mode advances a deterministic global low-discrepancy temporal rotation,
uses exact canonical screen-sample ranks, orders every allocation decision by
probe key, and uses the paper's hierarchical key-ordered Algorithm 3 offsets.
Directional `(J, beta)` intervals accumulate only on exact world/LOD/cache key
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

The sparse hierarchy retains prior probes inside a bounded view guard so
trilinear neighborhoods and parent ancestry do not pop as the camera moves.
After the paper's world-space interval accumulation, a production presentation
resolve reprojects the tone-mapped result by world position and surface normal.
It rejects disocclusions and normal changes, so it stabilizes sparse-to-screen
reconstruction without smearing newly visible geometry. This final resolve is
an explicitly documented browser-production extension, not a paper equation.

Stable mode advances the deterministic R2/Cranley-Patterson sequence
continuously. An odd 32-bit Weyl permutation supplies the global rotation, so
the 2D rotation pair cannot repeat before the full `2^32`-frame cycle and never
loses precision through a large float conversion. Paused lighting bootstraps
at 0.92 for 24 frames, then uses a 0.98 exact-key interval-history blend;
animated lighting remains at 0.92 and is separately checked for both smooth
motion and step-response lag. Toggling the lighting clock invalidates history
instead of retaining radiance from a discontinuous sun pose. The optional
secondary cache uses a stronger world-validated presentation history under
camera motion.

The built-in audit has eleven independent gates:

- matched world-probe irradiance while the camera moves
- two separately initialized runs of the same camera trajectory, compared from
  the final 8-bit framebuffer
- uninterrupted adjacent moving-camera frames compared after world-position
  reprojection (baseline and multibounce)
- the same uninterrupted motion with animated lights enabled
- a fixed-camera moving-light step response compared with a freshly converged
  target, which detects excessive temporal lag
- a deliberate near/far Sponza camera dolly aimed at view-sized distant meshes
- deterministic 512-spp, 64x36 cosine-weighted BVH references for the
  color-bleed lab, Sponza, Cornell, and the large concave heightmap
- raster sun and six-face point-shadow visibility compared sample-for-sample
  with exact software-BVH visibility, including mandatory coverage of all six
  point-shadow array layers in Cornell
- sparse-hash, key-range, capacity, BVH-stack, and WebGPU error diagnostics

On the development NVIDIA RTX 5080, the final 32-frame Sponza motion runs
measured at most 2/255 at p95 for baseline, moving-light, and multibounce paths;
the multibounce p99 maximum was 6/255. Every per-capture sparse diagnostic was
zero. This directly covers the final image rather than inferring stability
from probe values.

## Twelve validation scenes

1. Color bleed laboratory - near-field transfer and emissive geometry.
2. Sponza atrium - the official Crytek/Khronos geometry reconstructed with the
   paper's neutral/cyan presentation and a real red area-emitter quad as
   262,269 triangles, a 158,359-node SAH BVH, and a 4096px atlas made from all
   25 official base-color materials. The paper does not distribute its exact
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
albedo views, four quality tiers, and independent toggles for multibounce,
rough specular, `C(-1)`, and stable history.

## Measured result

Balanced mode bounds the internal screen to 360,000 pixels; the development
viewport resolves to 800x450 and traces one primary R2 ray per internal-screen
pixel:

- the final all-scene audit measured Sponza at 7.67 ms GPU and 60 FPS; the
  slowest callback rate across all 12 scenes was 57.13 FPS
- baseline, moving-light, and multibounce 32-frame motion gates measure 2/255
  at p95; multibounce measures 6/255 at p99
- the 512-spp Sponza reference gate measures 37.74% raw NRMSE, 31.22%
  99%-trimmed NRMSE, 21.30% low-frequency scale-invariant NRMSE, 0.141
  scene-linear p95 error, 0.35% severe under-light outliers, and 3.47%
  bright-leak candidates
- the point-shadow/BVH classification mismatch is 0% in the laboratory and
  Cornell scenes and 1.19% on the large heightmap; corresponding sun-shadow
  mismatch is 0%, 1.52%, and 1.04%
- no sparse-hash, capacity, or BVH-stack overflow was observed

The exact final machine-readable result is
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
hard shadows, and cannot represent sharp mirror reflections. The rough
specular path intentionally targets broad lobes. Static geometry is supported;
changing meshes requires rebuilding their BVH. These are method limits, not
silent feature omissions.

This is a from-paper WebGPU implementation, not the authors' unreleased source
code. WebGPU's portable software BVH and deterministic key-ordered allocation
replace native ray-tracing and CUDA-specific implementation details. The
paper's bordered irradiance field is copied from its storage write target into
a filterable RGBA16F texture atlas, so final diffuse reconstruction and
secondary-cache feedback use one bilinear texture sample for each of at most
eight sparse probe neighbors per LOD. Moving point shadows are rendered into
six alpha-tested depth layers and sampled with explicit face/UV mapping; this
avoids backend-dependent cube-face orientation assumptions.
