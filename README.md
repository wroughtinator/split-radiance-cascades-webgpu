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
shadow queries therefore traverse a balanced triangle BVH in WGSL. This
changes the traversal backend, not the Split RC interval or merge math.

## Stability

The default paper baseline is single-bounce Split RC. The paper's multibounce,
rough-specular, and directional `C(-1)` experiments are explicit opt-in
extensions so a screen-space term is never silently mixed into the baseline.

Stable mode freezes the paper's global R2 jitter, uses a deterministic
screen-sample seed, canonicalizes sparse probe indices in hash-slot order, and
uses probe-key ordering for Algorithm 3 offsets. Adjacent LODs cross-fade
across the paper's 0.9 overlap and history only accumulates exact
world/LOD/cache key matches. There is no screen-space reprojection.

The built-in audit now has three independent gates:

- matched world-probe irradiance while the camera moves
- two separately initialized runs of the same camera trajectory, compared from
  the final 8-bit framebuffer
- a deterministic 128-spp cosine-weighted BVH path-traced reference for Sponza

On the development NVIDIA RTX 5080, all ten framebuffer trajectory tests were
byte-identical at every tested pose except two Sponza captures with a worst
single-channel difference of 1/255; Sponza's p99 difference was 0/255. This
directly covers the final image rather than inferring stability from probe
values.

## Ten validation scenes

1. Color bleed laboratory - near-field transfer and emissive geometry.
2. Sponza atrium - the official Crytek/Khronos model used in the paper,
   prepacked as 262,267 triangles, a 131,317-node BVH, and a 4096px atlas made
   from all 25 official base-color materials.
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

- all ten scenes passed the final-frame repeatability and world-probe gates
- full GPU frame time remained below 11.6 ms
- the 262k-triangle textured Sponza scene measured 11.53 ms
- zero sparse-hash/probe-capacity overflows
- zero uncaptured WebGPU validation errors
- Sponza passed the independent path-traced-reference thresholds (50.7% raw
  NRMSE, 35.3% scale-invariant NRMSE, and 0.159 scene-linear p95 absolute
  irradiance error)

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
