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
- the paper's ambient form of screen-space `C(-1)` merging

WebGPU does not expose a portable hardware ray-tracing pipeline. Rays and
shadow queries therefore traverse a balanced triangle BVH in WGSL. This
changes the traversal backend, not the Split RC interval or merge math.

## Stability

The default stable mode freezes the paper's global R2 jitter, uses deterministic
probe-key ordering for Algorithm 3 offsets, blends adjacent LODs across the
paper's 0.9 overlap, and accumulates only exact world/LOD/cache key matches.
There is no screen-space reprojection.

The built-in audit measures matched world-probe irradiance while the camera is
moving and the lights are frozen. On the development NVIDIA RTX 5080, the
worst ten-scene 95th-percentile frame-to-frame change was 1.27%; the color
bleed lab measured 0.97%. This is below the audit's visual-shimmer threshold.

## Ten validation scenes

1. Color bleed laboratory - near-field transfer and emissive geometry.
2. Sponza atrium - the official Crytek/Khronos model used in the paper,
   prepacked as 262,267 triangles and a 131,317-node BVH.
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
grid and two primary rays per visible GI sample:

- all ten scenes held the 60 Hz display cap
- full GPU frame time ranged from roughly 1.4 to 5.7 ms
- the 262k-triangle Sponza scene remained below 6 ms
- zero sparse-hash/probe-capacity overflows
- zero uncaptured WebGPU validation errors
- all ten moving-camera probe-stability checks passed

These numbers are device-specific. Run the in-app audit on another machine for
its actual result.

## Development and deployable bundle

```powershell
npm install
npm test
npm run build
```

`npm run build` validates the Vinext/Sites production output and refreshes the
standalone `netlify-dist` folder. That folder is a complete static fallback
bundle, although the canonical deployment is published through OpenAI Sites.

The Sponza payload is same-origin and retains the Khronos sample asset's
[Cryengine Limited License attribution](./public/models/SPONZA-LICENSE.md).
The runtime sends no user data and makes no third-party requests.

## Known physical limits

Like the paper, the technique is biased by probe interpolation, can overblur
hard shadows, and cannot represent sharp mirror reflections. The rough
specular path intentionally targets broad lobes. Static geometry is supported;
changing meshes requires rebuilding their BVH. These are method limits, not
silent feature omissions.
