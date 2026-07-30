# Split Radiance Cascades — WebGPU

A browser implementation of Freeman and Sannikov's [Split Radiance Cascades](https://arxiv.org/abs/2607.20384) for real-time, single-bounce diffuse global illumination.

The renderer is production-oriented rather than a shader toy: it has a software triangle BVH, sparse world-space hash probes, fixed-point atomic interval deposition, four back-to-front radiance cascades, a 6×6 per-probe irradiance prefilter, world-keyed temporal reuse, moving lights and sun, diagnostics, GPU timestamp profiling, device-loss handling, and ten deterministic validation scenes.

## Netlify deliverable

Run:

```powershell
npm run build:netlify
```

Upload the entire `netlify-dist` folder to [Netlify Drop](https://app.netlify.com/drop). It is a dependency-free static deployment: no server, environment variables, database, or external assets are required.

WebGPU requires a secure context in production. Netlify HTTPS satisfies that requirement. Use a current hardware-accelerated Chrome or Edge release.

## Paper-to-GPU mapping

The implementation maps the paper's core diffuse path as follows:

- 4 cascades
- branching factor `K = 4`
- length scaling `l = 4`
- 32 base directions (`Θ₀ = 4`)
- equal-area spherical direction mapping
- R2 low-discrepancy surface-ray generation
- world-space-stable surface sequence keys
- sparse, half-cell-offset world-space probes
- ray splitting into transparent lower intervals and one terminating radiance interval
- premultiplied-alpha-style back-to-front cascade merging
- sparse trilinear interpolation with weight renormalization
- 6×6 octahedral irradiance prefilter for a maximum of eight probe reads per shaded pixel
- Chebyshev-distance probe LODs
- double-buffered world-probe history with exact-key rejection

WebGPU does not currently expose hardware ray-tracing pipelines, so the ray stage traverses a CPU-built, GPU-resident triangle BVH in WGSL. This preserves the paper's interval and cascade math while remaining portable across conformant browser implementations.

See [IMPLEMENTATION.md](./IMPLEMENTATION.md) for the pass graph and buffer layout.

## Ten validation scenes

1. Color bleed laboratory — near-field transfer and emissive geometry.
2. Sponza atrium — the official Crytek/Khronos geometry used by the paper, prepacked as 262,267 triangles and a 131,317-node BVH.
3. Concave canyon heightfield — a 72×72 terrain with nested craters and ravines.
4. Dense lantern forest — thousands of thin foliage triangles.
5. Multi-level atrium — stairs, balconies, skylight, and curved sculptures.
6. Industrial pipe maze — curved pipework, narrow gaps, and emissive furnaces.
7. Sun temple — layered portals and moving sharp-to-soft sunlight.
8. Orbital sculpture field — open-sky misses and high-curvature meshes.
9. Night market — many colored emitters and dark-region stability.
10. Megacity stress grid — 13,496 triangles and maximum sparse-probe pressure.

Every scene animates the sun and a colored point light. Camera paths are deterministic and can be paused. Drag to orbit, use the wheel to zoom, use WASD to translate, and press Space to pause/resume the camera.

## Quality and diagnostics

The control panel exposes four quality tiers and six render views:

- final composite
- indirect only
- direct only
- normals
- sparse-probe coverage
- albedo

The built-in ten-scene audit records FPS, GPU time when timestamp queries are available, triangle count, probe count, ray count, hit rate, hash overflow, and uncaptured WebGPU errors. A run passes only when every scene remains above 30 FPS and reports zero overflows/errors.

## Verification

On 2026-07-30, balanced mode at a 1152×648 internal resolution (192×108 GI grid, two rays per GI sample) was tested on an NVIDIA GeForce RTX 5080 / Blackwell adapter:

- all ten scenes held the 60 Hz display cap
- measured full GPU frames ranged from 0.39–1.84 ms, including the 262k-triangle Sponza scene
- zero sparse-hash overflows
- zero WebGPU validation errors
- two frozen camera/light frames 800 ms apart were byte-identical across all 1,478,400 viewport channels after settling

The captured scene sheet and machine-readable results are under [docs/validation](./docs/validation). These figures are device-specific; use the in-app audit on the deployment target rather than treating them as a universal guarantee.

## Development

```powershell
npm install
npm run dev
npm test
npm run build
```

`npm run build` validates the Vinext/Sites build and also refreshes `netlify-dist`. `npm test` covers cascade scaling, equal-area mapping, the R2 sequence, probe snapping/key stability, BVH intersections, and all scene/BVH payloads.

## Scope and limitations

- The renderer computes one diffuse indirect bounce, matching the paper's main evaluation mode.
- Geometry is static while lights and the sun are fully dynamic. Rebuild the scene BVH to move mesh geometry.
- The paper's optional secondary multibounce cache, rough specular extension, and screen-space C(-1) extension are not enabled.
- The GPU path uses world-keyed deterministic surface sequences instead of the paper's exact hierarchical prefix-sum allocator from Algorithm 3. This removes camera-indexed shimmer and preserves nested cascade directions, but it is not a claim of feature-for-feature parity with every optional paper path.
- Very thin geometry below the base probe spacing can blur or leak; this is an acknowledged limitation of the paper's method. The probe-coverage view makes sparse holes visible during content authoring.
- Timestamp queries are optional. The UI falls back to CPU frame cadence when the adapter does not expose them.

The Sponza payload is fetched only from the same deployment origin and retains the Khronos sample asset's [Cryengine Limited License attribution](./public/models/SPONZA-LICENSE.md). The runtime makes no third-party requests and sends no user data.
