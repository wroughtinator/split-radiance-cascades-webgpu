# Split Radiance Cascades - WebGPU

A production browser implementation of Freeman and Sannikov's
[Split Radiance Cascades](https://arxiv.org/abs/2607.20384).

**Live demo: <https://wroughtinator.github.io/>** (WebGPU-capable browser
required — Chrome or Edge 113+).

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
- a production 75%-to-boundary LOD residency overlap and cross-fade
- back-to-front cascade merge
- 6x6 octahedral irradiance prefilter
- exact-key world-space temporal accumulation
- dominant-normal surface-sheet keys across every cascade
- exact world-space enclosed-volume `C(-1)` visibility resolution
- automatic asset-scale probe spacing with no per-scene GI presets
- four camera-fitted, texel-stabilized directional shadow cascades with
  cross-faded boundaries and 5x5 tent PCF
- a production dynamic-scene extension: immutable object-space BLASes,
  per-frame current/swept/emissive TLASes, and motion-local cone invalidation

WebGPU does not expose a portable hardware ray-tracing pipeline. Rays and
shadow queries therefore traverse a binned-SAH triangle BVH in WGSL. This
changes the traversal backend, not the Split RC interval or merge math.
The equation-by-equation audit is in
[PAPER-COVERAGE.md](./PAPER-COVERAGE.md).

The paper assumes static geometry. Dynamic rigid objects are therefore labeled
as an extension, not misrepresented as paper baseline behavior. Static Sponza
remains one immutable world BLAS; four reusable local mesh BLASes feed 48
independent instances. The current TLAS, swept-change TLAS, emissive-only TLAS,
and packed instance records share the existing node/triangle arenas so the
compute pipeline requests nine compute-stage storage buffers after verifying
that exact adapter limit; this machine advertises sixteen.
Raster, four sun shadows, six point shadows, compute rays, and final `C(-1)`
visibility all consume the same frame's instance records.

## Stability

The shipped renderer has one diffuse Split RC configuration. The paper's
optional multibounce and rough-specular experiments are not compiled into the
production shader. The ambient-form `C(-1)` proposed in Section 7.1 is included
as an always-on production extension for topology-classified enclosed volumes
and closed-mesh back faces. Open receivers retain the smooth paper cascade plus
exact analytic near-emitter energy; it has no quality-changing UI toggle.

Stable mode advances a deterministic global low-discrepancy temporal rotation,
uses exact canonical screen-sample ranks, orders every allocation decision by
probe key, and uses the paper's hierarchical key-ordered Algorithm 3 offsets.
Directional `(J, beta)` intervals accumulate only on exact world/LOD key
matches. The paper begins its adjacent-LOD cross-fade at 0.9 of the boundary;
this production build widens residency to 0.75 so ordinary wheel impulses
cannot skip incoming support. There is no
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
as interpolation neighbors. Every key also carries a dominant-normal surface
sheet through c0-c3 so opposite sides of thin geometry cannot share radiance.
To remove the paper's documented nearest-only
quality tradeoff, c0 initialization allocates the four interpolation neighbors
in the dominant surface tangent plane while retaining one Algorithm 3 ray per
visible pixel. Zero-owner support probes reconstruct from all eight immediate
same-sheet tangent neighbors and exact-key world history, so camera motion
cannot toggle floor or wall coverage or abruptly swap a partial source set as
pixels cross a nearest probe boundary. The tone-mapped current Split RC result receives only
current-frame FXAA; there is no recursive screen-space presentation history.
These two invariants prevent a long camera translation from accumulating stale
block-shaped light and ensure that rebuilding history at a fixed pose produces
the same sparse population.

Rigid dynamic objects share the static path end to end: the same probes, the
same Algorithm 3 rays, the same merge, the same reconstruction. Motion is
expressed only as world-space cache invalidation. The swept-change TLAS holds
just the instances that actually moved, with bounds equal to the union of
their last twelve discrete pose AABBs; cones overlapping it blend history at
a capped effective sample count (a graceful decay instead of a pop) and
re-resolve from deterministic probe-keyed anchor rays, while
population-dependent screen rays feed only cones that still accumulate. A
light-corridor test additionally retires converged history whose stored
radiance embeds direct shading that a mover's sweep could have changed, so
indirect mover shadows cannot be frozen into the field. A dynamic object that
stops moving drains out of the swept hierarchy and becomes exactly static —
the persistent fixed-transport cache re-engages. Sub-probe-scale and
sub-pixel instances read a projected-size band-limited field, because neither
the probe grid nor the rasterizer supports directional detail below their
sampling limits. Moving emitters keep a compact emissive TLAS, and the
daylight door's hinged leaf is one reusable box BLAS instance under the same
rules. This rigid-object path is explicitly outside the paper.

For enclosed geometry below c0 spacing, an exact BVH `C(-1)` resolve evaluates
a rotation-balanced 14-point ambient quadrature. Blocker distance is filtered
across the finite c0 cone footprint, preserving a smooth scalar correction
without sparse directional color fans. Enclosed surfaces extend those
watertight rays through the complete scene bounds, with fixed world-space and
world-normal-only origins. The sealed-box audit requires zero displayed
luminance at every pixel while moving the camera and exact loop closure.

The GI configuration is universal across the complete scene suite. Base probe
spacing is derived automatically from asset bounds and triangle density; no
scene identity enters the cascade layout, temporal policy, C(-1) sample count,
or storage capacities. Scene-specific values describe only content (camera,
materials, lights, environment, and exposure), not GI behavior.

Stable mode advances the deterministic R2/Cranley-Patterson sequence
continuously. An odd 32-bit Weyl permutation supplies the global rotation, so
the 2D rotation pair cannot repeat before the full `2^32`-frame cycle and never
loses precision through a large float conversion. Paused lighting accumulates
an exact-key sample-count-weighted running average, capped at 65,535 effective
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
- a one-way staleness oracle that converges under continuous mover animation,
  holds the final pose, and must match a clean rebuild of the identical state
- simultaneous camera-path and mover animation, gated by owner-local
  Lagrangian acceleration with samples bucketed by projected instance size
- a deliberate near/far Sponza camera dolly aimed at view-sized distant meshes
- deterministic 512-spp, 64x36 cosine-weighted BVH references for the
  color-bleed lab, Sponza, Cornell, and the large concave heightmap
- raster sun and six-face point-shadow visibility compared sample-for-sample
  with exact software-BVH visibility, including an eight-position full sun
  sweep and mandatory coverage of all six point-shadow layers in Cornell
- sparse-hash, key-range, capacity, BVH-stack, and WebGPU error diagnostics

On the development NVIDIA RTX 5080, Sponza's world-keyed probe comparison under
camera motion measures 0.293% p95 relative change. Re-rendering the same poses
has 0/255 p95 and p99 delta (1/255 maximum). The uninterrupted final composite
measures 2/255 at p95 and 6/255 at p99; with moving lighting it remains 2/255
at p95 and 4/255 at p99. Rebuilding history at the same pose after the reported
low-camera translation measures 2/255 at p95 and 5/255 at p99. Every
per-capture sparse diagnostic is zero. These tests cover both cache stability
and the displayed image.

## Thirteen validation scenes

1. Color bleed laboratory - near-field transfer and emissive geometry.
2. Dynamic Sponza atrium - the official Crytek/Khronos geometry reconstructed with the
   paper's neutral/cyan presentation as 262,267 triangles, a 158,309-node SAH
   BVH, and a 25-layer 811px sRGB texture array with complete mip chains made
   from all official base-color materials, plus 48 moving rigid instances
   (seven emissive panels) sharing five local BLASes. No synthetic surface is
   baked into the reference asset.
   The paper does not distribute its exact
   prepared asset, camera, or exposure, so visual matching is not mislabeled
   as byte-identical scene provenance.
3. Concave canyon heightfield - a 72x72 terrain with craters and ravines.
4. Dense lantern forest - thousands of thin foliage triangles.
5. Multi-level atrium - stairs, balconies, skylight, and curved sculptures.
6. Industrial pipe maze - curved pipework, narrow gaps, and emitters.
7. Sun temple - layered portals and moving sunlight.
8. Orbital sculpture field - open-sky misses and high-curvature meshes.
9. Daylight door room - sealed-room darkness, animated aperture transport, and curved-normal detail.
10. Megacity stress grid - 12,992 triangles and high probe pressure.
11. Cornell box reference - canonical red/green enclosure, two occluders,
    ceiling area emitter, and a moving comparison light.
12. Grand concave heightmap - a 128x128 terrain with nested bowls, crater,
    ravine, terraces, shelves, and moving sun/fill light.
13. Universal visibility laboratory - broad and near-field area emitters,
    partial blockers, closed volumes, and explicitly open two-sided sheets.

Every scene has a deterministic camera path, moving sun, and moving colored
point light. The controls expose final, indirect, direct, normal, coverage, and
albedo views, four quality tiers, and stable-history diagnostics. There are no
optional shading-mode knobs.

## Measured result

Balanced mode bounds the internal screen to 360,000 pixels; the development
viewport resolves to 800x450 and traces one primary R2 ray per internal-screen
pixel:

- the final 13-scene audit passed at a 57.13 FPS minimum callback rate and a
  16.52 ms maximum measured GPU time on the validation GPU
- baseline and moving-light 32-frame motion gates measure 2/255 at p95
- the 512-spp Sponza reference gate measures 23.25% raw NRMSE, 21.37%
  99%-trimmed NRMSE, 13.14% low-frequency scale-invariant NRMSE, 0.0496
  scene-linear p95 error, 0.087% severe under-light outliers, and 0.349%
  bright-leak candidates
- the sealed-box audit measures zero luminance at every displayed pixel and
  exact same-pose loop closure while the camera moves inside the volume
- the point-shadow/BVH classification mismatch is 0% in the laboratory and
  Cornell scenes and 1.19% on the large heightmap; corresponding sun-shadow
  mismatch is 0%, 1.52%, and 1.04%
- no sparse-hash, capacity, BVH-stack, or WebGPU error was observed
- the dedicated dynamic-Sponza release bundle passes with 48 movers and seven
  mesh lights on the unified world-field path: owner-local Lagrangian
  acceleration on moving surfaces measures 1.5/255 at p95 (display) and
  0.00055 scene-linear (raw transport, a 22x margin under its gate) at a near
  pose projecting movers up to ~100 px, with zero screen-history blending;
  fixed static surfaces measure 1/255 acceleration p95 under moving lights;
  the one-way stale-shadow oracle closes at 2/255 p95 against a clean rebuild
  after 96 frames of continuous motion, and the simultaneous camera+object
  gate holds the same Lagrangian bounds along the animated camera path; CPU
  update p95 is 0.70 ms with 28,576 upload bytes/frame; GPU frame time is an
  interactive measurement (7.09 ms on the validation RTX 5080 — headless
  compositors present via CPU readback and cannot time the swapchain)

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
hard indirect detail, and cannot represent sharp mirror reflections. Rigid
dynamic instances, nonuniform scale, material overrides, and moving mesh
emitters are supported. Topology-changing/deforming meshes require a BLAS
refit or rebuild and are not claimed by this demo. These are stated method
limits, not silent feature omissions.

This is a from-paper WebGPU implementation, not the authors' unreleased source
code. WebGPU's portable software BVH and deterministic key-ordered allocation
replace native ray-tracing and CUDA-specific implementation details. The
paper's bordered irradiance field is copied from its storage write target into
a filterable RGBA16F texture atlas, so final diffuse reconstruction uses one
bilinear texture sample for each of at most four surface-tangent probe
neighbors per LOD. Moving point shadows are rendered into
six alpha-tested depth layers and sampled with explicit face/UV mapping; this
avoids backend-dependent cube-face orientation assumptions.
