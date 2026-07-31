# Implementation notes

## Frame graph

```text
sun shadow pass + six-face moving-point shadow pass
  -> G-buffer (world, two-sided normal, albedo/emission, depth)
  -> clear current hash, counters, prefixes, and interval accumulators
  -> insert primary c0 probes from every visible internal-screen pixel
  -> insert secondary c0 probes from previous-frame ray hits at LOD + 2
  -> initialize c1 through c3 parents
  -> assign one ray per visible internal-screen pixel and count rays bottom-up
  -> assign deterministic hierarchical R2 offsets c3 through c0
  -> trace and split primary rays
  -> trace and split secondary-cache rays
  -> exact-key interval accumulation and merge c3 through c0
  -> prefilter c0 into a filterable bordered 6x6 octahedral atlas
  -> diffuse + C(-1) + rough specular + direct current composite
  -> present the current composite directly
```

Every arrow is a WebGPU pass boundary, providing ordering for atomic writes and
subsequent reads.

## Cascade configuration

For cascade `n` and LOD `d`:

```text
probe spacing   ds(n,d) = ds0 * 2^(n+d)
directions      |Omega_n| = 32 * 4^n
cutoff          t(n,d) = 1.6 * ds0 * 2^d * 4^n
```

The angular grid is `2*Theta_n` by `Theta_n`, where
`Theta_n = 4 * 2^n`. This yields 32, 128, 512, and 2048 equal-area
sphere directions.

## Sparse probes and LODs

Each cascade owns an open-addressed hash range. A compare/exchange claims an
empty slot and maps the key to a compact probe index. Keys contain signed
9-bit cell coordinates, a 3-bit LOD, and a primary/secondary cache tag.
Coordinates outside the representable scene-local range are rejected and
counted as an overflow; they are never silently clamped onto an unrelated key.

Probe centers are half-cell offsets. Merging and final gathers use only
existing trilinear neighbors and divide by the accumulated weight.
The current hierarchy contains only probes induced by current visible surfaces
and their parent chain. A matching previous key can supply interval history,
but it is never inserted merely because it existed in the prior frame. This
matches the paper's per-frame hashmap recreation and prevents stale off-screen
probes from becoming sparse interpolation neighbors.

LOD selection uses Chebyshev distance. A coarser LOD begins at 90% of its
nominal boundary; both LODs are initialized and evaluated through the overlap,
then linearly blended.

## Algorithm 3 on WebGPU

The base pass counts one visible ray contribution per internal-screen pixel
into its c0 probe. Higher passes sum
child counts into their nearest parent. Offsets are then assigned from c3 to c0
using probe-key order. Key order is important: compact indices are allocated
concurrently and are not stable between frames. Each lower-cascade probe
enumerates only its parent's eight possible children. The bounded c3 root
prefix scans at most 256 active probes.

For a c0 probe, each ray obtains:

```text
sequence_index = hierarchical_offset + exact_stable_local_rank
direction = DecodeDir(R2(sequence_index) + global_jitter)
```

The stable local rank is computed from canonical screen-sample slots, not
atomic arrival order. Stable mode advances a deterministic global
Cranley-Patterson rotation each frame, so interval history converges without
introducing wall-clock nondeterminism. Resetting history restarts that sequence
under a new epoch tag, preventing stale ray-map records from matching. Turning
stable history off uses a wall-clock global jitter and disables history for the
paper's single-frame inspection path.

The temporal rotation uses an odd 32-bit Weyl permutation whose low/high halves
approximate the paper's two irrational rotation increments. The complete pair
does not repeat before `2^32` frames and avoids float precision collapse.
Exact-key intervals therefore keep converging under paused lighting. Static
intervals use their accumulated sample counts, capped at 16,384 effective
samples; animated lighting keeps a bounded 0.92 EMA and is validated against a
stepped, freshly-converged target.

## Ray splitting and merge

If a surface ray terminates in cascade `n`, lower cascades receive:

```text
J_k = 0
beta_k = 1, for k < n
```

The terminating cascade receives the hit radiance and zero transmittance. A
miss terminates in c3 with environment radiance. Values are accumulated as
non-negative 20.12 fixed-point atomics because portable WebGPU does not expose
atomic float addition.

Merge evaluates:

```text
I_n = J_n + beta_n * average(interpolate(I_(n+1)))
```

The four angular children of each lower direction are contiguous. Directions
with no interval samples are invalid and do not participate in spatial or
angular interpolation. A traced sky miss explicitly deposits environment
radiance at c3; an absent parent gather remains invalid instead of inventing
environment light through missing coverage.

## Irradiance and extensions

Diffuse shading prefilters 32 c0 directions into a 6x6 octahedral field inside
an evaluated one-texel border (an 8x8 allocation). Every active probe writes
its tile into a double-buffered RGBA16F storage atlas. WebGPU requires writable
storage and filtered sampling to occupy separate usage resources, so the
completed frame half is copied into a filterable atlas before final and
secondary-cache gathers. Each gather performs one bilinear texture lookup per
sparse trilinear neighbor: at most eight filtered samples per LOD, including
octahedral interpolation.

The secondary cache consumes previous-frame primary hit points, initializes
probes two LODs coarser, and samples its previous irradiance at new ray hits.
The cache can therefore feed itself for temporally converged multiple bounces.

Rough specular selects c2 as a broad cone, samples it along the reflection
direction, then composes c1 and c0 `(J, beta)` intervals in front.

`C(-1)` evaluates 32 directional screen-space near intervals and composes them
in front of the world-space c0 cones, matching the directional extension
described by the paper.

## Stability and diagnostics

Hash, interval, and irradiance storage are double-buffered. Temporal reuse
looks up the exact previous world/LOD/cache key and blends the previous
directional `(J, beta)` interval before recursive merge; missing, disoccluded,
or LOD-changed probes reject history. Fixed lighting uses exact sample-count
accumulation; animated lighting uses a 0.92 EMA. Changing the
lighting-animation mode invalidates history.

The final current composite is presented directly. Temporal filtering exists
only in the paper's world-space `(J, beta)` interval history; the renderer does
not recursively blend tone-mapped screen pixels. Validation captures include
world position and normal only to compare the same surfaces across moving
frames, never to feed presentation history back into rendering.

Visible emissive radiance has its own RGB16F G-buffer target. Moving point
lights render alpha-tested geometry into a six-layer depth texture. Explicit
face/UV selection samples those layers, so visible direct lighting and GI
ray-hit lighting both respect occlusion without relying on backend cube-face
orientation.

The in-app audit includes twelve independent gates: exact-key world-probe
comparison; byte-for-byte replay of independently initialized camera
trajectories; world-position reprojection between every adjacent frame of an
uninterrupted moving-camera sequence; the same sequence with moving lights;
fixed-camera moving-light step response versus a fresh target;
the deliberate near/far Sponza dolly that covers view-sized surfaces; the
reported low-camera 6.48-unit Sponza translation followed by a same-pose clean
history rebuild, including exact per-cascade sparse-population equality; a
512-spp 64x36 cosine-weighted BVH reference on four representative scenes with
raw, robust, dark-spot, bright-leak, percentile, energy-bias, and frozen
per-scene regression gates; raster sun/point shadows compared with exact BVH
visibility; plus explicit per-capture hash/key/capacity/BVH/device diagnostics.
Cornell additionally requires the shadow oracle to exercise all six
point-shadow array layers.
Baseline and multibounce
Sponza paths both run
the replay and continuous-motion checks. Captures include the world and normal
buffers so opposite-facing surfaces are never paired as temporal
correspondences. A path-reference/Split-RC/error triptych is rendered with the
report. The audit also records FPS, GPU timestamps, ray/probe counts,
overflows, and WebGPU errors for every scene.

The reference classifier reports raw and 99%-trimmed NRMSE plus raw and trimmed
3x3 low-frequency metrics. Raw metrics remain visible and have frozen
scene-specific ceilings; robust metrics, p95/p99 absolute error, signed energy
bias, and under/over-light outlier ratios are independent mandatory gates.

## Safety and portability

- high-performance adapter request
- adapter-limit checks
- shader/pipeline validation error scope
- device-loss and uncaptured-error reporting
- bounded hash probing and exact near-first SAH-BVH traversal with explicit
  stack-overflow diagnostics
- capacity and finite-value guards
- same-origin assets and no runtime trackers
