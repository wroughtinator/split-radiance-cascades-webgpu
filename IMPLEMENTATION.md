# Implementation notes

## Frame graph

```text
sun shadow pass
  -> G-buffer (world, two-sided normal, albedo/emission, depth)
  -> clear current hash, counters, prefixes, and interval accumulators
  -> retain the bounded prior sparse hierarchy inside a view guard
  -> insert primary c0 probes from visible surfaces
  -> insert secondary c0 probes from previous-frame ray hits at LOD + 2
  -> initialize c1 through c3 parents
  -> count rays bottom-up
  -> assign deterministic hierarchical R2 offsets c3 through c0
  -> trace and split primary rays
  -> trace and split secondary-cache rays
  -> exact-key interval accumulation and merge c3 through c0
  -> prefilter c0 into 6x6 octahedral irradiance
  -> diffuse + C(-1) + rough specular + direct current composite
  -> world-position/normal-validated temporal resolve
  -> present
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

Probe centers are half-cell offsets. Merging and final gathers use only
existing trilinear neighbors and divide by the accumulated weight.
Prior primary, secondary, and parent probes that remain inside a bounded
frustum guard are retained before current visibility insertion. This preserves
the complete sparse interpolation ancestry while keeping the cache bounded.

LOD selection uses Chebyshev distance. A coarser LOD begins at 90% of its
nominal boundary; both LODs are initialized and evaluated through the overlap,
then linearly blended.

## Algorithm 3 on WebGPU

The base pass counts visible ray contributions per c0 probe. Higher passes sum
child counts into their nearest parent. Offsets are then assigned from c3 to c0
using probe-key order. Key order is important: compact indices are allocated
concurrently and are not stable between frames.

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

After 32 rotations, stable mode freezes the global rotation. If lighting is
also paused, exact-key intervals lock at their converged value instead of being
recomputed from a view-dependent pixel set; newly visible keys are still
initialized immediately. Animated lighting keeps the 0.96 update blend active.

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

The four angular children of each lower direction are contiguous. Missing
interval samples are transparent; a completely missing parent gather falls
back to environment radiance.

## Irradiance and extensions

Diffuse shading prefilters 32 c0 directions into a 6x6 octahedral field inside
an evaluated one-texel border (an 8x8 allocation), then uses at most eight
sparse probe reads per LOD.

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
or LOD-changed probes reject history. The default history weight is 0.96.

The final sparse-to-screen reconstruction is also double-buffered. The current
world position is reprojected into the previous view, a 3x3 search chooses the
closest prior surface, and both world distance and normal agreement must pass
before history is accepted. Static-light presentation uses at most 0.96
history; animated lighting uses at most 0.84. Raw and 99.5%-trimmed RMSE are
both reported so ambiguous disocclusion-edge correspondences remain visible
without being mislabeled as global shimmer.

The in-app audit includes four independent checks: exact-key world-probe
comparison; byte-for-byte replay of independently initialized camera
trajectories; world-position reprojection between every adjacent frame of an
uninterrupted moving-camera sequence; and a 128-spp cosine-weighted BVH
reference with dark-spot, bright-leak, percentile, and energy-bias
classification. Baseline and multibounce Sponza paths both run the replay and
continuous-motion checks. A path-reference/Split-RC/error triptych is rendered
with the report. The audit also records FPS, GPU timestamps, ray/probe counts,
overflows, and WebGPU errors for every scene.

The reference classifier reports both raw pixel error and a 3x3 low-frequency
metric. The latter is the gating scale-invariant metric because Split RC
reconstructs low-frequency irradiance; the raw metric, p95/p99 absolute error,
and under/over-light outlier ratios remain mandatory report fields.

## Safety and portability

- high-performance adapter request
- adapter-limit checks
- shader/pipeline validation error scope
- device-loss and uncaptured-error reporting
- bounded hash probing and exact near-first SAH-BVH traversal with explicit
  stack-overflow diagnostics
- capacity and finite-value guards
- same-origin assets and no runtime trackers
