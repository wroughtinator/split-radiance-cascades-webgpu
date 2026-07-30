# Implementation notes

## Frame graph

```text
sun shadow pass
  -> G-buffer (world, two-sided normal, albedo/emission, depth)
  -> clear current hash, counters, prefixes, and interval accumulators
  -> insert primary c0 probes from visible surfaces
  -> insert secondary c0 probes from previous-frame ray hits at LOD + 2
  -> initialize c1 through c3 parents
  -> count rays bottom-up
  -> assign deterministic hierarchical R2 offsets c3 through c0
  -> trace and split primary rays
  -> trace and split secondary-cache rays
  -> merge c3 through c0
  -> prefilter c0 into 6x6 octahedral irradiance
  -> exact-key temporal blend
  -> diffuse + C(-1) + rough specular + direct composite
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
sequence_index = hierarchical_offset + atomic_local_rank
direction = DecodeDir(R2(sequence_index) + global_jitter)
```

Stable mode sets the global jitter to zero. Turning stable history off restores
the paper's temporally changing global jitter for single-frame inspection.

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

Diffuse shading prefilters 32 c0 directions into a 6x6 octahedral field, then
uses at most eight sparse probe reads per LOD.

The secondary cache consumes previous-frame primary hit points, initializes
probes two LODs coarser, and samples its previous irradiance at new ray hits.
The cache can therefore feed itself for temporally converged multiple bounces.

Rough specular selects c2 as a broad cone, samples it along the reflection
direction, then composes c1 and c0 `(J, beta)` intervals in front.

`C(-1)` uses the paper's suggested ambient optimization: a fixed, symmetric
screen-space near-field gather modulates the world-space diffuse result. It
captures sub-c0 occlusion without the paper's expensive directional gather.

## Stability and diagnostics

Hash and irradiance storage are double-buffered. Temporal reuse first looks up
the exact previous world/LOD/cache key; missing, disoccluded, or LOD-changed
probes reject history. The default history weight is 0.96.

The in-app audit reads both world-probe frames back, joins them by key, and
reports median, p95, and p99 irradiance deltas while the camera moves and lights
are frozen. It also records FPS, GPU timestamps, ray/probe counts, overflows,
and WebGPU errors for every scene.

## Safety and portability

- high-performance adapter request
- adapter-limit checks
- shader/pipeline validation error scope
- device-loss and uncaptured-error reporting
- bounded hash probing and BVH traversal
- capacity and finite-value guards
- same-origin assets and no runtime trackers
