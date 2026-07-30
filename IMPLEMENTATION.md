# Implementation notes

## Frame graph

```text
Sun depth pass
      |
      v
G-buffer (world position, normal, albedo/emission, depth)
      |
      v
Reset sparse hash and fixed-point accumulators
      |
      v
Insert c0 probes from visible surfaces
      |
      v
Initialize c1 -> c3 nearest parents
      |
      v
Trace quality-dependent, world-keyed R2 rays per GI surface sample
      |
      v
Split each ray into (J, beta) cascade intervals and atomically deposit
      |
      v
Merge c3 -> c0, four angular children per lower direction,
with sparse trilinear spatial interpolation
      |
      v
Prefilter each c0 probe to a 6x6 octahedral irradiance field
      |
      v
Blend exact matching world probes from the previous frame
      |
      v
Sparse eight-probe gather + direct light + shadow + ACES composite
```

Each operation is an independent WebGPU pass, which gives explicit memory ordering between hash initialization, interval deposition, merging, and shading.

## Cascade configuration

For cascade `n`:

```text
probe spacing      Δs_n = Δs_0 · 2^n
direction count    |Ω_n| = 32 · 4^n
ray cutoff         t_n = 1.6 · Δs_0 · 4^n
```

The angular grid is `2Θ_n × Θ_n`, where `Θ_n = 4 · 2^n`, yielding 32, 128, 512, and 2048 directions. Direction samples use the paper's equal-area mapping:

```text
φ = 2πu
z = 2v - 1
r = sqrt(1 - z²)
ω = (r cos φ, r sin φ, z)
```

## Sparse probes

Probe keys pack three signed nine-bit grid coordinates plus a three-bit LOD. Each cascade has a separate open-addressed hash range. A compare/exchange claims an empty slot; the slot maps to a compact per-cascade index.

Probe centers are positioned at half-cell offsets. Final shading and cascade merging only use present trilinear neighbors and divide by their summed weight, preventing darkening at sparse boundaries.

Capacity:

| Cascade | Max probes | Hash slots | Directions/probe |
|---:|---:|---:|---:|
| c0 | 2048 | 4096 | 32 |
| c1 | 1024 | 4096 | 128 |
| c2 | 512 | 2048 | 512 |
| c3 | 256 | 2048 | 2048 |

The built-in audit fails visibly if insertion exceeds those caps.

## Ray splitting

Each valid GI sample traces a hemisphere ray from the visible surface. If a hit distance terminates in cascade `n`, lower cascades receive:

```text
J_k = 0
beta_k = 1,  k < n
```

and the terminating cascade receives:

```text
J_n = direct_radiance_at_hit
beta_n = 0
```

A miss deposits environment radiance into the final cascade. Atomic float addition is not portable in core WebGPU, so non-negative radiance and transmittance use 20.12 fixed-point `atomic<u32>` sums plus an integer sample count. Values are range-checked before conversion.

## Merge

For each lower direction, the four corresponding directions in the next cascade are averaged. Each higher direction is itself sampled with renormalized sparse trilinear interpolation. The interval is then composed with the distant cone:

```text
I_n = J_n + beta_n · average(interpolate(I_(n+1)))
```

Missing interval samples act as transparent intervals; missing higher probes fall back to environment radiance. This prevents undefined reads and isolated black pixels.

## BVH

Every procedural mesh is triangulated on the CPU. Triangle centroids are Morton-sorted and recursively partitioned into a balanced BVH with leaves of at most four triangles. Nodes and reordered triangles are uploaded once per scene. The paper's Sponza scene is prepacked at build time into a 262,267-triangle, 131,317-node BVH and delivered as a 9.3 MB same-origin compressed payload.

WGSL traversal uses a fixed 32-entry stack, sufficient for the balanced trees in the validation suite (maximum depth is below 16). Shadow rays use the same traversal.

## Stability

The R2 sequence is derived from quantized world position, normal, LOD, and the per-sample lane rather than the screen-pixel index. Camera motion therefore does not scramble a surface's directional sample set.

The sparse hash and 6×6 irradiance field are double-buffered. After the current frame is prefiltered, the shader looks up the same probe key in the previous frame and blends only an exact world/LOD match. Newly visible or LOD-changed probes reject history automatically. Scene changes clear both history frames.

This is intentionally not screen-space reprojection: camera motion cannot drag old light across geometry. A frozen color-bleed viewport produced zero changed channels across two captures 800 ms apart after settling.

The implementation does not use the paper's exact Algorithm 3 hierarchical
prefix-sum allocator. It uses deterministic world-keyed sequence assignment,
which is a WebGPU-portability tradeoff and is documented rather than presented
as full feature parity.

## Safety and recovery

- high-performance adapter request
- adapter-limit check before buffer creation
- shader/pipeline validation error scope
- uncaptured error monitoring
- device-loss reporting
- bounded hash probing and BVH traversal
- capacity and finite-value guards
- secure, same-origin Netlify headers
- no external runtime dependencies or network requests
