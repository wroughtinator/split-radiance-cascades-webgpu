# Paper implementation map

| Paper component | Native implementation | Validation |
|---|---|---|
| Four spatial/angular cascades | `constants.rs`, `split_rc.glsl` | exact direction-count and Morton parent-quartet tests |
| Equal-area spherical bins | `math.rs`, shader encode/decode functions | center/index round trips |
| Sparse view-dependent probes | two-word keys and five-word state-last GPU hash slots, compact deferred-publication retry, and tombstone canonicalization | boundary round trips, adversarial same-token publication stress, duplicate/unpublished/reserved-slot audit, overflow/collision/range counters |
| Base ray per visible pixel | `StateLayout`, collect/canonicalize/map passes | layout test asserts no spatial decimation |
| Hierarchical R2 assignment | `sampling.rs`, prefix/parent/offset passes | reversed allocation order produces identical ranks; production frame rotation advances without a short repeat |
| Distance-interval splitting | `interval.rs`, `cs_trace_split` | associativity and transparent-identity tests |
| Front-to-back merge | `cs_merge` | raw interval is persisted separately from composed cone; missing-plus-missing remains invalid instead of fabricating environment radiance |
| Spatial interpolation | shade trilinear lookup over at most eight sparse probes | missing weights are renormalized |
| LOD overlap | primary collection and final shade | continuity tests at the band boundary |
| Temporal accumulation | double-buffered exact-key raw accumulators; the current tone-mapped Split RC field is presented directly without recursive screen history | temporal rotation, key-stability audit, full-resolution Lab/Sponza repeated-camera-loop raw RGB delta gates, fixed-camera animated-light response, and exact Sponza translation/clean-rebuild gate |
| Multi-bounce secondary cache | previous-hit stream and irradiance feedback | separate primary/secondary key bit |
| Directional irradiance cache | filterable RGBA16F atlas with bordered 8×8 octahedral tiles | seam-border shader construction and 12-scene captures |
| Directional `C(-1)` experiment | screen-space near interval plus 32 equal-area directional integrations weighted by `4π/32` | constant-radiance Lambertian oracle, runtime `M` toggle, and startup environment switch |
| Rough specular extension | opt-in coarse cone plus near interval composition | disabled by default without roughness data; explicit environment opt-in regression; every native backend compiles from one shader |
| Software BVH | 16-bin SAH, four-triangle leaves, near-first closest-hit traversal, alpha-aware any-hit shadow traversal | BVH versus brute-force rays on every scene; identical 12-scene visual suite after optimization |
| Packed Sponza materials | UV interpolation, atlas material index, alpha cutoff inside shared traversal | asset metadata test; primary, sun, point, and indirect rays share alpha-tested traversal |

## Frame graph

1. Clear the current hash/meta tables, raw interval frame, counters, maps, and
   rank blocks.
2. Trace primary visibility and point/sun shadows through the software BVH.
3. Collect exact fine/coarse primary keys and previous-frame secondary keys.
4. Canonicalize compact probe indices, map owners, and prefix screen blocks.
5. Collect/canonicalize each parent cascade, propagate counts, and assign
   key-canonical hierarchical offsets.
6. Trace one unbounded R2 ray for every owner, split it into four distance
   intervals, and atomically deposit `(J, beta, count)`.
7. Match exact previous keys, accumulate/filter raw intervals, and merge
   cascades from coarse to fine while preserving invalid zero-count cones.
8. Build the bordered, hardware-filterable octahedral irradiance atlas.
9. Shade diffuse GI, optional directional `C(-1)`, and rough specular; tone map
   into RGBA8 and present.

## Validation boundary

The CPU audit proves deterministic math, scene construction, asset metadata,
finite lighting, hash publication behavior, radiometric quadrature, and BVH
agreement. Cross-compilation proves the selected native backends accept the
complete frame graph. D3D11 capture/readback verifies actual GPU output;
full-resolution repeated-pose gates measure raw-field temporal stability in
both the laboratory and Sponza; an indirect-only Sponza gate reproduces the
reported downward/forward translation and compares the accumulated final pose
with a clean same-pose rebuild; a separate fixed-camera Sponza gate proves that
moving sun and point lights still produce a strong output response; runtime
diagnostics expose overflow and publication failures. Performance remains
machine-specific and is recorded by repeated GPU timestamps distributed
through the measured interval plus callback throughput rather than inferred
from unit tests. Every machine-generated report carries executable,
shader-source, generated-shader, GPU-harness, and lockfile SHA-256 provenance.
