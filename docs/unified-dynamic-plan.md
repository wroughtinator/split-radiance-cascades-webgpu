# Unified dynamic-receiver plan (WIP working notes)

Goal: dynamic surfaces use the SAME sparse world-space Split RC field as static
surfaces. Motion only invalidates world-space cache history (swept-change
TLAS). A dynamic object that does not move must be indistinguishable from
static geometry (no extra per-frame cost, converged accumulation). Zero camera
jitter and temporally consistent lighting under full object motion, with the
screen-history path remaining disabled (historyBlendCoverage === 0).

## What already exists (keep)

- Sparse hash probes keyed (cell, LOD, sheet); per-frame recreation; exact-key
  (J, beta) running-average accumulation with 65,535-sample cap
  (shaders.js mergeCascade ~2735).
- Swept-change TLAS built ONLY from instances whose transform changed this
  frame (dynamic.js:514). Empty swept TLAS => all history valid.
- Conservative swept-cone predicate with full spatio-angular support
  (dynamicConeRootClear ~980).
- Hazard-anchor machinery: cones whose support can see a rigid instance are
  re-sampled every frame with fixed per-probe-key R2 directions via 64 lanes x
  2 representative pixels (stableStaticProbeDirection ~2044,
  staticDirectionNeedsCurrentSample ~2062, selectStaticHazardRepresentatives).
  This makes near-mover cones deterministic per frame => no rotation flicker
  where history cannot accumulate.
- Feature bits: 512 hard transport discontinuity (reject all), 2048 rigid
  motion discontinuity, 4096 dynamic emission moving, 8192 preserve converged
  static cones, 16384 analytic source moving (EMA).

## Phase A: field-side fixes

- A1 Tight swept bounds: replace sphere-based sweep
  (conservativeRigidSweep, dynamic.js:68) with union of the previous and
  current exact 8-corner transformed AABBs. History only ever contains poses
  sampled at frame boundaries, so the endpoint-union is sufficient; no arc
  dilation is needed. Fixes room-scale invalidation from the hinged door.
- A2 Motion-state gating, not scene-type gating: preserveConvergedStaticCones
  (engine.js ~2073) and the persistent c0 cache currently require
  !this.dynamicScene. Gate them instead on "no instance moved and no dynamic
  emission change for K frames" so stationary dynamic scenes behave statically.
- A3 Endpoint-shading staleness: J embeds direct light at the hit point; a
  mover crossing the light path to the endpoint does not intersect the
  receiver cone and leaves stale shadows in converged history. Store a
  quantized mean hit distance per direction (spare 16 bits of the accum count
  word: count is capped at 65535) and reject history when the segment
  (hit point -> sun) or (hit point -> point light) intersects the swept TLAS.
  Conservative footprint = cone footprint at the hit distance.
- A5 Moving-emitter estimator: exact near-field polygon integration currently
  disables itself for MOVING dynamic emitters (bit 4096 skip in finalFS
  ~4335). Evaluate always using current transforms; the swept TLAS already
  invalidates stochastic history near the emitter, and deposits subtract
  near-source ownership at deposit time, so double counting stays bounded by
  the same ownership radius. Removes the on/off estimator pop at motion
  start/stop.

(A4 partial invalidation / M-cap is deferred: binary rejection + anchors
currently pass the acceleration gates; revisit only if far-cascade shimmer is
measurable after Phase B.)

## Phase B: unified receivers

- B1 Mover pixels join the paper path end to end: remove the dynamicReceiver
  early-return in splitRays (~2659) and the dynamic exclusion in hazard
  selection (~2148); mover pixels allocate probes (already true in initBase),
  count rays, trace, split, and deposit exactly like static pixels. Their
  surface probes see the mover itself => swept-invalid + anchored while
  moving, accumulate when still: the invalidation IS the dynamic path.
- B2 finalFS: dynamic receivers reconstruct from the same world field
  (samplePrimaryIrradianceLod + C(-1) + analytic near-emitter), no
  material-node lookup.
- B3 Delete the material-node subsystem: collectDynamicMaterialNodes,
  traceDynamicMaterialNodes (1024 rays/node), shadeDynamicMaterialNodes,
  dynamicReceiverAccumBuffer, dynamicReceiverIrradianceTexture, plus engine
  wiring. Repoint the dynamic-field audit at unified reconstruction validity.
- B4 Anti-pop determinism details:
  - Seed anchored direction sets by (cell, LOD, lane) EXCLUDING the sheet
    class so sheet flips under rotation do not switch the quadrature.
  - If cell-crossing kinks are measurable in the Lagrangian acceleration
    gates, decorrelate less: share one anchor sequence across all cells
    (seed by lane only) near movers so per-cell bias becomes spatially
    constant and motion cannot modulate it.
- B5 Gates: keep every existing gate green; add
  - stale-shadow one-way audit: converge, translate one mover A->B over many
    frames, hold, compare converged vs clean rebuild at B (catches A3).
  - stationary-equivalence audit: dynamic scene with zero motion for N frames
    must match the static path's camera-motion stability profile and stop
    tracing anchor rays (cost check via ray counters).

## Validation loop

Headless: scripts/headless-audit.mjs + scripts/run-gates.mjs against
http://localhost:8791/?autotest=<gate>. Primary gates: dynamic-sponza,
dynamic-field, dynamic-roundtrip, motion-1, floor-loop-1, door-zoom,
reference, cornell-artifacts, enclosure-leak, plus the two new audits.
