import assert from "node:assert/strict";
import test from "node:test";
import {
  computeShader, finalShader, shaderConstants, temporalShader,
} from "../public/rc/shaders.js";

test("Algorithm 3 allocation, LOD overlap, and history are implemented", () => {
  assert.equal(shaderConstants.hashFrames, 2);
  assert.equal(shaderConstants.irradianceFrames, 2);
  assert.match(computeShader, /fn assignRayOffsets/);
  assert.match(computeShader, /RAY_COUNT_OFFSET/);
  assert.match(computeShader, /RAY_OFFSET_OFFSET/);
  assert.match(computeShader, /sequenceIndex=atomicLoad/);
  assert.match(computeShader, /let overlapStart = boundary \* 0\.9/);
  assert.match(computeShader, /lookupProbeFrame\(0u,keyFromCell\(cell\+bits,lod\),previousFrame\)/);
  assert.match(computeShader, /interval=mix\(interval,previousInterval,temporalWeight\)/);
  assert.match(computeShader, /accumIndexFrame\(cascade,previousProbe,direction,previousFrame\)/);
  assert.match(computeShader, /fn canonicalizeProbes/);
  assert.match(computeShader, /if\(otherKey!=EMPTY&&otherKey<key\)/);
  assert.match(computeShader, /fn countBaseRays/);
  assert.match(computeShader, /fn mapPrimaryRaySamples/);
  assert.match(computeShader, /fn prefixRayBlocks/);
  assert.match(computeShader, /fn deterministicLocalRank/);
  assert.match(computeShader, /if\(mappedProbe\(other\)==probe\)\{rank\+=1u;\}/);
  assert.match(computeShader, /let localRank=deterministicLocalRank/);
  assert.match(computeShader, /let sampleFrame=min\(passParams\.pad1&65535u,31u\)/);
  assert.match(computeShader, /var jitter=fract\(vec2f\(0\.5\)\+f32\(sampleFrame\+1u\)/);
  assert.match(computeShader, /fn intervalHistoryWeight/);
  assert.match(computeShader, /fn retainPreviousProbes/);
  assert.match(computeShader, /gid\.x>=HASH_FRAME_STRIDE/);
  assert.match(computeShader, /!featureEnabled\(32u\)&&sampleFrame>=32u/);
  assert.match(finalShader, /frameIndex\*IRRADIANCE_FRAME_STRIDE/);
  assert.match(temporalShader, /previousViewProj/);
  assert.match(temporalShader, /bestDistance>tolerance\*tolerance/);
  assert.equal(shaderConstants.irradianceTexels, 64);
});

test("paper extension paths are present in the production shader", () => {
  assert.match(computeShader, /fn sampleSecondaryHistory/);
  assert.match(computeShader, /fn initSecondary/);
  assert.match(computeShader, /fn splitSecondaryRays/);
  assert.match(finalShader, /fn roughSpecularLod/);
  assert.match(finalShader, /fn screenSpaceNearInterval/);
  assert.match(finalShader, /fn directionalCMinusOne/);
});

test("reference and final-frame quality gates are compiled into production", () => {
  assert.match(computeShader, /fn validateReference/);
  assert.match(computeShader, /samplePrimaryIrradiance/);
  assert.match(finalShader, /octCoordinate/);
});
