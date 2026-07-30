import assert from "node:assert/strict";
import test from "node:test";
import { computeShader, finalShader, shaderConstants } from "../public/rc/shaders.js";

test("Algorithm 3 allocation, LOD overlap, and history are implemented", () => {
  assert.equal(shaderConstants.hashFrames, 2);
  assert.equal(shaderConstants.irradianceFrames, 2);
  assert.match(computeShader, /fn assignRayOffsets/);
  assert.match(computeShader, /RAY_COUNT_OFFSET/);
  assert.match(computeShader, /RAY_OFFSET_OFFSET/);
  assert.match(computeShader, /sequenceIndex=atomicLoad/);
  assert.match(computeShader, /let overlapStart = boundary \* 0\.9/);
  assert.match(computeShader, /lookupProbeFrame\(0u,keyFromCell\(cell\+bits,lod\),previousFrame\)/);
  assert.match(computeShader, /filtered=mix\(filtered,history,blend\)/);
  assert.match(computeShader, /fn canonicalizeProbes/);
  assert.match(computeShader, /for\(var previous=0u;previous<gid\.x;previous\+\+\)/);
  assert.match(computeShader, /fn countBaseRays/);
  assert.match(finalShader, /frameIndex\*IRRADIANCE_FRAME_STRIDE/);
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
