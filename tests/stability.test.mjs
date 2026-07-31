import assert from "node:assert/strict";
import test from "node:test";
import {
  computeShader, finalShader, presentShader, shaderConstants,
} from "../public/rc/shaders.js";

test("Algorithm 3 allocation, LOD overlap, and history are implemented", () => {
  assert.equal(shaderConstants.hashFrames, 2);
  assert.equal(shaderConstants.irradianceFrames, 2);
  assert.match(computeShader, /fn assignRayOffsets/);
  assert.match(computeShader, /fn mortonDirectionIndex/);
  assert.match(computeShader, /let parentDirection=direction\*4u\+child/);
  assert.match(computeShader, /RAY_COUNT_OFFSET/);
  assert.match(computeShader, /RAY_OFFSET_OFFSET/);
  assert.match(computeShader, /sequenceIndex=atomicLoad/);
  assert.match(computeShader, /let overlapStart = boundary \* 0\.9/);
  assert.match(computeShader, /lookupProbeFrame\(0u,keyFromCell\(cell\+bits,lod\),previousFrame\)/);
  assert.match(computeShader, /interval=mix\(interval,previousInterval,temporalWeight\)/);
  assert.match(computeShader, /let boundedPrevious=min\(previousSamples,16384u\)/);
  assert.match(computeShader, /resolvedSamples=min\(16384u,totalSamples\)/);
  assert.match(computeShader, /let storageScale=FIXED_SCALE\*f32\(storedSamples\)/);
  assert.match(computeShader, /atomicStore\(&accum\[base\+4u\],storedSamples\)/);
  assert.match(computeShader, /accumIndexFrame\(cascade,previousProbe,direction,previousFrame\)/);
  assert.match(computeShader, /fn canonicalizeProbes/);
  assert.match(computeShader, /let compactIndex=atomicAdd\(&state\[cascade\],1u\)/);
  assert.match(computeShader, /probeKeyFromInfo\(otherInfo,cascade\)<key/);
  assert.match(computeShader, /for\(var child=0u;child<8u;child\+\+\)/);
  assert.match(computeShader, /let sibling=lookupProbe\(cascade,siblingKey\)/);
  assert.match(computeShader, /fn countBaseRays/);
  assert.match(computeShader, /fn mapPrimaryRaySamples/);
  assert.match(computeShader, /fn prefixRayBlocks/);
  assert.match(computeShader, /fn deterministicLocalRank/);
  assert.match(computeShader, /if\(mappedProbe\(other\)==probe\)\{rank\+=1u;\}/);
  assert.match(computeShader, /let localRank=deterministicLocalRank/);
  assert.match(computeShader, /let sampleFrame=passParams\.sampleFrame/);
  assert.match(computeShader, /bitcast<u32>\(entry\.y\)==passParams\.sampleEpoch/);
  assert.match(computeShader, /let scrambledFrame=sampleFrame\*0x91e1c141u/);
  assert.match(computeShader, /f32\(scrambledFrame>>16u\)/);
  assert.match(computeShader, /fn intervalHistoryWeight/);
  assert.match(computeShader, /return historyWeight\(\)/);
  assert.doesNotMatch(computeShader, /retainPreviousProbes/);
  assert.doesNotMatch(computeShader, /Preserve the complete cascade ancestry/);
  assert.match(computeShader, /gid\.x>=HASH_FRAME_STRIDE/);
  assert.doesNotMatch(computeShader, /sampleFrame>=32u/);
  assert.match(computeShader, /textureStore\(irradianceAtlasStorage,atlasCoordinate,stored\)/);
  assert.match(computeShader, /textureSampleLevel\(\s*irradianceAtlasSampled,irradianceAtlasSampler,atlasUv,0\.0/);
  assert.match(finalShader, /textureSampleLevel\(irradianceAtlas,irradianceSampler,atlasUv,0\.0\)/);
  assert.match(presentShader, /textureLoad\(currentComposite/);
  assert.doesNotMatch(presentShader, /previousComposite|previousViewProj|temporalFS/);
  assert.equal(shaderConstants.irradianceTexels, 64);
  assert.equal(shaderConstants.probeCaps[0], 4096);
  assert.match(finalShader, /fn sampleIrradianceLod/);
  assert.doesNotMatch(finalShader, /fn sampleIrradianceWithCoverage/);
  assert.match(finalShader, /var emissiveTex: texture_2d<f32>/);
  assert.match(finalShader, /fn pointShadowVisibility/);
  assert.match(finalShader, /var pointShadowTex: texture_depth_2d_array/);
  assert.match(finalShader, /fn pointShadowArrayCoordinate/);
  assert.doesNotMatch(finalShader, /texture_depth_cube/);
  assert.match(computeShader, /fn validateShadowMaps/);
  assert.match(computeShader, /pointShadowAuditVisibility/);
  assert.match(computeShader, /sunShadowAuditVisibility/);
  assert.doesNotMatch(finalShader, /position\+normal\*frame\.envBaseSpacing\.w\*0\.35/);
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
