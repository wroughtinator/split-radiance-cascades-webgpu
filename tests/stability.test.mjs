import assert from "node:assert/strict";
import test from "node:test";
import {
  computeShader, finalShader, presentShader, rasterShader, shaderConstants,
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
  assert.match(computeShader, /lookupProbeFrame\(cascade,key,previousFrame\)/);
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
  assert.match(presentShader, /textureSampleLevel\(currentComposite,currentCompositeSampler/);
  assert.doesNotMatch(presentShader, /previousComposite|previousViewProj|temporalFS/);
  assert.equal(shaderConstants.irradianceTexels, 64);
  assert.equal(shaderConstants.probeCaps[0], 6144);
  assert.equal(shaderConstants.irradianceAtlasFrameHeight, 768);
  assert.deepEqual(shaderConstants.probeOffsets, [0, 6144, 7680, 8192]);
  assert.deepEqual(shaderConstants.dataOffsets, [0, 196608, 393216, 655360]);
  assert.equal(
    shaderConstants.totalDirectionData,
    shaderConstants.probeCaps.reduce(
      (sum, probes, cascade) => sum + probes * shaderConstants.directions[cascade],
      0,
    ),
  );
  assert.equal(
    shaderConstants.irradianceAtlasFrameHeight * shaderConstants.irradianceAtlasWidth,
    shaderConstants.probeCaps[0] * shaderConstants.irradianceTexels,
  );
  assert.equal(shaderConstants.stateWords, 16 + shaderConstants.totalProbeMeta * 3);
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
  assert.match(computeShader, /fn insertTangentSupport/);
  assert.match(computeShader, /for\(var corner=0u;corner<4u;corner\+\+\)/);
  assert.match(computeShader, /insertTangentSupport\(world\.xyz,normal,fine\)/);
  assert.match(computeShader, /insertTangentSupport\(world\.xyz,normal,coarse\)/);
  assert.match(finalShader, /total\/max\(normalWeight,1e-5\)/);
  assert.match(rasterShader, /texture_2d_array<f32>/);
  assert.match(rasterShader, /textureSampleGrad/);
});

test("optional experiment paths are absent from the production baseline", () => {
  assert.doesNotMatch(computeShader, /sampleSecondaryHistory|initSecondary|splitSecondaryRays/);
  assert.doesNotMatch(finalShader, /roughSpecular|screenSpaceNearInterval|directionalCMinusOne/);
  assert.doesNotMatch(computeShader, /SECONDARY_TAG|featureEnabled\(1u\)|featureEnabled\(4u\)/);
  assert.match(computeShader, /return TOTAL_PROBE_META/);
  assert.match(computeShader, /samplesPerFrame\(\)\*2u/);
});

test("stabilized directional shadows use four blended array cascades", () => {
  assert.match(computeShader, /texture_depth_2d_array/);
  assert.match(finalShader, /matrices: array<mat4x4<f32>,4>/);
  assert.match(finalShader, /for\(var y=-2;y<=2;y\+\+\)/);
  assert.match(finalShader, /sampleSunShadowCascade\(world,normal,cascade\+1u\)/);
  assert.doesNotMatch(finalShader, /ndc\.z-0\.0015/);
});

test("reference and final-frame quality gates are compiled into production", () => {
  assert.match(computeShader, /fn validateReference/);
  assert.match(computeShader, /samplePrimaryIrradiance/);
  assert.match(finalShader, /octCoordinate/);
});
