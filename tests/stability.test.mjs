import assert from "node:assert/strict";
import test from "node:test";
import {
  computeShader, finalShader, presentShader, rasterShader, shaderConstants,
} from "../public/rc/shaders.js";

test("Algorithm 3 allocation, LOD overlap, and history are implemented", () => {
  assert.equal(shaderConstants.hashFrames, 4);
  assert.equal(shaderConstants.irradianceFrames, 4);
  assert.match(computeShader, /fn assignRayOffsets/);
  assert.match(computeShader, /fn mortonDirectionIndex/);
  assert.match(computeShader, /let parentDirection=direction\*4u\+child/);
  assert.match(computeShader, /RAY_COUNT_OFFSET/);
  assert.match(computeShader, /RAY_OFFSET_OFFSET/);
  assert.match(computeShader, /sequenceIndex=atomicLoad/);
  assert.match(computeShader, /let overlapStart = boundary \* 0\.75/);
  assert.match(computeShader, /lookupProbeFrame\(cascade,key,previousFrame\)/);
  assert.match(computeShader, /interval=mix\(interval,previousInterval,temporalWeight\)/);
  // Graceful invalidation: swept-invalid cones keep history at a capped
  // effective sample count instead of hard rejection.
  assert.match(computeShader, /let boundedPrevious=min\(previousSamples,historySampleCap\)/);
  assert.match(computeShader, /historySampleCap=select\(12u,0u,featureEnabled\(2048u\)\)/);
  assert.match(computeShader, /resolvedSamples=min\(65535u,totalSamples\)/);
  assert.match(computeShader, /let storageScale=FIXED_SCALE\*f32\(storedSamples\)/);
  assert.match(computeShader, /atomicStore\(&accum\[base\+4u\],storedSamples\)/);
  assert.match(computeShader, /accumIndexFrame\(cascade,previousProbe,direction,previousFrame\)/);
  assert.match(computeShader, /fn canonicalizeProbes/);
  assert.doesNotMatch(computeShader, /resolveTangentSupport/);
  assert.match(computeShader, /if\(irradiance\.a>0\.001\)/);
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
  assert.match(computeShader, /epoch==\(passParams\.sampleEpoch&0x00ffffffu\)/);
  assert.doesNotMatch(computeShader, /bitcast<f32>\(passParams\.sampleEpoch\)/);
  assert.match(computeShader, /fn loadAtlasIrradianceBilinear/);
  assert.match(finalShader, /fn loadFinalAtlasIrradianceBilinear/);
  assert.doesNotMatch(computeShader, /textureSampleLevel\(\s*irradianceAtlasSampled/);
  assert.doesNotMatch(finalShader, /textureSampleLevel\(\s*irradianceAtlas/);
  assert.match(computeShader, /let temporalX=sampleFrame\*0x91e10da5u/);
  assert.match(computeShader, /let temporalY=sampleFrame\*0xc13fa9a9u/);
  assert.match(computeShader, /f32\(temporalX>>8u\),f32\(temporalY>>8u\)/);
  assert.match(computeShader, /fn intervalHistoryWeight/);
  assert.match(computeShader, /featureEnabled\(128u\)/);
  assert.match(computeShader, /fn dynamicConeHistoryValid/);
  assert.match(computeShader, /historySampleCap>0u/);
  assert.doesNotMatch(computeShader, /retainPreviousProbes/);
  assert.doesNotMatch(computeShader, /Preserve the complete cascade ancestry/);
  assert.match(computeShader, /gid\.x>=HASH_FRAME_STRIDE/);
  assert.doesNotMatch(computeShader, /sampleFrame>=32u/);
  assert.match(computeShader, /textureStore\(irradianceAtlasStorage,atlasCoordinate,stored\)/);
  assert.match(computeShader, /textureLoad\(irradianceAtlasSampled,tile\+vec2i\(low\.x,low\.y\),0\)/);
  assert.match(finalShader, /textureLoad\(irradianceAtlas,tile\+vec2i\(low\.x,low\.y\),0\)/);
  assert.match(presentShader, /textureSampleLevel\(currentComposite,currentCompositeSampler/);
  assert.doesNotMatch(presentShader, /previousComposite|previousViewProj|temporalFS/);
  assert.equal(shaderConstants.irradianceTexels, 64);
  assert.equal(shaderConstants.probeCaps[0], 16384);
  assert.equal(shaderConstants.irradianceAtlasFrameHeight, 2048);
  assert.deepEqual(shaderConstants.probeOffsets, [0, 16384, 20480, 21504]);
  assert.deepEqual(shaderConstants.dataOffsets, [0, 524288, 1048576, 1572864]);
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
  assert.equal(
    shaderConstants.stateWords,
    16 + shaderConstants.totalProbeMeta * 3
      + shaderConstants.probeCaps[0] * 128,
  );
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
  assert.match(computeShader, /fn surfaceClass\(normalIn:vec3f\)->u32/);
  assert.match(computeShader, /fn keyFromCellSurface/);
  assert.match(computeShader, /pixelSurfaceClass\(normal\)/);
  assert.match(computeShader, /fn setRigidSheetOverride/);
  assert.match(computeShader, /probeCell\(info\.xyz,cascade,lod\),lod,probeSurfaceClass\(info\)/);
  assert.match(computeShader, /let sheet=probeSurfaceClass\(info\)/);
  assert.match(computeShader, /sampleParentDirection\(cascade,position,lod,sheet,parentDirection\)/);
  assert.match(finalShader, /fn surfaceClass\(normalIn:vec3f\)->u32/);
  assert.match(finalShader, /keyFromCellSurface\(cell\+bits,lod,pixelSurfaceClass\(normal\)\)/);
  assert.match(finalShader, /fn cMinusOneIrradiance/);
  assert.match(finalShader, /fn traceShortRange/);
  assert.match(finalShader, /fn traceShortRangeWatertight/);
  assert.match(finalShader, /fn shortTriangleWatertight/);
  assert.match(finalShader, /var<storage,read> shortBvhNodes/);
  assert.match(finalShader, /const C_MINUS_DIRECTIONS=array<vec3f,14>/);
  assert.match(finalShader, /directionIndexValue=0u;directionIndexValue<14u/);
  assert.match(computeShader, /fn classifyEnvironmentAccess/);
  assert.match(computeShader, /atomicStore\(&state\[8\],1u\)/);
  assert.match(computeShader, /const ACCESS_RAY_COUNT=512u/);
  assert.match(computeShader, /let origin=frame\.cameraPos\.xyz\+direction/);
  assert.doesNotMatch(computeShader, /if\(world\.w<0\.5\)\{atomicStore\(&state\[8\]/);
  assert.doesNotMatch(finalShader, /atomicLoad\(&frameState\[8\]\)/);
  assert.match(finalShader, /let intervalEnd=frame\.envBaseSpacing\.w/);
  assert.match(finalShader, /frame\.sceneBounds\.w\*1\.001/);
  assert.doesNotMatch(finalShader, /needsLocalInterval|intervalEnd\*15\.0/);
  assert.doesNotMatch(computeShader, /primaryNeedsLocalInterval|intervalEnd\*15\.0/);
  assert.doesNotMatch(finalShader, /let continuation=sampleConeDirection\(world,normal,parentDirection\)/);
  assert.match(finalShader, /ambientVisible\/ambientWeight/);
  assert.match(finalShader, /if\(!closedBackFace\)/);
  assert.match(finalShader, /return baseIrradiance\*visibilityCorrection\+nearEmission/);
  assert.match(finalShader, /fn clippedTriangleFormFactor/);
  assert.match(finalShader, /var<storage,read> emissiveBvhNodes/);
  assert.match(finalShader, /nearEmissiveIrradiance\(origin,normal,intervalEnd\)/);
  assert.match(finalShader, /if\(visibleSamples<=0\.5\)\{return vec3f\(0\);\}/);
  assert.match(finalShader, /let intervalBlend=1\.0-smoothstep\(radius\*0\.72,radius,proximityDistance\)/);
  assert.match(finalShader, /fn emissivePatchIrradiance/);
  assert.match(finalShader, /for\(var i=0u;i<8u;i\+\+\)/);
  assert.match(finalShader, /let rayEpsilon=max\(/);
  assert.match(computeShader, /fn primaryClippedTriangleFormFactor/);
  assert.match(computeShader, /fn primaryEmissivePatchIrradiance/);
  assert.match(rasterShader, /let closedBackFace=!frontFacing&&v\.materialCutoff\.y< -0\.5/);
  assert.match(rasterShader, /let sourceVisible=dot\(normalize\(v\.normal\),frame\.cameraPos\.xyz-v\.world\)>0\.0/);
  assert.match(rasterShader, /let flags=select\(0u,1u,closedBackFace\)[\s\S]*visibleEmission/);
  assert.match(rasterShader, /packedSurface=0x800000u/);
  assert.doesNotMatch(finalShader, /compartmentSun|compartmentPoint|enclosureRadius/);
  assert.doesNotMatch(finalShader, /if\(hit\.t>=9999\.0\)\{return baseIrradiance;\}/);
  assert.match(computeShader, /let cMinusOneEnd=0\.0/);
  assert.match(computeShader, /let origin=surfaceOrigin\+direction\*cMinusOneEnd/);
  assert.match(computeShader, /radiance-hit\.emissive\.xyz/);
  assert.match(finalShader, /let origin=world\+normal\*max\(0\.006,intervalEnd\*0\.012\)/);
  assert.doesNotMatch(finalShader, /creaseSafeOrigin/);
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
  // 3x3 tent + hardware comparison filtering (6x6 effective footprint); the
  // former assertion matched a since-removed unrelated 5x5 loop.
  assert.match(finalShader, /let weight=\(2\.0-abs\(f32\(x\)\)\)\*\(2\.0-abs\(f32\(y\)\)\)/);
  assert.match(finalShader, /sampleSunShadowCascade\(world,normal,cascade\+1u\)/);
  assert.doesNotMatch(finalShader, /ndc\.z-0\.0015/);
});

test("reference and final-frame quality gates are compiled into production", () => {
  assert.match(computeShader, /fn validateReference/);
  assert.match(computeShader, /samplePrimaryIrradiance/);
  assert.match(computeShader, /fn resolvedPrimaryIrradiance/);
  assert.match(computeShader, /resolvedPrimaryIrradiance\(\s*world\.xyz,normal,packedClosedSurface\(world\.w\)\s*\)/);
  assert.match(finalShader, /octCoordinate/);
});
