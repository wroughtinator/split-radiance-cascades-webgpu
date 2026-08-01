import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const engine = await readFile(new URL("../public/rc/engine.js", import.meta.url), "utf8");
const shaders = await readFile(new URL("../public/rc/shaders.js", import.meta.url), "utf8");
const scenes = await readFile(new URL("../public/rc/scenes.js", import.meta.url), "utf8");
const app = await readFile(new URL("../app/radiance-cascades-lab.tsx", import.meta.url), "utf8");
const standalone = await readFile(new URL("../standalone/index.html", import.meta.url), "utf8");

test("production audit gates camera motion, moving lights, floor loops, and sun shadows", () => {
  assert.match(engine, /compareReprojectedFrames/);
  assert.match(engine, /matchedPixelRatio >= 0\.35/);
  assert.match(engine, /trimmedRmseByteDelta/);
  assert.match(engine, /maximum\("p95Absolute"\) <= 0\.006/);
  assert.match(engine, /warmup: 384/);
  assert.doesNotMatch(engine, /warmup: i ===|warmup: index ===/);
  assert.match(engine, /p95ByteDeltaMax/);
  assert.match(engine, /movingLightContinuousMotion/);
  assert.match(engine, /runLongTranslationCacheAudit/);
  assert.match(engine, /cacheMotionRecovery/);
  assert.match(engine, /sparsePopulationMatched/);
  assert.match(engine, /failedMotionSamples === 0/);
  assert.match(engine, /comparison\.p95ByteDelta <= 12/);
  assert.match(engine, /comparison\.p999ByteDelta <= 128/);
  assert.match(engine, /recoveryDifference\.p95ByteDelta <= 4/);
  assert.match(engine, /r\.cacheMotionRecovery && !r\.cacheMotionRecovery\.passed/);
  assert.match(engine, /runMovingLightResponseAudit/);
  assert.match(engine, /responseRatio <= 0\.72/);
  assert.match(engine, /movingLights: true/);
  assert.match(engine, /runFloorRoundTripAudit/);
  assert.match(engine, /leg: direction > 0 \? "forward" : "backward"/);
  assert.match(engine, /const coverageExact = debugMode === 4/);
  assert.match(engine, /maximum\("maxByteDelta"\) === 0/);
  assert.match(engine, /maximum\("p95ByteDelta"\) <= 3/);
  assert.match(engine, /loopClosure\.p99ByteDelta <= 8/);
  assert.match(engine, /runViewDistanceInvarianceAudit/);
  assert.match(engine, /comparison\.p999ByteDelta <= 96/);
  assert.match(engine, /comparison\.largeDeltaRatio <= 0\.02/);
  assert.match(engine, /normalPixels/);
  assert.match(engine, /normalAgreement < 0\.88/);
  assert.match(engine, /const destinationWorld = new Float32Array/);
  assert.match(engine, /!r\.movingLightContinuousMotion\.passed/);
  assert.match(engine, /r\.movingLightResponse && !r\.movingLightResponse\.passed/);
  assert.match(engine, /runShadowMapAudit/);
  assert.match(engine, /runSunShadowSweepAudit/);
  assert.match(engine, /classificationMismatchRatioMax/);
  assert.match(engine, /r\.shadowMapCorrectness && !r\.shadowMapCorrectness\.passed/);
  assert.match(engine, /runEnclosureLeakAudit/);
  assert.match(engine, /displayLeak\.maximumLuminanceByte <= 1/);
  assert.match(engine, /displayLeak\.severePixelRatio === 0/);
  assert.match(engine, /report\.trajectory\.p95ByteDeltaMax <= 4/);
  assert.match(engine, /r\.enclosureLeak && !r\.enclosureLeak\.passed/);
  assert.match(engine, /classificationMismatchRatio <= 0\.06/);
  assert.match(engine, /pointFaceCoverageRequired = this\.sceneIndex === 10/);
  assert.match(engine, /point\.perFace\.every\(\(face\) => face\.samples >= 4\)/);
  assert.match(engine, /diagnosticOverflows === 0/);
  assert.match(engine, /per-capture sparse diagnostics/);
  assert.match(engine, /this\.sampleFrameIndex < 24/);
  assert.match(engine, /this\.animateLights[\s\S]*\? 0\.965/);
  assert.doesNotMatch(engine, /multibounce|roughSpecular|cMinusOne/);
  assert.match(engine, /animate-lights[\s\S]*this\.resetProbeHistory\(\)/);
  assert.doesNotMatch(engine, /retainProbes|retainPreviousProbes|temporalPipeline|historyTextures|previousWorldTexture/);
});

test("quality presets preserve Algorithm 3's one-ray-per-screen-pixel assignment", () => {
  assert.match(engine, /balanced: \{ giDivisor: 1,[^}]+raysPerSample: 1 \}/);
  assert.match(engine, /pass\.dispatchWorkgroups\(Math\.ceil\(this\.width \/ 8\), Math\.ceil\(this\.height \/ 8\)\)/);
  assert.match(engine, /const requestedQuality = new URLSearchParams/);
  assert.match(engine, /this\.adaptivePixelBudgetScale/);
  assert.match(engine, /adaptiveBudgetScale\(this\.adaptivePixelBudgetScale, observedMs\)/);
  assert.match(engine, /const vsyncRecoveryProbe = gpu <= 0/);
  assert.match(engine, /const observedMs = gpu > 0 \? gpu : \(vsyncRecoveryProbe \? 12\.5 : avg\)/);
  assert.match(engine, /firstAdjustment \? 30 : 60/);
  assert.match(engine, /const mipLevelCount = Math\.floor\(Math\.log2\(layerSize\)\) \+ 1/);
  assert.match(engine, /mipLevelCount,/);
  assert.match(engine, /maxAnisotropy: 16/);
});

test("path-reference gate classifies dark spots and bright leaks", () => {
  assert.match(engine, /severeUnderlitRatio/);
  assert.match(engine, /severeOverlitRatio/);
  assert.match(engine, /report\.p99Absolute <= 0\.40/);
  assert.match(engine, /report\.severeUnderlitRatio <= 0\.01/);
  assert.match(engine, /lowFrequencyScaleInvariantNrmse/);
  assert.match(engine, /report\.trimmedLowFrequencyScaleInvariantNrmse99 <= 0\.36/);
  assert.match(engine, /report\.trimmedNrmse99 <= 0\.34/);
  assert.match(engine, /report\.frozenBaselinePassed/);
  assert.match(engine, /const REFERENCE_BASELINES/);
  assert.match(engine, /report\.paperSceneStrictPassed/);
  assert.match(engine, /report\.nrmse <= 0\.25/);
  assert.match(engine, /report\.p99Absolute <= 0\.10/);
  assert.match(engine, /width = 64, height = 36, samples = 512/);
  assert.match(engine, /warmup: 192/);
  assert.match(engine, /renderReferenceComparison/);
});

test("sampling epochs replay deterministically without stale ray-map tags", () => {
  assert.match(engine, /this\.sampleFrameIndex = 0/);
  assert.match(engine, /this\.sampleEpoch = 1/);
  assert.match(engine, /this\.sampleFrameIndex >>> 0/);
  assert.match(engine, /this\.sampleEpoch >>> 0/);
  assert.match(engine, /this\.sampleFrameIndex = \(this\.sampleFrameIndex \+ 1\) >>> 0/);
  assert.match(engine, /this\.sampleEpoch = \(\(this\.sampleEpoch \+ 1\) >>> 0\) \|\| 1/);
  assert.doesNotMatch(engine, /sampleEpoch & 0xffff/);
});

test("presentation preserves shadow detail and height fields use smooth normals", () => {
  assert.match(shaders, /fn displayEncode\(x:vec3f\)->vec3f/);
  assert.match(shaders, /pow\(aces\(x\),vec3f\(1\.0\/2\.2\)\)/);
  assert.match(shaders, /color=displayEncode\(color\*frame\.controls\.y\)/);
  assert.match(scenes, /const normal=\(point\)=>normalize3/);
  assert.match(scenes, /const terracing=1\+Math\.tanh/);
  assert.doesNotMatch(scenes, /const terracing=Math\.floor/);
});

test("production module graph uses a release cache key", () => {
  const releaseKey = "v=2026-07-31-near-emitter1";
  assert.ok(app.includes(`/rc/engine.js?${releaseKey}`));
  assert.ok(standalone.includes(`/rc/engine.js?${releaseKey}`));
  assert.ok(engine.includes(`./math.js?${releaseKey}`));
  assert.ok(engine.includes(`./scenes.js?${releaseKey}`));
  assert.ok(engine.includes(`./shaders.js?${releaseKey}`));
  assert.ok(scenes.includes(`./math.js?${releaseKey}`));
});

test("world-space GI inputs retain full precision within the WebGPU MRT budget", () => {
  assert.match(engine, /"G-buffer full-precision world position",\s*"rgba32float"/);
  assert.doesNotMatch(engine, /this\.emissiveTexture\s*=/);
  assert.match(engine, /worldPixels: new Float32Array/);
  assert.match(shaders, /fn encodeNormalOct/);
  assert.match(shaders, /fn encodeSurfaceEmission/);
});
