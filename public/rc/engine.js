import {
  adaptiveBudgetScale, add3, clamp, cross3, dot3, mat4LookAt, mat4Multiply, mat4Ortho,
  mat4Perspective, mul3, normalize3, sub3,
} from "./math.js?v=2026-08-01-dynamic-tlas10";
import {
  createScene, SCENE_INFO,
} from "./scenes.js?v=2026-08-01-dynamic-tlas10";
import { createDynamicScene } from "./dynamic.js?v=2026-08-01-dynamic-tlas10";
import {
  computeShader, finalShader, presentShader, rasterShader, shaderConstants as K,
} from "./shaders.js?v=2026-08-01-dynamic-tlas10";

const SUN_CASCADE_COUNT = 4;

const GPU = globalThis.GPUBufferUsage;
const TEX = globalThis.GPUTextureUsage;
const MAP = globalThis.GPUMapMode;
const SHADER = globalThis.GPUShaderStage;

const QUALITY = {
  // Bound each preset by pixels as well as DPR. This makes Balanced the same
  // workload in a small embedded browser, a Retina MacBook, and a maximized
  // 4K Edge window instead of silently multiplying fill and ray cost.
  //
  // Algorithm 3 assigns one R2 ray to every visible screen pixel. Earlier
  // presets emulated the same ray count on a sparse grid; that was fast, but
  // missed thin/distant surfaces and produced view-dependent environment
  // leakage. Every preset now keeps the paper's full-screen assignment and
  // varies the bounded internal screen resolution instead.
  performance: { giDivisor: 1, pixelRatio: 0.72, maxPixels: 220_000, shadow: 768, raysPerSample: 1 },
  balanced: { giDivisor: 1, pixelRatio: 0.9, maxPixels: 360_000, shadow: 1024, raysPerSample: 1 },
  quality: { giDivisor: 1, pixelRatio: 1, maxPixels: 620_000, shadow: 1536, raysPerSample: 1 },
  ultra: { giDivisor: 1, pixelRatio: 1, maxPixels: 1_000_000, shadow: 2048, raysPerSample: 1 },
};

// Frozen 512-spp, 64x36 reference ceilings for the four validation scenes.
// Raw energy NRMSE is retained because it reveals rare bright-emitter misses,
// but the paired robust/percentile gates prevent one emitter pixel from
// masking regressions across the rest of the irradiance field.
const REFERENCE_BASELINES = Object.freeze({
  0: Object.freeze({
    // The open room uses the smooth paper cascade field while the exact C(-1)
    // path is reserved for locally closed volumes. Freeze the resulting
    // 512-spp comparison with roughly 10-20% deterministic headroom.
    nrmse: 0.22, trimmedNrmse99: 0.10,
    lowFrequencyScaleInvariantNrmse: 0.18,
    trimmedLowFrequencyScaleInvariantNrmse99: 0.14,
    p95Absolute: 0.05, p99Absolute: 0.15,
  }),
  1: Object.freeze({
    // Surface-sheet keys and tangent-complete c0 support remove the coverage
    // holes while keeping the paper-scene comparison tightly bounded.
    nrmse: 0.25, trimmedNrmse99: 0.22,
    lowFrequencyScaleInvariantNrmse: 0.15,
    trimmedLowFrequencyScaleInvariantNrmse99: 0.14,
    p95Absolute: 0.06, p99Absolute: 0.10,
  }),
  10: Object.freeze({
    nrmse: 0.83, trimmedNrmse99: 0.26,
    lowFrequencyScaleInvariantNrmse: 0.45,
    trimmedLowFrequencyScaleInvariantNrmse99: 0.36,
    p95Absolute: 0.06, p99Absolute: 0.12,
  }),
  11: Object.freeze({
    nrmse: 0.29, trimmedNrmse99: 0.28,
    lowFrequencyScaleInvariantNrmse: 0.19,
    trimmedLowFrequencyScaleInvariantNrmse99: 0.18,
    p95Absolute: 0.07, p99Absolute: 0.15,
  }),
});

// Release-regression views reconstructed from the two reported Cornell
// failures. They are camera tests only; no scene identity enters GI settings.
const CORNELL_ARTIFACT_POSES = Object.freeze({
  "cornell-skylight": Object.freeze({
    position: Object.freeze([-0.35, 1.8, 2.1]),
    target: Object.freeze([0.0, 3.1, -3.3]),
  }),
  "cornell-box-gap": Object.freeze({
    position: Object.freeze([-0.1, 4.65, -0.25]),
    target: Object.freeze([0.0, 1.25, -3.25]),
  }),
});

const $ = (id) => document.getElementById(id);

function setStatus(title, detail, error = false) {
  const card = $("status-card");
  card?.classList.remove("hidden");
  if ($("status-title")) $("status-title").textContent = title;
  if ($("status-detail")) $("status-detail").textContent = detail;
  if (error && card) card.style.borderColor = "rgba(255,110,80,.55)";
}

function hideStatus() {
  $("status-card")?.classList.add("hidden");
}

function createBuffer(device, label, size, usage, data) {
  const aligned = Math.max(4, Math.ceil(size / 4) * 4);
  const buffer = device.createBuffer({ label, size: aligned, usage, mappedAtCreation: !!data });
  if (data) {
    const destination = new Uint8Array(buffer.getMappedRange());
    destination.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
    buffer.unmap();
  }
  return buffer;
}

function halfToFloat(value) {
  const sign = (value & 0x8000) ? -1 : 1;
  const exponent = (value >> 10) & 0x1f;
  const mantissa = value & 0x03ff;
  if (exponent === 0) return sign * 2 ** -14 * (mantissa / 1024);
  if (exponent === 31) return mantissa ? NaN : sign * Infinity;
  return sign * 2 ** (exponent - 15) * (1 + mantissa / 1024);
}

function transformPoint(matrix, point) {
  const [x, y, z] = point;
  return [
    matrix[0] * x + matrix[4] * y + matrix[8] * z + matrix[12],
    matrix[1] * x + matrix[5] * y + matrix[9] * z + matrix[13],
    matrix[2] * x + matrix[6] * y + matrix[10] * z + matrix[14],
    matrix[3] * x + matrix[7] * y + matrix[11] * z + matrix[15],
  ];
}

function buildSunShadowCascades({
  cameraPosition,
  cameraTarget,
  sunDirection,
  nearPlane,
  farPlane,
  aspect,
  sceneRadius,
  shadowResolution,
}) {
  const forward = normalize3(sub3(cameraTarget, cameraPosition));
  const referenceUp = Math.abs(dot3(forward, [0, 1, 0])) > 0.98 ? [0, 0, 1] : [0, 1, 0];
  const right = normalize3(cross3(forward, referenceUp));
  const up = normalize3(cross3(right, forward));
  const fractions = [0.25, 0.5, 0.75, 1.0];
  const splitDepths = fractions.map((fraction) => {
    const logarithmic = nearPlane * ((farPlane / nearPlane) ** fraction);
    const uniform = nearPlane + (farPlane - nearPlane) * fraction;
    return logarithmic * 0.65 + uniform * 0.35;
  });
  const matrices = [];
  const texelSizes = [];
  const tangent = Math.tan(Math.PI / 6);
  let previousSplit = nearPlane;

  for (let cascade = 0; cascade < SUN_CASCADE_COUNT; cascade++) {
    // Overlap adjacent slices so the fragment shader can cross-fade the last
    // 15% of each cascade without ever sampling outside the next one.
    const sliceNear = cascade === 0 ? nearPlane : previousSplit * 0.82;
    const sliceFar = splitDepths[cascade];
    const corners = [];
    for (const depth of [sliceNear, sliceFar]) {
      const halfHeight = tangent * depth;
      const halfWidth = halfHeight * aspect;
      const center = add3(cameraPosition, mul3(forward, depth));
      for (const vertical of [-1, 1]) {
        for (const horizontal of [-1, 1]) {
          corners.push(add3(
            add3(center, mul3(right, halfWidth * horizontal)),
            mul3(up, halfHeight * vertical),
          ));
        }
      }
    }
    const center = corners.reduce((sum, corner) => add3(sum, corner), [0, 0, 0])
      .map((value) => value / corners.length);
    const rawRadius = Math.max(...corners.map((corner) => Math.hypot(...sub3(corner, center))));
    // A quantized bounding sphere makes the projection invariant to camera
    // rotation; only its snapped light-space translation can change.
    const radius = Math.ceil(rawRadius * 16) / 16 * 1.03;
    const lightDistance = sceneRadius * 2.2 + radius + 25;
    const lightPosition = sub3(center, mul3(sunDirection, lightDistance));
    const lightUp = Math.abs(sunDirection[1]) > 0.95 ? [0, 0, 1] : [0, 1, 0];
    const lightView = mat4LookAt(lightPosition, center, lightUp);
    const lightFar = lightDistance + sceneRadius * 2.4 + radius + 25;
    const lightProjection = mat4Ortho(-radius, radius, -radius, radius, 0.1, lightFar);
    const viewProjection = mat4Multiply(lightProjection, lightView);

    // Lock the global origin to the shadow texel lattice. This removes the
    // sub-texel crawl that otherwise occurs during camera translation.
    const origin = transformPoint(viewProjection, [0, 0, 0]);
    const halfResolution = shadowResolution * 0.5;
    const originX = origin[0] / Math.max(1e-8, origin[3]) * halfResolution;
    const originY = origin[1] / Math.max(1e-8, origin[3]) * halfResolution;
    viewProjection[12] += (Math.round(originX) - originX) / halfResolution;
    viewProjection[13] += (Math.round(originY) - originY) / halfResolution;
    matrices.push(viewProjection);
    texelSizes.push((radius * 2) / shadowResolution);
    previousSplit = splitDepths[cascade];
  }
  return { matrices, splitDepths, texelSizes, forward };
}

function makePassBuffer(device, cascade) {
  return createBuffer(
    device,
    `cascade-${cascade}-params`,
    32,
    GPU.UNIFORM | GPU.COPY_DST,
    new Uint32Array([cascade, 0, 0, 0, 0, 1, 0, 0]),
  );
}

class SplitRadianceCascades {
  constructor(canvas) {
    this.canvas = canvas;
    this.context = null;
    this.adapter = null;
    this.device = null;
    this.format = null;
    this.scene = null;
    this.sceneIndex = 0;
    const requestedQuality = new URLSearchParams(location.search).get("quality");
    this.qualityName = Object.hasOwn(QUALITY, requestedQuality) ? requestedQuality : "balanced";
    this.automaticTest = new URLSearchParams(location.search).has("autotest");
    this.adaptivePixelBudgetScale = 1;
    this.lastAdaptiveAdjustmentFrame = 0;
    this.indirectStrength = 1;
    this.sunSpeed = 1;
    this.debugMode = 0;
    this.temporalStability = true;
    this.historyValid = false;
    this.animateCamera = true;
    this.animateLights = true;
    this.running = false;
    this.destroyed = false;
    this.frameIndex = 0;
    // frameIndex is the lifetime presentation counter.  Sampling needs a
    // separate, restartable counter so audits and history resets replay the
    // same low-discrepancy temporal sequence byte-for-byte.  The epoch keeps
    // stale ray-map entries from a prior reset from matching the new frame.
    this.sampleFrameIndex = 0;
    this.sampleEpoch = 1;
    this.startTime = performance.now();
    this.lastTime = this.startTime;
    this.frameSamples = [];
    this.gpuSamples = [];
    this.passTimes = { frame: 0, geometry: 0, gi: 0, composite: 0 };
    this.probeCounts = [0, 0, 0, 0];
    this.rayCount = 0;
    this.hitCount = 0;
    this.overflowCount = 0;
    this.profilePending = false;
    this.statusPending = false;
    this.readbackPause = false;
    this.loadingScene = null;
    this.loadToken = 0;
    this.mouse = { down: false, x: 0, y: 0 };
    this.camera = { azimuth: 0.75, elevation: 0.32, distance: 20, target: [0, 0, 0] };
    this.keys = new Set();
    this.cleanup = [];
    this.testTimeOverride = null;
    this.testFrameTime = null;
    this.testFrameStep = null;
    this.captureRequest = null;
    this.dynamicCpuSamples = [];
    this.dynamicUpdateMs = 0;
    this.dynamicUploadBytes = 0;
  }

  async initialize() {
    if (!navigator.gpu) {
      throw new Error("WebGPU is not available in this browser. Use current Chrome or Edge with hardware acceleration enabled.");
    }
    this.adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (this.destroyed) return false;
    if (!this.adapter) throw new Error("No WebGPU adapter was returned by the browser.");
    const requiredFeatures = [];
    if (this.adapter.features.has("timestamp-query")) requiredFeatures.push("timestamp-query");
    this.device = await this.adapter.requestDevice({ requiredFeatures });
    if (this.destroyed) {
      this.device.destroy();
      return false;
    }
    this.device.addEventListener("uncapturederror", (event) => {
      const message = String(event.error?.message || event.error);
      console.error(`[Split RC] WebGPU validation error: ${message}`);
      this.lastGpuError = message;
    });
    this.device.lost.then((info) => {
      if (!this.destroyed) setStatus("GPU device lost", `${info.message || "The graphics device was reset."} Reload to recover.`, true);
    });
    if (this.device.limits.maxStorageBuffersPerShaderStage < 8) {
      throw new Error("This GPU exposes fewer than the eight storage buffers required by the Split RC pipeline.");
    }

    this.context = this.canvas.getContext("webgpu");
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.context.configure({ device: this.device, format: this.format, alphaMode: "opaque" });
    this.timestampSupported = this.device.features.has("timestamp-query");
    await this.createPipelines();
    if (this.destroyed) return false;
    await this.createMaterialAtlas();
    if (this.destroyed) return false;
    this.createPersistentResources();
    if (this.destroyed) return false;
    this.installUI();
    await this.loadScene(1);
    if (this.destroyed) return false;
    this.running = true;
    this.lastTime = performance.now();
    requestAnimationFrame((t) => this.frame(t));
    hideStatus();
    this.updateAdapterLabel();
    console.info("[Split RC] renderer-ready", this.adapterInfo());

    const automaticTest = new URLSearchParams(location.search).get("autotest");
    if (automaticTest === "enclosure-leak") {
      setTimeout(async () => {
        await this.loadScene(0);
        this.exposeTestReport(await this.runEnclosureLeakAudit({ preservePose: true }));
      }, 200);
    } else if (automaticTest === "cornell-artifacts") {
      setTimeout(async () => {
        await this.loadScene(10);
        this.exposeTestReport(await this.runCornellArtifactAudit({ preservePose: true }));
      }, 200);
    } else if (automaticTest === "reference") {
      setTimeout(async () => {
        await this.loadScene(1);
        const report = {
          timestamp: new Date().toISOString(),
          adapter: this.adapterInfo(),
          scene: SCENE_INFO[1].name,
          reference: await this.runPathTracedReferenceAudit(),
          viewDistanceInvariance: await this.runViewDistanceInvarianceAudit(),
          repeatability: await this.runFinalFrameRepeatabilityAudit(),
          continuousMotion: await this.runContinuousMotionAudit({ frames: 32 }),
          cacheMotionRecovery: await this.runLongTranslationCacheAudit({
            motionFrames: 54,
            captureInterval: 6,
          }),
          movingLightResponse: await this.runMovingLightResponseAudit(),
          metrics: this.metricsSnapshot(),
        };
        report.passed = report.reference.passed
          && report.repeatability.passed
          && report.viewDistanceInvariance.passed
          && report.continuousMotion.passed
          && report.cacheMotionRecovery.passed
          && report.movingLightResponse.passed;
        this.exposeTestReport(report);
      }, 200);
    } else if (automaticTest === "view-invariance") {
      setTimeout(async () => {
        await this.loadScene(1);
        const report = await this.runViewDistanceInvarianceAudit();
        this.exposeTestReport(report);
      }, 200);
    } else if (automaticTest === "door-zoom") {
      setTimeout(async () => {
        await this.loadScene(8);
        this.exposeTestReport(await this.runDoorZoomContinuityAudit({ preservePose: true }));
      }, 200);
    } else if (automaticTest === "dynamic-sponza") {
      setTimeout(async () => {
        await this.loadScene(1);
        this.testTimeOverride = 2.0;
        const report = {
          scene: 1,
          motion: await this.runContinuousMotionAudit({
            frames: 48,
            warmup: 72,
            startTime: 0.8,
            timeStep: 1 / 60,
            movingLights: true,
          }),
          roundTrip: await this.runDynamicRoundTripAudit(),
          emitterResponse: await this.runDynamicEmitterResponseAudit(),
          shadowAgreement: await this.runShadowMapAudit(),
          reference: await this.runPathTracedReferenceAudit({
            warmup: 96,
            samples: 128,
          }),
          metrics: this.metricsSnapshot(),
        };
        report.passed = report.motion.passed
          && report.roundTrip.passed
          && report.emitterResponse.passed
          && report.shadowAgreement.passed
          && report.reference.passed
          && report.metrics.dynamicInstances >= 48
          && report.metrics.dynamicEmissiveInstances >= 6
          && report.metrics.gpuMs <= 16.67
          && report.metrics.dynamicUpdateP95Ms <= 1.0
          && report.metrics.dynamicUploadBytes < 65536
          && !report.metrics.overflows
          && !report.metrics.gpuError;
        this.exposeTestReport(report);
      }, 200);
    } else if (automaticTest?.startsWith("probe-")) {
      setTimeout(async () => {
        const requested = Number(automaticTest.slice(6));
        const index = clamp(Number.isFinite(requested) ? Math.floor(requested) : 0, 0, SCENE_INFO.length - 1);
        const requestedWarmup = Number(new URLSearchParams(location.search).get("warmup"));
        await this.loadScene(index);
        const report = {
          scene: index,
          motionStability: await this.runMotionStabilityAudit({
            samples: 5,
            interval: 3,
            warmup: Number.isFinite(requestedWarmup) && requestedWarmup > 0
              ? requestedWarmup
              : 48,
          }),
          metrics: this.metricsSnapshot(),
        };
        report.passed = report.motionStability.passed;
        this.exposeTestReport(report);
      }, 200);
    } else if (automaticTest?.startsWith("cache-motion-")) {
      setTimeout(async () => {
        const requested = Number(automaticTest.slice(13));
        const index = clamp(Number.isFinite(requested) ? Math.floor(requested) : 1, 0, SCENE_INFO.length - 1);
        await this.loadScene(index);
        const parameters = new URLSearchParams(location.search);
        const numericParameter = (name, fallback) => {
          const value = Number(parameters.get(name));
          return Number.isFinite(value) && value > 0 ? value : fallback;
        };
        const report = await this.runLongTranslationCacheAudit({
          warmup: numericParameter("warmup", 48),
          motionFrames: numericParameter("motionFrames", 180),
          captureInterval: numericParameter("captureInterval", 6),
          translationPerFrame: numericParameter("translationPerFrame", 0.12),
          recoveryWarmup: numericParameter("recoveryWarmup", 48),
          preservePose: parameters.get("show") === "1",
        });
        this.exposeTestReport(report);
      }, 200);
    } else if (automaticTest?.startsWith("floor-loop-")) {
      setTimeout(async () => {
        const requested = Number(automaticTest.slice(11));
        const index = clamp(Number.isFinite(requested) ? Math.floor(requested) : 1, 0, SCENE_INFO.length - 1);
        await this.loadScene(index);
        const parameters = new URLSearchParams(location.search);
        const numericParameter = (name, fallback) => {
          const value = Number(parameters.get(name));
          return Number.isFinite(value) && value > 0 ? value : fallback;
        };
        const report = await this.runFloorRoundTripAudit({
          warmup: numericParameter("warmup", 48),
          steps: numericParameter("steps", 8),
          translationPerFrame: numericParameter("translationPerFrame", 0.06),
          recoveryWarmup: numericParameter("recoveryWarmup", 48),
          movingLights: parameters.get("movingLights") === "1",
          debugMode: Math.max(0, Math.min(7, Math.floor(numericParameter("mode", 1)))),
          preservePose: parameters.get("show") === "1",
        });
        this.exposeTestReport(report);
      }, 200);
    } else if (automaticTest?.startsWith("motion-")) {
      setTimeout(async () => {
        const requested = Number(automaticTest.slice(7));
        const index = clamp(Number.isFinite(requested) ? Math.floor(requested) : 0, 0, SCENE_INFO.length - 1);
        const requestedWarmup = Number(new URLSearchParams(location.search).get("warmup"));
        const motionWarmup = Number.isFinite(requestedWarmup) && requestedWarmup > 0
          ? Math.floor(requestedWarmup)
          : 64;
        await this.loadScene(index);
        const report = {
          scene: index,
          baseline: await this.runContinuousMotionAudit({ frames: 32, warmup: motionWarmup }),
          movingLights: await this.runContinuousMotionAudit({
            frames: 32,
            warmup: motionWarmup,
            movingLights: true,
            timeStep: 1 / 60,
          }),
          metrics: this.metricsSnapshot(),
        };
        report.passed = report.baseline.passed
          && report.movingLights.passed
          && !report.metrics.overflows && !report.metrics.gpuError;
        this.exposeTestReport(report);
      }, 200);
    } else if (automaticTest?.startsWith("path-")) {
      setTimeout(async () => {
        const requested = Number(automaticTest.slice(5));
        const index = clamp(Number.isFinite(requested) ? Math.floor(requested) : 0, 0, SCENE_INFO.length - 1);
        const requestedWarmup = Number(new URLSearchParams(location.search).get("warmup"));
        const requestedSamples = Number(new URLSearchParams(location.search).get("samples"));
        const pathWarmup = Number.isFinite(requestedWarmup) && requestedWarmup > 0
          ? Math.floor(requestedWarmup)
          : 96;
        await this.loadScene(index);
        const report = {
          scene: index,
          reference: await this.runPathTracedReferenceAudit({
            warmup: pathWarmup,
            samples: Number.isFinite(requestedSamples) && requestedSamples > 0
              ? Math.floor(requestedSamples)
              : 512,
          }),
          metrics: this.metricsSnapshot(),
        };
        report.passed = report.reference.passed
          && !report.metrics.overflows && !report.metrics.gpuError;
        this.exposeTestReport(report);
      }, 200);
    } else if (automaticTest?.startsWith("response-")) {
      setTimeout(async () => {
        const requested = Number(automaticTest.slice(9));
        const index = clamp(Number.isFinite(requested) ? Math.floor(requested) : 1, 0, SCENE_INFO.length - 1);
        await this.loadScene(index);
        const report = await this.runMovingLightResponseAudit();
        this.exposeTestReport(report);
      }, 200);
    } else if (automaticTest?.startsWith("shadow-sweep-")) {
      setTimeout(async () => {
        const requested = Number(automaticTest.slice(13));
        const index = clamp(Number.isFinite(requested) ? Math.floor(requested) : 1, 0, SCENE_INFO.length - 1);
        await this.loadScene(index);
        const report = await this.runSunShadowSweepAudit();
        this.exposeTestReport(report);
      }, 200);
    } else if (automaticTest?.startsWith("shadow-")) {
      setTimeout(async () => {
        const requested = Number(automaticTest.slice(7));
        const index = clamp(Number.isFinite(requested) ? Math.floor(requested) : 0, 0, SCENE_INFO.length - 1);
        await this.loadScene(index);
        const report = await this.runShadowMapAudit();
        this.exposeTestReport(report);
      }, 200);
    } else if (automaticTest?.startsWith("scene-")) {
      setTimeout(async () => {
        const requested = Number(automaticTest.slice(6));
        const index = clamp(Number.isFinite(requested) ? Math.floor(requested) : 0, 0, SCENE_INFO.length - 1);
        await this.loadScene(index);
        await this.waitFrames(36);
        const result = this.metricsSnapshot();
        result.motionStability = await this.runMotionStabilityAudit({
          samples: 5,
          interval: 3,
          warmup: 384,
        });
        result.finalFrameRepeatability = await this.runFinalFrameRepeatabilityAudit({ poses: 5, warmup: 64 });
        result.continuousMotion = await this.runContinuousMotionAudit({
          frames: 32,
          warmup: 192,
        });
        result.movingLightContinuousMotion = await this.runContinuousMotionAudit({
          frames: 32,
          warmup: 192,
          movingLights: true,
          timeStep: 1 / 60,
        });
        if (index === 1 || index === 10) {
          result.movingLightResponse = await this.runMovingLightResponseAudit();
        }
        if (index === 1) {
          result.cacheMotionRecovery = await this.runLongTranslationCacheAudit({
            motionFrames: 54,
            captureInterval: 6,
          });
        }
        if ([0, 1, 10, 11, 12].includes(index)) {
          result.pathTracedReference = await this.runPathTracedReferenceAudit({
            warmup: 192,
          });
        }
        if (index === 0) {
          result.enclosureLeak = await this.runEnclosureLeakAudit();
        }
        if (index === 10) {
          result.cornellArtifacts = await this.runCornellArtifactAudit();
        }
        result.passed = !result.overflows
          && !result.gpuError
          && result.motionStability.passed
          && result.finalFrameRepeatability.passed
          && result.continuousMotion.passed
          && result.movingLightContinuousMotion.passed
          && (!result.movingLightResponse || result.movingLightResponse.passed)
          && (!result.cacheMotionRecovery || result.cacheMotionRecovery.passed)
          && (!result.pathTracedReference || result.pathTracedReference.passed)
          && (!result.enclosureLeak || result.enclosureLeak.passed)
          && (!result.cornellArtifacts || result.cornellArtifacts.passed);
        this.exposeTestReport(result);
      }, 200);
    } else if (automaticTest != null) {
      // Sixty scene frames guarantee at least one 45-frame timestamp-query
      // cycle after each scene resets its metric history.
      setTimeout(() => this.runValidation({ framesPerScene: 60 }), 200);
    } else if (new URLSearchParams(location.search).has("scene")) {
      setTimeout(async () => {
        const requested = Number(new URLSearchParams(location.search).get("scene"));
        const index = clamp(
          Number.isFinite(requested) ? Math.floor(requested) : 1,
          0,
          SCENE_INFO.length - 1,
        );
        await this.loadScene(index);
        const sceneParams = new URLSearchParams(location.search);
        const requestedTime = sceneParams.has("time")
          ? Number(sceneParams.get("time"))
          : NaN;
        if (Number.isFinite(requestedTime)) this.testTimeOverride = requestedTime;
        const requestedMode = sceneParams.has("mode")
          ? Number(sceneParams.get("mode"))
          : NaN;
        if (Number.isFinite(requestedMode)) {
          this.debugMode = clamp(Math.floor(requestedMode), 0, 6);
          const debugView = document.getElementById("debug-view");
          if (debugView) debugView.value = String(this.debugMode);
        }
      }, 200);
    } else if (new URLSearchParams(location.search).get("pose") === "inside-box") {
      setTimeout(async () => {
        await this.loadScene(0);
        this.animateCamera = false;
        this.animateLights = false;
        this.setCameraPose([-2.2, 1.25, 0.5], [-1.3, 1.72, -0.42]);
        this.resetProbeHistory();
      }, 200);
    } else if (new URLSearchParams(location.search).get("pose") === "visibility-lab") {
      setTimeout(async () => {
        await this.loadScene(12);
        this.animateCamera = false;
        this.animateLights = false;
        this.temporalStability = true;
        this.debugMode = 1;
        this.testTimeOverride = 0.7;
        this.setCameraPose([7.8, 5.0, 10.5], [0, 1.4, -1.8]);
        this.resetProbeHistory();
        if ($("view-mode")) $("view-mode").value = "1";
        if ($("animate-camera")) $("animate-camera").checked = false;
        if ($("animate-lights")) $("animate-lights").checked = false;
      }, 200);
    } else if (CORNELL_ARTIFACT_POSES[new URLSearchParams(location.search).get("pose")]) {
      setTimeout(async () => {
        const pose = CORNELL_ARTIFACT_POSES[new URLSearchParams(location.search).get("pose")];
        await this.loadScene(10);
        this.animateCamera = false;
        this.animateLights = false;
        this.temporalStability = true;
        this.debugMode = 1;
        this.testTimeOverride = 0.7;
        this.setCameraPose(pose.position, pose.target);
        this.resetProbeHistory();
        if ($("view-mode")) $("view-mode").value = "1";
        if ($("animate-camera")) $("animate-camera").checked = false;
        if ($("animate-lights")) $("animate-lights").checked = false;
      }, 200);
    }
    return true;
  }

  adapterInfo() {
    const info = this.adapter?.info || {};
    return {
      vendor: info.vendor || "unknown",
      architecture: info.architecture || "unknown",
      device: info.device || "unknown",
      description: info.description || "High-performance WebGPU adapter",
      timestampQuery: !!this.timestampSupported,
      limits: {
        maxStorageBufferBindingSize: this.device?.limits.maxStorageBufferBindingSize,
        maxComputeInvocationsPerWorkgroup: this.device?.limits.maxComputeInvocationsPerWorkgroup,
      },
    };
  }

  updateAdapterLabel() {
    const info = this.adapterInfo();
    const label = info.description !== "High-performance WebGPU adapter"
      ? info.description
      : [info.vendor, info.architecture].filter((v) => v !== "unknown").join(" ") || "WebGPU adapter";
    if ($("gpu-name")) $("gpu-name").textContent = `${label}${this.timestampSupported ? " · GPU timestamps" : ""}`;
  }

  async createPipelines() {
    const device = this.device;
    device.pushErrorScope("validation");
    const rasterModule = device.createShaderModule({ label: "Split RC raster shader", code: rasterShader });
    const computeModule = device.createShaderModule({ label: "Split RC compute shader", code: computeShader });
    const finalModule = device.createShaderModule({ label: "Split RC composite shader", code: finalShader });
    const presentModule = device.createShaderModule({ label: "Split RC direct presentation shader", code: presentShader });

    this.frameLayout = device.createBindGroupLayout({
      label: "frame-uniform-layout",
      entries: [
        { binding: 0, visibility: SHADER.VERTEX | SHADER.FRAGMENT, buffer: { type: "uniform" } },
        {
          binding: 1,
          visibility: SHADER.FRAGMENT,
          texture: { sampleType: "float", viewDimension: "2d-array" },
        },
        { binding: 2, visibility: SHADER.FRAGMENT, sampler: { type: "filtering" } },
        { binding: 3, visibility: SHADER.VERTEX, buffer: { type: "uniform" } },
      ],
    });
    this.rasterLayout = device.createPipelineLayout({ bindGroupLayouts: [this.frameLayout] });
    this.dynamicFrameLayout = device.createBindGroupLayout({
      label: "dynamic-frame-uniform-layout",
      entries: [
        { binding: 0, visibility: SHADER.VERTEX | SHADER.FRAGMENT, buffer: { type: "uniform" } },
        { binding: 1, visibility: SHADER.FRAGMENT, texture: { sampleType: "float", viewDimension: "2d-array" } },
        { binding: 2, visibility: SHADER.FRAGMENT, sampler: { type: "filtering" } },
        { binding: 3, visibility: SHADER.VERTEX, buffer: { type: "uniform" } },
        { binding: 4, visibility: SHADER.VERTEX, buffer: { type: "read-only-storage" } },
      ],
    });
    this.dynamicRasterLayout = device.createPipelineLayout({
      bindGroupLayouts: [this.dynamicFrameLayout],
    });
    const vertexBuffers = [{
      arrayStride: 64,
      attributes: [
        { shaderLocation: 0, offset: 0, format: "float32x3" },
        { shaderLocation: 1, offset: 12, format: "float32x3" },
        { shaderLocation: 2, offset: 24, format: "float32x3" },
        { shaderLocation: 3, offset: 36, format: "float32x3" },
        { shaderLocation: 4, offset: 48, format: "float32x2" },
        { shaderLocation: 5, offset: 56, format: "float32x2" },
      ],
    }];
    this.gbufferPipeline = device.createRenderPipeline({
      label: "G-buffer pipeline",
      layout: this.rasterLayout,
      vertex: { module: rasterModule, entryPoint: "gbufferVS", buffers: vertexBuffers },
      fragment: {
        module: rasterModule,
        entryPoint: "gbufferFS",
        targets: [
          { format: "rgba8unorm" },
          { format: "rgba16float" },
          // Probe keys and C(-1) visibility are world-space algorithms. Half
          // precision here can move a raster sample across a thin wall or a
          // probe-cell boundary as the camera moves, so positions stay f32.
          { format: "rgba32float" },
        ],
      },
      primitive: { topology: "triangle-list", cullMode: "none" },
      depthStencil: { format: "depth24plus", depthWriteEnabled: true, depthCompare: "less" },
    });
    this.dynamicGbufferPipeline = device.createRenderPipeline({
      label: "Dynamic instance G-buffer pipeline",
      layout: this.dynamicRasterLayout,
      vertex: { module: rasterModule, entryPoint: "dynamicGbufferVS", buffers: vertexBuffers },
      fragment: {
        module: rasterModule,
        entryPoint: "gbufferFS",
        targets: [
          { format: "rgba8unorm" },
          { format: "rgba16float" },
          { format: "rgba32float" },
        ],
      },
      primitive: { topology: "triangle-list", cullMode: "none" },
      depthStencil: { format: "depth24plus", depthWriteEnabled: true, depthCompare: "less" },
    });
    this.shadowPipeline = device.createRenderPipeline({
      label: "Sun shadow pipeline",
      layout: this.rasterLayout,
      vertex: { module: rasterModule, entryPoint: "shadowVS", buffers: vertexBuffers },
      fragment: { module: rasterModule, entryPoint: "shadowFS", targets: [] },
      primitive: { topology: "triangle-list", cullMode: "none" },
      depthStencil: { format: "depth32float", depthWriteEnabled: true, depthCompare: "less", depthBias: 2, depthBiasSlopeScale: 1.5 },
    });
    this.dynamicShadowPipeline = device.createRenderPipeline({
      label: "Dynamic instance sun shadow pipeline",
      layout: this.dynamicRasterLayout,
      vertex: { module: rasterModule, entryPoint: "dynamicShadowVS", buffers: vertexBuffers },
      fragment: { module: rasterModule, entryPoint: "shadowFS", targets: [] },
      primitive: { topology: "triangle-list", cullMode: "none" },
      depthStencil: { format: "depth32float", depthWriteEnabled: true, depthCompare: "less", depthBias: 2, depthBiasSlopeScale: 1.5 },
    });
    this.pointShadowPipeline = device.createRenderPipeline({
      label: "Point-light cube shadow pipeline",
      layout: this.rasterLayout,
      vertex: { module: rasterModule, entryPoint: "pointShadowVS", buffers: vertexBuffers },
      fragment: { module: rasterModule, entryPoint: "shadowFS", targets: [] },
      primitive: { topology: "triangle-list", cullMode: "none" },
      depthStencil: {
        format: "depth32float",
        depthWriteEnabled: true,
        depthCompare: "less",
        depthBias: 2,
        depthBiasSlopeScale: 1.5,
      },
    });
    this.dynamicPointShadowPipeline = device.createRenderPipeline({
      label: "Dynamic instance point shadow pipeline",
      layout: this.dynamicRasterLayout,
      vertex: { module: rasterModule, entryPoint: "dynamicPointShadowVS", buffers: vertexBuffers },
      fragment: { module: rasterModule, entryPoint: "shadowFS", targets: [] },
      primitive: { topology: "triangle-list", cullMode: "none" },
      depthStencil: {
        format: "depth32float",
        depthWriteEnabled: true,
        depthCompare: "less",
        depthBias: 2,
        depthBiasSlopeScale: 1.5,
      },
    });

    this.computeLayout = device.createBindGroupLayout({
      label: "Split RC compute layout",
      entries: [
        { binding: 0, visibility: SHADER.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: SHADER.COMPUTE, texture: { sampleType: "unfilterable-float" } },
        { binding: 2, visibility: SHADER.COMPUTE, texture: { sampleType: "unfilterable-float" } },
        { binding: 3, visibility: SHADER.COMPUTE, buffer: { type: "storage" } },
        { binding: 4, visibility: SHADER.COMPUTE, buffer: { type: "storage" } },
        { binding: 5, visibility: SHADER.COMPUTE, buffer: { type: "storage" } },
        { binding: 6, visibility: SHADER.COMPUTE, buffer: { type: "storage" } },
        { binding: 7, visibility: SHADER.COMPUTE, buffer: { type: "storage" } },
        { binding: 8, visibility: SHADER.COMPUTE, buffer: { type: "storage" } },
        { binding: 9, visibility: SHADER.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 10, visibility: SHADER.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 11, visibility: SHADER.COMPUTE, buffer: { type: "uniform" } },
        {
          binding: 12,
          visibility: SHADER.COMPUTE,
          texture: { sampleType: "float", viewDimension: "2d-array" },
        },
        { binding: 13, visibility: SHADER.COMPUTE, sampler: { type: "filtering" } },
        {
          binding: 14,
          visibility: SHADER.COMPUTE,
          storageTexture: { access: "write-only", format: "rgba16float", viewDimension: "2d" },
        },
        { binding: 15, visibility: SHADER.COMPUTE, texture: { sampleType: "float" } },
        { binding: 16, visibility: SHADER.COMPUTE, sampler: { type: "filtering" } },
        {
          binding: 17,
          visibility: SHADER.COMPUTE,
          texture: { sampleType: "depth", viewDimension: "2d-array" },
        },
        { binding: 18, visibility: SHADER.COMPUTE, sampler: { type: "comparison" } },
        {
          binding: 19,
          visibility: SHADER.COMPUTE,
          texture: { sampleType: "depth", viewDimension: "2d-array" },
        },
        { binding: 20, visibility: SHADER.COMPUTE, sampler: { type: "comparison" } },
        { binding: 21, visibility: SHADER.COMPUTE, buffer: { type: "uniform" } },
      ],
    });
    this.computePipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [this.computeLayout] });
    const cp = (entryPoint) => device.createComputePipeline({
      label: `Split RC ${entryPoint}`,
      layout: this.computePipelineLayout,
      compute: { module: computeModule, entryPoint },
    });
    this.computePipelines = {
      reset: cp("resetSlots"),
      initBase: cp("initBase"),
      initHigher: cp("initHigher"),
      canonicalize: cp("canonicalizeProbes"),
      countBase: cp("countBaseRays"),
      countHigher: cp("countHigherRays"),
      assignOffsets: cp("assignRayOffsets"),
      mapPrimary: cp("mapPrimaryRaySamples"),
      prefixRayBlocks: cp("prefixRayBlocks"),
      classifyEnvironmentAccess: cp("classifyEnvironmentAccess"),
      splitRays: cp("splitRays"),
      merge: cp("mergeCascade"),
      resolveTangentSupportSources: cp("resolveTangentSupportSources"),
      resolveTangentSupportIrradiance: cp("resolveTangentSupportIrradiance"),
      prefilter: cp("prefilterIrradiance"),
      validateReference: cp("validateReference"),
      validateShadowMaps: cp("validateShadowMaps"),
    };

    this.finalLayout = device.createBindGroupLayout({
      label: "Split RC composite layout",
      entries: [
        { binding: 0, visibility: SHADER.FRAGMENT, buffer: { type: "uniform" } },
        { binding: 1, visibility: SHADER.FRAGMENT, texture: { sampleType: "float" } },
        { binding: 2, visibility: SHADER.FRAGMENT, texture: { sampleType: "unfilterable-float" } },
        { binding: 3, visibility: SHADER.FRAGMENT, texture: { sampleType: "unfilterable-float" } },
        {
          binding: 4,
          visibility: SHADER.FRAGMENT,
          texture: { sampleType: "depth", viewDimension: "2d-array" },
        },
        { binding: 5, visibility: SHADER.FRAGMENT, sampler: { type: "comparison" } },
        { binding: 6, visibility: SHADER.FRAGMENT, buffer: { type: "storage" } },
        { binding: 7, visibility: SHADER.FRAGMENT, texture: { sampleType: "float" } },
        { binding: 8, visibility: SHADER.FRAGMENT, buffer: { type: "read-only-storage" } },
        { binding: 9, visibility: SHADER.FRAGMENT, buffer: { type: "storage" } },
        { binding: 10, visibility: SHADER.FRAGMENT, sampler: { type: "filtering" } },
        { binding: 11, visibility: SHADER.FRAGMENT, texture: { sampleType: "unfilterable-float" } },
        {
          binding: 12,
          visibility: SHADER.FRAGMENT,
          texture: { sampleType: "depth", viewDimension: "2d-array" },
        },
        { binding: 13, visibility: SHADER.FRAGMENT, sampler: { type: "comparison" } },
        { binding: 14, visibility: SHADER.FRAGMENT, buffer: { type: "uniform" } },
        { binding: 15, visibility: SHADER.FRAGMENT, buffer: { type: "read-only-storage" } },
        { binding: 16, visibility: SHADER.FRAGMENT, buffer: { type: "read-only-storage" } },
        {
          binding: 17,
          visibility: SHADER.FRAGMENT,
          texture: { sampleType: "float", viewDimension: "2d-array" },
        },
        { binding: 18, visibility: SHADER.FRAGMENT, sampler: { type: "filtering" } },
        { binding: 19, visibility: SHADER.FRAGMENT, buffer: { type: "read-only-storage" } },
        { binding: 20, visibility: SHADER.FRAGMENT, buffer: { type: "read-only-storage" } },
      ],
    });
    this.finalPipeline = device.createRenderPipeline({
      label: "Split RC final composite",
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.finalLayout] }),
      vertex: { module: finalModule, entryPoint: "fullscreenVS" },
      fragment: { module: finalModule, entryPoint: "finalFS", targets: [{ format: this.format }] },
      primitive: { topology: "triangle-list" },
    });
    this.presentLayout = device.createBindGroupLayout({
      label: "current composite presentation layout",
      entries: [
        { binding: 0, visibility: SHADER.FRAGMENT, texture: { sampleType: "float" } },
        { binding: 1, visibility: SHADER.FRAGMENT, sampler: { type: "filtering" } },
      ],
    });
    this.presentPipeline = device.createRenderPipeline({
      label: "current composite presentation",
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.presentLayout] }),
      vertex: { module: presentModule, entryPoint: "presentVS" },
      fragment: { module: presentModule, entryPoint: "presentFS", targets: [{ format: this.format }] },
      primitive: { topology: "triangle-list" },
    });
    const error = await device.popErrorScope();
    if (error) throw new Error(`WebGPU pipeline validation failed: ${error.message}`);
  }

  async createMaterialAtlas() {
    const response = await fetch("/models/sponza-atlas.webp?v=2026-08-01-dynamic-tlas10");
    if (!response.ok) throw new Error(`Sponza material atlas request failed (${response.status}).`);
    const bitmap = await createImageBitmap(await response.blob(), { colorSpaceConversion: "default" });
    const layerSize = 811;
    const layerStride = 819;
    const layerBorder = 4;
    const layerCount = 25;
    const mipLevelCount = Math.floor(Math.log2(layerSize)) + 1;
    this.materialAtlas = this.device.createTexture({
      label: "mipmapped Sponza base-color texture array",
      size: [layerSize, layerSize, layerCount],
      mipLevelCount,
      format: "rgba8unorm-srgb",
      usage: TEX.TEXTURE_BINDING | TEX.COPY_DST | TEX.RENDER_ATTACHMENT,
    });
    for (let layer = 0; layer < layerCount; layer++) {
      this.device.queue.copyExternalImageToTexture(
        {
          source: bitmap,
          origin: [
            (layer % 5) * layerStride + layerBorder,
            Math.floor(layer / 5) * layerStride + layerBorder,
          ],
        },
        { texture: this.materialAtlas, origin: [0, 0, layer] },
        [layerSize, layerSize, 1],
      );
    }
    bitmap.close();

    const mipModule = this.device.createShaderModule({
      label: "material-array mip generator",
      code: /* wgsl */`
        @group(0) @binding(0) var source: texture_2d<f32>;
        @group(0) @binding(1) var sourceSampler: sampler;
        @vertex fn vs(@builtin(vertex_index) vertexIndex:u32)->@builtin(position) vec4f {
          let p=array<vec2f,3>(vec2f(-1,-1),vec2f(3,-1),vec2f(-1,3));
          return vec4f(p[vertexIndex],0,1);
        }
        @fragment fn fs(@builtin(position) position:vec4f)->@location(0) vec4f {
          let sourceSize=textureDimensions(source);
          let destinationSize=max(vec2u(1),sourceSize/2u);
          let uv=position.xy/vec2f(destinationSize);
          let color=textureSampleLevel(source,sourceSampler,uv,0.0);
          let center=uv*vec2f(sourceSize)-vec2f(0.5);
          let base=vec2i(floor(center));
          let maximum=vec2i(sourceSize)-vec2i(1);
          let a=textureLoad(source,clamp(base,vec2i(0),maximum),0).a;
          let b=textureLoad(source,clamp(base+vec2i(1,0),vec2i(0),maximum),0).a;
          let c=textureLoad(source,clamp(base+vec2i(0,1),vec2i(0),maximum),0).a;
          let d=textureLoad(source,clamp(base+vec2i(1),vec2i(0),maximum),0).a;
          return vec4f(color.rgb,max(max(a,b),max(c,d)));
        }
      `,
    });
    const mipPipeline = this.device.createRenderPipeline({
      label: "material-array mip generator",
      layout: "auto",
      vertex: { module: mipModule, entryPoint: "vs" },
      fragment: {
        module: mipModule,
        entryPoint: "fs",
        targets: [{ format: "rgba8unorm-srgb" }],
      },
      primitive: { topology: "triangle-list" },
    });
    const mipSampler = this.device.createSampler({
      minFilter: "linear",
      magFilter: "linear",
    });
    const encoder = this.device.createCommandEncoder({ label: "generate material-array mips" });
    for (let mip = 1; mip < mipLevelCount; mip++) {
      for (let layer = 0; layer < layerCount; layer++) {
        const sourceView = this.materialAtlas.createView({
          dimension: "2d",
          baseMipLevel: mip - 1,
          mipLevelCount: 1,
          baseArrayLayer: layer,
          arrayLayerCount: 1,
        });
        const destinationView = this.materialAtlas.createView({
          dimension: "2d",
          baseMipLevel: mip,
          mipLevelCount: 1,
          baseArrayLayer: layer,
          arrayLayerCount: 1,
        });
        const pass = encoder.beginRenderPass({
          label: `material mip ${mip} layer ${layer}`,
          colorAttachments: [{
            view: destinationView,
            clearValue: [0, 0, 0, 0],
            loadOp: "clear",
            storeOp: "store",
          }],
        });
        pass.setPipeline(mipPipeline);
        pass.setBindGroup(0, this.device.createBindGroup({
          layout: mipPipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: sourceView },
            { binding: 1, resource: mipSampler },
          ],
        }));
        pass.draw(3);
        pass.end();
      }
    }
    this.device.queue.submit([encoder.finish()]);
    this.materialAtlasView = this.materialAtlas.createView({ dimension: "2d-array" });
    this.materialSampler = this.device.createSampler({
      minFilter: "linear",
      magFilter: "linear",
      mipmapFilter: "linear",
      maxAnisotropy: 16,
      addressModeU: "clamp-to-edge",
      addressModeV: "clamp-to-edge",
    });
  }

  createPersistentResources() {
    const d = this.device;
    this.frameBuffer = createBuffer(d, "frame uniforms", 288, GPU.UNIFORM | GPU.COPY_DST);
    this.hashBuffer = createBuffer(d, "world-key probe hash history", K.totalHashSlots * K.hashFrames * 8, GPU.STORAGE | GPU.COPY_SRC | GPU.COPY_DST);
    this.stateBuffer = createBuffer(d, "probe counters, ray prefixes, and diagnostics", K.stateWords * 4, GPU.STORAGE | GPU.COPY_SRC | GPU.COPY_DST);
    this.probeMetaBuffer = createBuffer(d, "sparse probe metadata", K.totalProbeMeta * 16, GPU.STORAGE | GPU.COPY_DST);
    this.accumBuffer = createBuffer(
      d,
      "double-buffered fixed-point ray intervals",
      K.totalDirectionData * 5 * K.accumFrames * 4,
      GPU.STORAGE | GPU.COPY_DST,
    );
    this.coneBuffer = createBuffer(d, "merged radiance cones", K.totalDirectionData * 16, GPU.STORAGE | GPU.COPY_DST);
    this.irradianceBuffer = createBuffer(d, "world-key bordered 6x6 probe irradiance history", K.probeCaps[0] * K.irradianceTexels * K.irradianceFrames * 16, GPU.STORAGE | GPU.COPY_SRC | GPU.COPY_DST);
    // WebGPU forbids writable storage and sampled usage for one texture in a
    // single synchronization scope. Keep the compute target separate, then
    // copy the completed frame half into the filterable atlas.
    this.irradianceAtlasWrite = d.createTexture({
      label: "world-key 8x8 probe irradiance storage atlas history",
      size: [K.irradianceAtlasWidth, K.irradianceAtlasFrameHeight * K.irradianceFrames],
      format: "rgba16float",
      usage: TEX.STORAGE_BINDING | TEX.COPY_SRC,
    });
    this.irradianceAtlas = d.createTexture({
      label: "world-key filterable 8x8 probe irradiance atlas history",
      size: [K.irradianceAtlasWidth, K.irradianceAtlasFrameHeight * K.irradianceFrames],
      format: "rgba16float",
      usage: TEX.TEXTURE_BINDING | TEX.COPY_DST,
    });
    this.irradianceSampler = d.createSampler({
      minFilter: "linear",
      magFilter: "linear",
      addressModeU: "clamp-to-edge",
      addressModeV: "clamp-to-edge",
    });
    this.presentSampler = d.createSampler({
      minFilter: "linear",
      magFilter: "linear",
      addressModeU: "clamp-to-edge",
      addressModeV: "clamp-to-edge",
    });
    this.passBuffers = [0, 1, 2, 3].map((i) => makePassBuffer(d, i));
    this.sunShadowBuffers = Array.from({ length: SUN_CASCADE_COUNT }, (_, i) =>
      createBuffer(d, `sun shadow cascade ${i}`, 64, GPU.UNIFORM | GPU.COPY_DST));
    this.sunShadowDataBuffer = createBuffer(
      d,
      "camera-fitted stabilized sun shadow cascades",
      320,
      GPU.UNIFORM | GPU.COPY_DST,
    );
    this.pointShadowBuffers = [0, 1, 2, 3, 4, 5].map((i) =>
      createBuffer(d, `point shadow face ${i}`, 64, GPU.UNIFORM | GPU.COPY_DST));
    this.shadowSampler = d.createSampler({ compare: "less-equal", minFilter: "linear", magFilter: "linear" });
    this.pointShadowSampler = d.createSampler({
      compare: "less-equal",
      minFilter: "linear",
      magFilter: "linear",
    });
    this.rasterBindGroup = d.createBindGroup({
      layout: this.frameLayout,
      entries: [
        { binding: 0, resource: { buffer: this.frameBuffer } },
        { binding: 1, resource: this.materialAtlasView },
        { binding: 2, resource: this.materialSampler },
        { binding: 3, resource: { buffer: this.pointShadowBuffers[0] } },
      ],
    });
    this.sunShadowBindGroups = this.sunShadowBuffers.map((buffer, i) => d.createBindGroup({
      label: `sun shadow cascade ${i} bind group`,
      layout: this.frameLayout,
      entries: [
        { binding: 0, resource: { buffer: this.frameBuffer } },
        { binding: 1, resource: this.materialAtlasView },
        { binding: 2, resource: this.materialSampler },
        { binding: 3, resource: { buffer } },
      ],
    }));
    this.pointShadowBindGroups = this.pointShadowBuffers.map((buffer, i) => d.createBindGroup({
      label: `point shadow face bind group ${i}`,
      layout: this.frameLayout,
      entries: [
        { binding: 0, resource: { buffer: this.frameBuffer } },
        { binding: 1, resource: this.materialAtlasView },
        { binding: 2, resource: this.materialSampler },
        { binding: 3, resource: { buffer } },
      ],
    }));
    if (this.timestampSupported) {
      this.querySet = d.createQuerySet({ type: "timestamp", count: 8 });
      this.queryResolveBuffer = createBuffer(d, "timestamp resolve", 64, GPU.QUERY_RESOLVE | GPU.COPY_SRC);
    }
    this.resetProbeHistory();
  }

  resetProbeHistory() {
    if (!this.device || !this.hashBuffer || !this.irradianceBuffer) return;
    const emptyHash = new Uint32Array(K.totalHashSlots * K.hashFrames * 2);
    emptyHash.fill(0xffffffff);
    this.device.queue.writeBuffer(this.hashBuffer, 0, emptyHash);
    this.device.queue.writeBuffer(
      this.irradianceBuffer,
      0,
      new Uint8Array(K.probeCaps[0] * K.irradianceTexels * K.irradianceFrames * 16),
    );
    this.sampleFrameIndex = 0;
    this.sampleEpoch = ((this.sampleEpoch + 1) >>> 0) || 1;
    this.historyValid = false;
  }

  createSizedResources() {
    const quality = QUALITY[this.qualityName];
    const rect = this.canvas.getBoundingClientRect();
    const requestedWidth = Math.max(2, rect.width * Math.min(devicePixelRatio, 2) * quality.pixelRatio);
    const requestedHeight = Math.max(2, rect.height * Math.min(devicePixelRatio, 2) * quality.pixelRatio);
    const pixelBudgetScale = Math.min(
      1,
      Math.sqrt(
        quality.maxPixels * this.adaptivePixelBudgetScale
        / Math.max(1, requestedWidth * requestedHeight),
      ),
    );
    const width = Math.max(2, Math.floor(requestedWidth * pixelBudgetScale));
    const height = Math.max(2, Math.floor(requestedHeight * pixelBudgetScale));
    this.resolutionScale = pixelBudgetScale;
    if (this.width === width && this.height === height && this.gbuffer) return;
    this.width = width;
    this.height = height;
    this.giWidth = Math.max(1, Math.ceil(width / quality.giDivisor));
    this.giHeight = Math.max(1, Math.ceil(height / quality.giDivisor));
    this.raysPerSample = quality.raysPerSample;
    this.probeMetaBuffer?.destroy();
    this.stateBuffer?.destroy();
    const samplesPerFrame = this.giWidth * this.giHeight * this.raysPerSample;
    const deterministicRayVec4s = samplesPerFrame * 2;
    const stableSlots = samplesPerFrame * 2;
    const preferredStateBytes = Math.min(
      this.device.limits.maxStorageBufferBindingSize,
      64 * 1024 * 1024,
    );
    const maximumBlocksPerProbe = Math.max(
      1,
      Math.floor((preferredStateBytes / 4 - K.stateWords) / K.probeCaps[0]),
    );
    this.rayBlockSize = 256;
    while (Math.ceil(stableSlots / this.rayBlockSize) > maximumBlocksPerProbe) {
      this.rayBlockSize *= 2;
    }
    this.rayBlockCount = Math.ceil(stableSlots / this.rayBlockSize);
    this.stateBuffer = createBuffer(
      this.device,
      "probe counters, ray prefixes, diagnostics, and deterministic block prefixes",
      (K.stateWords + K.probeCaps[0] * this.rayBlockCount) * 4,
      GPU.STORAGE | GPU.COPY_SRC | GPU.COPY_DST,
    );
    this.probeMetaBuffer = createBuffer(
      this.device,
      "sparse probe metadata and deterministic primary ray ownership",
      (K.totalProbeMeta + deterministicRayVec4s) * 16,
      GPU.STORAGE | GPU.COPY_DST,
    );
    this.historyValid = false;
    this.canvas.width = width;
    this.canvas.height = height;
    for (const texture of this.gbuffer || []) texture.destroy();
    this.shadowTexture?.destroy();
    this.pointShadowTexture?.destroy();
    const d = this.device;
    const texture = (label, format, usage = TEX.RENDER_ATTACHMENT | TEX.TEXTURE_BINDING) =>
      d.createTexture({ label, size: [width, height], format, usage });
    this.albedoTexture = texture("G-buffer albedo", "rgba8unorm");
    this.normalTexture = texture(
      "G-buffer normal",
      "rgba16float",
      TEX.RENDER_ATTACHMENT | TEX.TEXTURE_BINDING | TEX.COPY_SRC,
    );
    this.worldTexture = texture(
      "G-buffer full-precision world position",
      "rgba32float",
      TEX.RENDER_ATTACHMENT | TEX.TEXTURE_BINDING | TEX.COPY_SRC,
    );
    this.depthTexture = texture("G-buffer depth", "depth24plus", TEX.RENDER_ATTACHMENT);
    const compositeUsage = TEX.RENDER_ATTACHMENT | TEX.TEXTURE_BINDING | TEX.COPY_SRC;
    this.compositeTexture = texture("current Split RC composite", this.format, compositeUsage);
    this.gbuffer = [
      this.albedoTexture,
      this.normalTexture,
      this.worldTexture,
      this.depthTexture,
      this.compositeTexture,
    ];
    this.shadowTexture = d.createTexture({
      label: "four camera-fitted stabilized sun shadow cascades",
      size: [quality.shadow, quality.shadow, SUN_CASCADE_COUNT],
      format: "depth32float",
      usage: TEX.RENDER_ATTACHMENT | TEX.TEXTURE_BINDING,
    });
    this.sunShadowArrayView = this.shadowTexture.createView({ dimension: "2d-array" });
    this.sunShadowCascadeViews = Array.from({ length: SUN_CASCADE_COUNT }, (_, cascade) =>
      this.shadowTexture.createView({
        dimension: "2d",
        baseArrayLayer: cascade,
        arrayLayerCount: 1,
      }));
    const pointShadowSize = Math.max(256, Math.floor(quality.shadow / 2));
    this.pointShadowTexture = d.createTexture({
      label: "moving point-light cube shadow map",
      size: [pointShadowSize, pointShadowSize, 6],
      format: "depth32float",
      usage: TEX.RENDER_ATTACHMENT | TEX.TEXTURE_BINDING,
    });
    this.pointShadowArrayView = this.pointShadowTexture.createView({ dimension: "2d-array" });
    this.pointShadowFaceViews = [0, 1, 2, 3, 4, 5].map((face) =>
      this.pointShadowTexture.createView({
        dimension: "2d",
        baseArrayLayer: face,
        arrayLayerCount: 1,
      }));
    this.rebuildBindGroups();
  }

  rebuildBindGroups() {
    if (!this.worldTexture || !this.bvhNodeBuffer) return;
    const dynamicEntries = (lightBuffer) => [
      { binding: 0, resource: { buffer: this.frameBuffer } },
      { binding: 1, resource: this.materialAtlasView },
      { binding: 2, resource: this.materialSampler },
      { binding: 3, resource: { buffer: lightBuffer } },
      { binding: 4, resource: { buffer: this.triangleBuffer } },
    ];
    this.dynamicRasterBindGroup = this.dynamicScene ? this.device.createBindGroup({
      label: "dynamic G-buffer instance arena",
      layout: this.dynamicFrameLayout,
      entries: dynamicEntries(this.sunShadowBuffers[0]),
    }) : null;
    this.dynamicSunShadowBindGroups = this.dynamicScene
      ? this.sunShadowBuffers.map((buffer, cascade) => this.device.createBindGroup({
        label: `dynamic sun shadow instance arena ${cascade}`,
        layout: this.dynamicFrameLayout,
        entries: dynamicEntries(buffer),
      }))
      : [];
    this.dynamicPointShadowBindGroups = this.dynamicScene
      ? this.pointShadowBuffers.map((buffer, face) => this.device.createBindGroup({
        label: `dynamic point shadow instance arena ${face}`,
        layout: this.dynamicFrameLayout,
        entries: dynamicEntries(buffer),
      }))
      : [];
    const commonEntries = (passBuffer) => [
      { binding: 0, resource: { buffer: this.frameBuffer } },
      { binding: 1, resource: this.worldTexture.createView() },
      { binding: 2, resource: this.normalTexture.createView() },
      { binding: 3, resource: { buffer: this.hashBuffer } },
      { binding: 4, resource: { buffer: this.stateBuffer } },
      { binding: 5, resource: { buffer: this.probeMetaBuffer } },
      { binding: 6, resource: { buffer: this.accumBuffer } },
      { binding: 7, resource: { buffer: this.coneBuffer } },
      { binding: 8, resource: { buffer: this.irradianceBuffer } },
      { binding: 9, resource: { buffer: this.bvhNodeBuffer } },
      { binding: 10, resource: { buffer: this.triangleBuffer } },
      { binding: 11, resource: { buffer: passBuffer } },
      { binding: 12, resource: this.materialAtlasView },
      { binding: 13, resource: this.materialSampler },
      { binding: 14, resource: this.irradianceAtlasWrite.createView() },
      { binding: 15, resource: this.irradianceAtlas.createView() },
      { binding: 16, resource: this.irradianceSampler },
      { binding: 17, resource: this.pointShadowArrayView },
      { binding: 18, resource: this.pointShadowSampler },
      { binding: 19, resource: this.sunShadowArrayView },
      { binding: 20, resource: this.shadowSampler },
      { binding: 21, resource: { buffer: this.sunShadowDataBuffer } },
    ];
    this.computeBindGroups = this.passBuffers.map((buffer, i) => this.device.createBindGroup({
      label: `compute bind group cascade ${i}`,
      layout: this.computeLayout,
      entries: commonEntries(buffer),
    }));
    this.finalBindGroup = this.device.createBindGroup({
      layout: this.finalLayout,
      entries: [
        { binding: 0, resource: { buffer: this.frameBuffer } },
        { binding: 1, resource: this.albedoTexture.createView() },
        { binding: 2, resource: this.normalTexture.createView() },
        { binding: 3, resource: this.worldTexture.createView() },
        { binding: 4, resource: this.sunShadowArrayView },
        { binding: 5, resource: this.shadowSampler },
        { binding: 6, resource: { buffer: this.hashBuffer } },
        { binding: 7, resource: this.irradianceAtlas.createView() },
        { binding: 8, resource: { buffer: this.coneBuffer } },
        { binding: 9, resource: { buffer: this.stateBuffer } },
        { binding: 10, resource: this.irradianceSampler },
        // Binding 11 aliases the normal/emission packed target. Keeping the
        // binding separate makes the fragment interface explicit while the
        // G-buffer itself remains three attachments and 32 bytes/sample.
        { binding: 11, resource: this.normalTexture.createView() },
        { binding: 12, resource: this.pointShadowArrayView },
        { binding: 13, resource: this.pointShadowSampler },
        { binding: 14, resource: { buffer: this.sunShadowDataBuffer } },
        { binding: 15, resource: { buffer: this.bvhNodeBuffer } },
        { binding: 16, resource: { buffer: this.triangleBuffer } },
        { binding: 17, resource: this.materialAtlasView },
        { binding: 18, resource: this.materialSampler },
        { binding: 19, resource: { buffer: this.emissiveBvhNodeBuffer } },
        { binding: 20, resource: { buffer: this.emissiveTriangleBuffer } },
      ],
    });
    this.presentBindGroup = this.device.createBindGroup({
      label: "current composite presentation bind group",
      layout: this.presentLayout,
      entries: [
        { binding: 0, resource: this.compositeTexture.createView() },
        { binding: 1, resource: this.presentSampler },
      ],
    });
  }

  async loadScene(index) {
    index = (index + SCENE_INFO.length) % SCENE_INFO.length;
    const token = ++this.loadToken;
    const info = SCENE_INFO[index];
    setStatus(`Building ${info.short}`, "Generating geometry and its GPU ray-tracing hierarchy…");
    await new Promise((resolve) => setTimeout(resolve, 0));
    const scene = await createScene(index);
    if (token !== this.loadToken) return;
    await this.device.queue.onSubmittedWorkDone();
    this.vertexBuffer?.destroy();
    this.dynamicVertexBuffer?.destroy();
    this.bvhNodeBuffer?.destroy();
    this.triangleBuffer?.destroy();
    this.emissiveBvhNodeBuffer?.destroy();
    this.emissiveTriangleBuffer?.destroy();
    const geometry = scene.geometry;
    const dynamicScene = createDynamicScene(index, geometry);
    const rayNodes = dynamicScene?.combinedNodes ?? geometry.nodes;
    const rayTriangles = dynamicScene?.combinedTriangles ?? geometry.triangles;
    this.vertexBuffer = createBuffer(this.device, `${info.short} raster geometry`, geometry.vertices.byteLength, GPU.VERTEX | GPU.COPY_DST, geometry.vertices);
    this.dynamicVertexBuffer = dynamicScene
      ? createBuffer(
        this.device,
        `${info.short} dynamic raster instances`,
        dynamicScene.rasterVertices.byteLength,
        GPU.VERTEX | GPU.COPY_DST,
        dynamicScene.rasterVertices,
      )
      : null;
    if (dynamicScene) dynamicScene.rasterDirty = false;
    // The portable WebGPU compute limit is eight storage bindings and this
    // renderer already uses all eight. Dynamic TLAS/BLAS data therefore lives
    // in reserved arenas inside the existing node/triangle buffers instead of
    // relying on a ninth, device-specific storage binding.
    this.bvhNodeBuffer = createBuffer(this.device, `${info.short} static BLAS + dynamic TLAS/BLAS nodes`, rayNodes.byteLength, GPU.STORAGE | GPU.COPY_DST, rayNodes);
    this.triangleBuffer = createBuffer(this.device, `${info.short} static triangles + dynamic mesh/instance arena`, rayTriangles.byteLength, GPU.STORAGE | GPU.COPY_DST, rayTriangles);
    this.emissiveBvhNodeBuffer = createBuffer(
      this.device,
      `${info.short} emissive-only BVH nodes`,
      geometry.emissiveGeometry.nodes.byteLength,
      GPU.STORAGE | GPU.COPY_DST,
      geometry.emissiveGeometry.nodes,
    );
    this.emissiveTriangleBuffer = createBuffer(
      this.device,
      `${info.short} emissive-only BVH triangles`,
      geometry.emissiveGeometry.triangles.byteLength,
      GPU.STORAGE | GPU.COPY_DST,
      geometry.emissiveGeometry.triangles,
    );
    this.dynamicScene = dynamicScene;
    this.dynamicCpuSamples.length = 0;
    this.dynamicUpdateMs = 0;
    this.dynamicUploadBytes = 0;
    this.scene = scene;
    this.sceneIndex = index;
    this.setCameraFromScene(scene);
    this.resetProbeHistory();
    this.createSizedResources();
    this.rebuildBindGroups();
    this.updateSceneUI();
    this.frameSamples.length = 0;
    this.gpuSamples.length = 0;
    hideStatus();
    console.info("[Split RC] scene-loaded", {
      index,
      name: info.name,
      triangles: geometry.triangleCount,
      dynamicInstances: dynamicScene?.instanceCount || 0,
      dynamicTriangles: dynamicScene?.triangleCount || 0,
      bvhNodes: geometry.nodeCount,
      emissiveTriangles: geometry.emissiveGeometry.emissiveTriangleCount,
    });
  }

  updateDynamicSceneGeometry(seconds) {
    if (!this.dynamicScene) return;
    const updateStarted = performance.now();
    const dynamic = this.dynamicScene.update(seconds);
    this.device.queue.writeBuffer(
      this.bvhNodeBuffer,
      dynamic.tlasNodeOffset * 32,
      dynamic.tlasData,
    );
    this.device.queue.writeBuffer(
      this.bvhNodeBuffer,
      dynamic.sweptTlasNodeOffset * 32,
      dynamic.sweptTlasData,
    );
    this.device.queue.writeBuffer(
      this.bvhNodeBuffer,
      dynamic.emissiveTlasNodeOffset * 32,
      dynamic.emissiveTlasData,
    );
    this.device.queue.writeBuffer(
      this.triangleBuffer,
      dynamic.instanceRecordOffset * 128,
      dynamic.instanceData,
    );
    if (dynamic.rasterDirty) {
      if (dynamic.rasterVertices.byteLength > this.dynamicVertexBuffer.size) {
        throw new Error("Dynamic raster topology exceeded its prepared vertex capacity.");
      }
      this.device.queue.writeBuffer(this.dynamicVertexBuffer, 0, dynamic.rasterVertices);
      dynamic.rasterDirty = false;
    }
    this.dynamicUploadBytes = dynamic.tlasData.byteLength
      + dynamic.sweptTlasData.byteLength
      + dynamic.emissiveTlasData.byteLength
      + dynamic.instanceData.byteLength;
    const elapsed = performance.now() - updateStarted;
    this.dynamicCpuSamples.push(elapsed);
    if (this.dynamicCpuSamples.length > 120) this.dynamicCpuSamples.shift();
    const sorted = [...this.dynamicCpuSamples].sort((a, b) => a - b);
    this.dynamicUpdateMs = sorted[Math.floor((sorted.length - 1) * 0.95)] || elapsed;
  }

  setCameraFromScene(scene) {
    this.setCameraPose(scene.camera, scene.target);
  }

  setCameraPose(position, target) {
    const offset = sub3(position, target);
    const horizontal = Math.hypot(offset[0], offset[2]);
    this.camera = {
      target: [...target],
      distance: Math.hypot(...offset),
      azimuth: Math.atan2(offset[2], offset[0]),
      elevation: Math.atan2(offset[1], horizontal),
    };
  }

  cameraPose(time) {
    let azimuth = this.camera.azimuth;
    let elevation = this.camera.elevation;
    let distance = this.camera.distance;
    let target = [...this.camera.target];
    if (this.animateCamera) {
      if (this.sceneIndex === 1) {
        azimuth += Math.sin(time * 0.22 + 0.37) * 0.09;
        elevation += Math.sin(time * 0.17 + 1.1) * 0.006;
        distance *= 1 + Math.sin(time * 0.13 + 0.8) * 0.012;
        target = add3(target, [
          Math.sin(time * 0.11) * 0.45,
          Math.sin(time * 0.19 + 0.4) * 0.06,
          Math.cos(time * 0.14) * 0.25,
        ]);
      } else {
        azimuth += Math.sin(time * 0.16 + this.sceneIndex * 0.37) * 0.28;
      }
    }
    const ce = Math.cos(elevation);
    const position = add3(target, [
      Math.cos(azimuth) * ce * distance,
      Math.sin(elevation) * distance,
      Math.sin(azimuth) * ce * distance,
    ]);
    return { position, target };
  }

  updateUniforms(now) {
    const seconds = this.testTimeOverride ?? this.testFrameTime ?? (now - this.startTime) / 1000;
    this.updateDynamicSceneGeometry(seconds);
    const cameraPose = this.cameraPose(seconds);
    const cameraPosition = cameraPose.position;
    const view = mat4LookAt(cameraPosition, cameraPose.target);
    const cameraNear = Math.max(0.03, this.scene.radius * 0.001);
    const cameraFar = this.scene.radius * 5 + 100;
    const projection = mat4Perspective(Math.PI / 3, this.width / this.height, cameraNear, cameraFar);
    const viewProjection = mat4Multiply(projection, view);
    this.currentViewProjection = viewProjection;
    const sunTime = this.animateLights ? seconds * this.sunSpeed : 0.7;
    const sunAngle = sunTime * 0.12 + this.sceneIndex * 0.61;
    const sunHorizontal = this.scene.sunHorizontal ?? 0.7;
    const sunHeight = this.scene.sunHeight ?? -0.74;
    const sunDirection = normalize3([
      Math.cos(sunAngle) * sunHorizontal,
      sunHeight,
      Math.sin(sunAngle) * sunHorizontal,
    ]);
    const cameraForward = normalize3(sub3(cameraPose.target, cameraPosition));
    const sceneCenterDepth = dot3(sub3(this.camera.target, cameraPosition), cameraForward);
    const shadowFar = clamp(
      sceneCenterDepth + this.scene.radius * 1.35,
      cameraNear + 12,
      cameraFar,
    );
    const sunCascades = buildSunShadowCascades({
      cameraPosition,
      cameraTarget: cameraPose.target,
      sunDirection,
      nearPlane: cameraNear,
      farPlane: shadowFar,
      aspect: this.width / this.height,
      sceneRadius: this.scene.radius,
      shadowResolution: QUALITY[this.qualityName].shadow,
    });
    const sunVP = sunCascades.matrices[0];
    const shadowData = new Float32Array(80);
    for (let cascade = 0; cascade < SUN_CASCADE_COUNT; cascade++) {
      shadowData.set(sunCascades.matrices[cascade], cascade * 16);
      this.device.queue.writeBuffer(this.sunShadowBuffers[cascade], 0, sunCascades.matrices[cascade]);
    }
    shadowData.set(sunCascades.splitDepths, 64);
    shadowData.set(sunCascades.texelSizes, 68);
    shadowData.set([...sunCascades.forward, 0], 72);
    shadowData.set([QUALITY[this.qualityName].shadow, shadowFar, 0, 0], 76);
    this.device.queue.writeBuffer(this.sunShadowDataBuffer, 0, shadowData);
    const pointAngle = sunTime * 0.67 + this.sceneIndex;
    const pointOrbit = this.scene.pointOrbit ?? this.scene.radius * 0.28;
    const pointBaseHeight = this.scene.pointBaseHeight ?? 2.5;
    const pointHeight = this.scene.pointHeight ?? 1.8;
    const pointPosition = add3(this.camera.target, [
      Math.cos(pointAngle) * pointOrbit,
      pointBaseHeight + Math.sin(pointAngle * 1.7) * pointHeight,
      Math.sin(pointAngle) * pointOrbit,
    ]);
    const pointColor = this.scene.pointColor ?? [0.08, 0.5, 1.0];
    const sunColor = this.scene.sunColor ?? [1.0, 0.84, 0.63];
    const pointIntensity = this.scene.pointIntensity ?? 10.0;
    const pointRange = this.scene.radius * 0.72;
    this.pointShadowsEnabled = pointIntensity > 0.0001;
    this.currentPointIntensity = pointIntensity;
    if (this.pointShadowsEnabled) {
      const pointProjection = mat4Perspective(
        Math.PI * 0.5,
        1,
        Math.max(0.01, pointRange * 0.001),
        pointRange,
      );
      const faceAxes = [
        [[1, 0, 0], [0, 1, 0]],
        [[-1, 0, 0], [0, 1, 0]],
        [[0, 1, 0], [0, 0, -1]],
        [[0, -1, 0], [0, 0, 1]],
        [[0, 0, 1], [0, 1, 0]],
        [[0, 0, -1], [0, 1, 0]],
      ];
      for (let face = 0; face < 6; face++) {
        const [axis, up] = faceAxes[face];
        const pointView = mat4LookAt(pointPosition, add3(pointPosition, axis), up);
        const pointVP = mat4Multiply(pointProjection, pointView);
        this.device.queue.writeBuffer(this.pointShadowBuffers[face], 0, pointVP);
      }
    }

    const u = new Float32Array(72);
    u.set(viewProjection, 0);
    u.set(sunVP, 16);
    const featureFlags = (this.temporalStability ? 8 : 0)
      | (this.scene.paperPalette ? 16 : 0)
      | (this.animateLights ? 32 : 0)
      | (this.scene.geometry.emissiveGeometry.emissiveTriangleCount > 0
        || (this.dynamicScene?.emissiveInstanceCount || 0) > 0 ? 64 : 0)
      | (this.dynamicScene ? 128 : 0);
    u.set([...cameraPosition, featureFlags], 32);
    u.set([...sunDirection, seconds], 36);
    u.set([...sunColor, this.scene.sun], 40);
    u.set([...pointPosition, pointRange], 44);
    u.set([...pointColor, pointIntensity], 48);
    u.set([...this.scene.env, this.scene.baseSpacing], 52);
    u.set([this.width, this.height, this.giWidth, this.giHeight], 56);
    const frameParity = this.frameIndex & (K.hashFrames - 1);
    // A nonzero fixed-light value enables exact sample-count accumulation for
    // measured cones; its magnitude controls only reconstructed support probes
    // (and the separate moving-light EMA). Exact-key rejection still makes
    // disocclusions immediate.
    // Support probes have no primary ray of their own, so a long-tail EMA can
    // make the reconstructed field depend on how long a camera path happened
    // to run before returning to a pose. A fixed, quickly convergent weight
    // reaches the same steady state after every reset while exact measured
    // cones still use the sample-count accumulation in mergeCascade.
    const fixedLightHistory = 0.92;
    const historyBlend = this.temporalStability && this.historyValid
      ? (this.animateLights
        ? 0.965
        : fixedLightHistory)
      : 0;
    const exposure = this.scene.exposure ?? 1.0;
    u.set([this.indirectStrength, exposure, this.debugMode, frameParity + historyBlend], 60);
    const boundsMin = this.scene.geometry.boundsMin;
    const boundsMax = this.scene.geometry.boundsMax;
    const sceneDiagonal = Math.hypot(
      boundsMax[0] - boundsMin[0],
      boundsMax[1] - boundsMin[1],
      boundsMax[2] - boundsMin[2],
    );
    u.set([...boundsMin, Math.max(this.scene.baseSpacing, sceneDiagonal)], 64);
    const uniformWords = new Uint32Array(u.buffer);
    uniformWords.set(
      this.dynamicScene?.frameInfo() ?? [0xffffffff, 0, 0, 0],
      68,
    );
    this.device.queue.writeBuffer(this.frameBuffer, 0, u);
  }

  render(now) {
    this.createSizedResources();
    this.updateUniforms(now);
    for (let cascade = 0; cascade < 4; cascade++) {
      this.device.queue.writeBuffer(
        this.passBuffers[cascade],
        0,
        new Uint32Array([
          cascade,
          this.raysPerSample,
          this.rayBlockSize,
          0,
          this.sampleFrameIndex >>> 0,
          this.sampleEpoch >>> 0,
          0,
          0,
        ]),
      );
    }
    const d = this.device;
    const encoder = d.createCommandEncoder({ label: `Split RC frame ${this.frameIndex}` });
    const profile = this.timestampSupported && !this.profilePending && this.frameIndex % 45 === 0;
    const captureJob = this.captureRequest;
    this.captureRequest = null;

    for (let cascade = 0; cascade < SUN_CASCADE_COUNT; cascade++) {
      const timestampWrites = profile && (cascade === 0 || cascade === SUN_CASCADE_COUNT - 1)
        ? {
            querySet: this.querySet,
            ...(cascade === 0 ? { beginningOfPassWriteIndex: 0 } : {}),
            ...(cascade === SUN_CASCADE_COUNT - 1 ? { endOfPassWriteIndex: 1 } : {}),
          }
        : undefined;
      const shadow = encoder.beginRenderPass({
        label: `stabilized sun shadow cascade ${cascade}`,
        colorAttachments: [],
        depthStencilAttachment: {
          view: this.sunShadowCascadeViews[cascade],
          depthClearValue: 1,
          depthLoadOp: "clear",
          depthStoreOp: "store",
        },
        timestampWrites,
      });
      shadow.setPipeline(this.shadowPipeline);
      shadow.setBindGroup(0, this.sunShadowBindGroups[cascade]);
      shadow.setVertexBuffer(0, this.vertexBuffer);
      shadow.draw(this.scene.geometry.vertexCount);
      if (this.dynamicScene?.vertexCount) {
        shadow.setPipeline(this.dynamicShadowPipeline);
        shadow.setBindGroup(0, this.dynamicSunShadowBindGroups[cascade]);
        shadow.setVertexBuffer(0, this.dynamicVertexBuffer);
        shadow.draw(this.dynamicScene.vertexCount);
      }
      shadow.end();
    }

    if (this.pointShadowsEnabled) {
      for (let face = 0; face < 6; face++) {
        const pointShadow = encoder.beginRenderPass({
          label: `point-light cube shadow face ${face}`,
          colorAttachments: [],
          depthStencilAttachment: {
            view: this.pointShadowFaceViews[face],
            depthClearValue: 1,
            depthLoadOp: "clear",
            depthStoreOp: "store",
          },
        });
        pointShadow.setPipeline(this.pointShadowPipeline);
        pointShadow.setBindGroup(0, this.pointShadowBindGroups[face]);
        pointShadow.setVertexBuffer(0, this.vertexBuffer);
        pointShadow.draw(this.scene.geometry.vertexCount);
        if (this.dynamicScene?.vertexCount) {
          pointShadow.setPipeline(this.dynamicPointShadowPipeline);
          pointShadow.setBindGroup(0, this.dynamicPointShadowBindGroups[face]);
          pointShadow.setVertexBuffer(0, this.dynamicVertexBuffer);
          pointShadow.draw(this.dynamicScene.vertexCount);
        }
        pointShadow.end();
      }
    }

    const gbuffer = encoder.beginRenderPass({
      label: "G-buffer",
      colorAttachments: [
        { view: this.albedoTexture.createView(), clearValue: [0,0,0,0], loadOp: "clear", storeOp: "store" },
        { view: this.normalTexture.createView(), clearValue: [0,0,0,0], loadOp: "clear", storeOp: "store" },
        { view: this.worldTexture.createView(), clearValue: [0,0,0,0], loadOp: "clear", storeOp: "store" },
      ],
      depthStencilAttachment: { view: this.depthTexture.createView(), depthClearValue: 1, depthLoadOp: "clear", depthStoreOp: "store" },
      timestampWrites: profile ? { querySet: this.querySet, beginningOfPassWriteIndex: 2, endOfPassWriteIndex: 3 } : undefined,
    });
    gbuffer.setPipeline(this.gbufferPipeline);
    gbuffer.setBindGroup(0, this.rasterBindGroup);
    gbuffer.setVertexBuffer(0, this.vertexBuffer);
    gbuffer.draw(this.scene.geometry.vertexCount);
    if (this.dynamicScene?.vertexCount) {
      gbuffer.setPipeline(this.dynamicGbufferPipeline);
      gbuffer.setBindGroup(0, this.dynamicRasterBindGroup);
      gbuffer.setVertexBuffer(0, this.dynamicVertexBuffer);
      gbuffer.draw(this.dynamicScene.vertexCount);
    }
    gbuffer.end();

    encoder.clearBuffer(this.stateBuffer);
    const accumFrameBytes = K.totalDirectionData * 5 * 4;
    encoder.clearBuffer(this.accumBuffer, (this.frameIndex & 1) * accumFrameBytes, accumFrameBytes);
    let pass = encoder.beginComputePass({ label: "classify environment access" });
    pass.setPipeline(this.computePipelines.classifyEnvironmentAccess);
    pass.setBindGroup(0, this.computeBindGroups[0]);
    pass.dispatchWorkgroups(Math.ceil(512 / 64));
    pass.end();
    pass = encoder.beginComputePass({
      label: "reset sparse hash",
      timestampWrites: profile ? { querySet: this.querySet, beginningOfPassWriteIndex: 4 } : undefined,
    });
    pass.setPipeline(this.computePipelines.reset);
    pass.setBindGroup(0, this.computeBindGroups[0]);
    pass.dispatchWorkgroups(Math.ceil(K.totalHashSlots / 256));
    pass.end();

    pass = encoder.beginComputePass({ label: "initialize sparse probes" });
    pass.setPipeline(this.computePipelines.initBase);
    pass.setBindGroup(0, this.computeBindGroups[0]);
    pass.dispatchWorkgroups(Math.ceil(this.width / 8), Math.ceil(this.height / 8));
    pass.end();
    pass = encoder.beginComputePass({ label: "canonicalize base probe indices" });
    pass.setPipeline(this.computePipelines.canonicalize);
    pass.setBindGroup(0, this.computeBindGroups[0]);
    pass.dispatchWorkgroups(Math.ceil(K.hashSizes[0] / 64));
    pass.end();
    pass = encoder.beginComputePass({ label: "count deterministic base rays" });
    pass.setPipeline(this.computePipelines.countBase);
    pass.setBindGroup(0, this.computeBindGroups[0]);
    pass.dispatchWorkgroups(Math.ceil(this.giWidth / 8), Math.ceil(this.giHeight / 8));
    pass.end();
    for (let cascade = 1; cascade < 4; cascade++) {
      pass = encoder.beginComputePass({ label: `initialize cascade ${cascade}` });
      pass.setPipeline(this.computePipelines.initHigher);
      pass.setBindGroup(0, this.computeBindGroups[cascade]);
      pass.dispatchWorkgroups(Math.ceil(K.probeCaps[cascade - 1] / 64));
      pass.end();
      pass = encoder.beginComputePass({ label: `canonicalize cascade ${cascade} probe indices` });
      pass.setPipeline(this.computePipelines.canonicalize);
      pass.setBindGroup(0, this.computeBindGroups[cascade]);
      pass.dispatchWorkgroups(Math.ceil(K.hashSizes[cascade] / 64));
      pass.end();
      pass = encoder.beginComputePass({ label: `count cascade ${cascade} rays` });
      pass.setPipeline(this.computePipelines.countHigher);
      pass.setBindGroup(0, this.computeBindGroups[cascade]);
      pass.dispatchWorkgroups(Math.ceil(K.probeCaps[cascade - 1] / 64));
      pass.end();
    }

    for (let cascade = 3; cascade >= 0; cascade--) {
      pass = encoder.beginComputePass({ label: `assign hierarchical R2 offsets c${cascade}` });
      pass.setPipeline(this.computePipelines.assignOffsets);
      pass.setBindGroup(0, this.computeBindGroups[cascade]);
      pass.dispatchWorkgroups(Math.ceil(K.probeCaps[cascade] / 64));
      pass.end();
    }

    pass = encoder.beginComputePass({ label: "map deterministic primary ray samples" });
    pass.setPipeline(this.computePipelines.mapPrimary);
    pass.setBindGroup(0, this.computeBindGroups[0]);
    pass.dispatchWorkgroups(Math.ceil(this.giWidth / 8), Math.ceil(this.giHeight / 8), this.raysPerSample);
    pass.end();
    pass = encoder.beginComputePass({ label: "prefix Algorithm 3 sample blocks" });
    pass.setPipeline(this.computePipelines.prefixRayBlocks);
    pass.setBindGroup(0, this.computeBindGroups[0]);
    pass.dispatchWorkgroups(K.probeCaps[0]);
    pass.end();

    pass = encoder.beginComputePass({ label: "trace and split surface rays" });
    pass.setPipeline(this.computePipelines.splitRays);
    pass.setBindGroup(0, this.computeBindGroups[0]);
    pass.dispatchWorkgroups(Math.ceil(this.giWidth / 8), Math.ceil(this.giHeight / 8), this.raysPerSample);
    pass.end();

    for (let cascade = 3; cascade >= 0; cascade--) {
      pass = encoder.beginComputePass({ label: `merge radiance cascade ${cascade}` });
      pass.setPipeline(this.computePipelines.merge);
      pass.setBindGroup(0, this.computeBindGroups[cascade]);
      pass.dispatchWorkgroups(Math.ceil(K.probeCaps[cascade] * K.directions[cascade] / 64));
      pass.end();
    }
    pass = encoder.beginComputePass({ label: "find measured same-sheet support sources" });
    pass.setPipeline(this.computePipelines.resolveTangentSupportSources);
    pass.setBindGroup(0, this.computeBindGroups[0]);
    pass.dispatchWorkgroups(Math.ceil(K.probeCaps[0] / 64));
    pass.end();
    pass = encoder.beginComputePass({
      label: "prefilter probe irradiance",
    });
    pass.setPipeline(this.computePipelines.prefilter);
    pass.setBindGroup(0, this.computeBindGroups[0]);
    pass.dispatchWorkgroups(Math.ceil(K.probeCaps[0] * K.irradianceTexels / 64));
    pass.end();
    pass = encoder.beginComputePass({
      label: "resolve same-sheet tangent support irradiance",
      timestampWrites: profile ? { querySet: this.querySet, endOfPassWriteIndex: 5 } : undefined,
    });
    pass.setPipeline(this.computePipelines.resolveTangentSupportIrradiance);
    pass.setBindGroup(0, this.computeBindGroups[0]);
    pass.dispatchWorkgroups(Math.ceil(K.probeCaps[0] * K.irradianceTexels / 64));
    pass.end();
    const atlasFrameY = (this.frameIndex & (K.hashFrames - 1)) * K.irradianceAtlasFrameHeight;
    encoder.copyTextureToTexture(
      { texture: this.irradianceAtlasWrite, origin: [0, atlasFrameY, 0] },
      { texture: this.irradianceAtlas, origin: [0, atlasFrameY, 0] },
      [K.irradianceAtlasWidth, K.irradianceAtlasFrameHeight, 1],
    );

    const finalPass = encoder.beginRenderPass({
      label: "tone-mapped current composite",
      colorAttachments: [{ view: this.compositeTexture.createView(), clearValue: [0,0,0,1], loadOp: "clear", storeOp: "store" }],
      timestampWrites: profile ? { querySet: this.querySet, beginningOfPassWriteIndex: 6 } : undefined,
    });
    finalPass.setPipeline(this.finalPipeline);
    finalPass.setBindGroup(0, this.finalBindGroup);
    finalPass.draw(3);
    finalPass.end();

    const presentPass = encoder.beginRenderPass({
      label: "present current Split RC composite",
      colorAttachments: [{
        view: this.context.getCurrentTexture().createView(),
        clearValue: [0, 0, 0, 1],
        loadOp: "clear",
        storeOp: "store",
      }],
      timestampWrites: profile ? { querySet: this.querySet, endOfPassWriteIndex: 7 } : undefined,
    });
    presentPass.setPipeline(this.presentPipeline);
    presentPass.setBindGroup(0, this.presentBindGroup);
    presentPass.draw(3);
    presentPass.end();

    let captureResources;
    if (captureJob) {
      const bytesPerRow = Math.ceil(this.width * 4 / 256) * 256;
      const worldBytesPerRow = Math.ceil(this.width * 16 / 256) * 256;
      const normalBytesPerRow = Math.ceil(this.width * 8 / 256) * 256;
      const buffer = createBuffer(
        d,
        "final-frame audit readback",
        bytesPerRow * this.height,
        GPU.COPY_DST | GPU.MAP_READ,
      );
      const worldBuffer = createBuffer(
        d,
        "world-position audit readback",
        worldBytesPerRow * this.height,
        GPU.COPY_DST | GPU.MAP_READ,
      );
      const normalBuffer = createBuffer(
        d,
        "surface-normal audit readback",
        normalBytesPerRow * this.height,
        GPU.COPY_DST | GPU.MAP_READ,
      );
      const diagnosticBuffer = createBuffer(
        d,
        "per-capture sparse diagnostics",
        64,
        GPU.COPY_DST | GPU.MAP_READ,
      );
      encoder.copyTextureToBuffer(
        { texture: this.compositeTexture },
        { buffer, bytesPerRow, rowsPerImage: this.height },
        [this.width, this.height],
      );
      encoder.copyTextureToBuffer(
        { texture: this.worldTexture },
        { buffer: worldBuffer, bytesPerRow: worldBytesPerRow, rowsPerImage: this.height },
        [this.width, this.height],
      );
      encoder.copyTextureToBuffer(
        { texture: this.normalTexture },
        { buffer: normalBuffer, bytesPerRow: normalBytesPerRow, rowsPerImage: this.height },
        [this.width, this.height],
      );
      encoder.copyBufferToBuffer(this.stateBuffer, 0, diagnosticBuffer, 0, 64);
      captureResources = {
        ...captureJob,
        buffer,
        worldBuffer,
        normalBuffer,
        diagnosticBuffer,
        bytesPerRow,
        worldBytesPerRow,
        normalBytesPerRow,
        width: this.width,
        height: this.height,
        viewProjection: new Float32Array(this.currentViewProjection),
        baseSpacing: this.scene.baseSpacing,
      };
    }
    let timestampRead;
    if (profile) {
      timestampRead = createBuffer(d, "timestamp readback", 64, GPU.COPY_DST | GPU.MAP_READ);
      encoder.resolveQuerySet(this.querySet, 0, 8, this.queryResolveBuffer, 0);
      encoder.copyBufferToBuffer(this.queryResolveBuffer, 0, timestampRead, 0, 64);
      this.profilePending = true;
    }
    let stateRead;
    const readState = !this.statusPending && this.frameIndex % 30 === 0;
    if (readState) {
      stateRead = createBuffer(d, "diagnostic readback", 64, GPU.COPY_DST | GPU.MAP_READ);
      encoder.copyBufferToBuffer(this.stateBuffer, 0, stateRead, 0, 64);
      this.statusPending = true;
    }
    d.queue.submit([encoder.finish()]);
    this.historyValid = true;
    if (timestampRead) this.consumeTimestamps(timestampRead);
    if (stateRead) this.consumeState(stateRead);
    if (captureResources) this.consumeFinalCapture(captureResources);
  }

  captureFinalFrame() {
    if (this.captureRequest) return Promise.reject(new Error("A final-frame capture is already pending."));
    return new Promise((resolve, reject) => {
      this.captureRequest = { resolve, reject };
    });
  }

  async consumeFinalCapture(job) {
    try {
      await Promise.all([
        job.buffer.mapAsync(MAP.READ),
        job.worldBuffer.mapAsync(MAP.READ),
        job.normalBuffer.mapAsync(MAP.READ),
        job.diagnosticBuffer.mapAsync(MAP.READ),
      ]);
      const diagnostics = new Uint32Array(job.diagnosticBuffer.getMappedRange().slice(0));
      job.resolve({
        width: job.width,
        height: job.height,
        bytesPerRow: job.bytesPerRow,
        pixels: new Uint8Array(job.buffer.getMappedRange().slice(0)),
        worldBytesPerRow: job.worldBytesPerRow,
        worldPixels: new Float32Array(job.worldBuffer.getMappedRange().slice(0)),
        normalBytesPerRow: job.normalBytesPerRow,
        normalPixels: new Uint16Array(job.normalBuffer.getMappedRange().slice(0)),
        diagnostics,
        diagnosticOverflows: diagnostics[6] + diagnostics[7],
        viewProjection: job.viewProjection,
        baseSpacing: job.baseSpacing,
      });
    } catch (error) {
      job.reject(error);
    } finally {
      job.buffer.destroy();
      job.worldBuffer.destroy();
      job.normalBuffer.destroy();
      job.diagnosticBuffer.destroy();
    }
  }

  async consumeTimestamps(buffer) {
    try {
      await buffer.mapAsync(MAP.READ);
      const q = new BigUint64Array(buffer.getMappedRange().slice(0));
      const ms = (a, b) => Number(q[b] - q[a]) / 1e6;
      this.passTimes = {
        frame: ms(0, 7),
        geometry: ms(0, 3),
        gi: ms(4, 5),
        composite: ms(6, 7),
      };
      this.gpuSamples.push(this.passTimes.frame);
      if (this.gpuSamples.length > 90) this.gpuSamples.shift();
    } catch (error) {
      console.warn("[Split RC] timestamp readback failed", error);
    } finally {
      buffer.destroy();
      this.profilePending = false;
    }
  }

  async consumeState(buffer) {
    try {
      await buffer.mapAsync(MAP.READ);
      const s = new Uint32Array(buffer.getMappedRange().slice(0));
      this.probeCounts = [...s.slice(0, 4)];
      this.rayCount = s[4];
      this.hitCount = s[5];
      this.overflowCount = s[6] + s[7];
      this.environmentAccess = s[8] !== 0;
    } catch (error) {
      console.warn("[Split RC] diagnostic readback failed", error);
    } finally {
      buffer.destroy();
      this.statusPending = false;
    }
  }

  frame(now) {
    if (!this.running || this.destroyed) return;
    if (this.readbackPause) {
      requestAnimationFrame((time) => this.frame(time));
      return;
    }
    const dt = Math.min(100, now - this.lastTime);
    this.lastTime = now;
    this.updateKeyboard(dt / 1000);
    if (this.scene && this.computeBindGroups) {
      try {
        this.render(now);
      } catch (error) {
        this.running = false;
        setStatus("Renderer stopped", error.message || String(error), true);
        console.error(error);
        return;
      }
    }
    this.frameSamples.push(dt);
    if (this.frameSamples.length > 120) this.frameSamples.shift();
    if (this.frameIndex % 10 === 0) this.updateMetrics();
    if (this.testTimeOverride == null && this.testFrameTime != null && this.testFrameStep != null) {
      this.testFrameTime += this.testFrameStep;
    }
    this.sampleFrameIndex = (this.sampleFrameIndex + 1) >>> 0;
    this.frameIndex++;
    requestAnimationFrame((t) => this.frame(t));
  }

  updateKeyboard(dt) {
    if (!this.keys.size || !this.scene) return;
    const speed = this.camera.distance * dt * 0.35;
    const forward = normalize3([-Math.cos(this.camera.azimuth), 0, -Math.sin(this.camera.azimuth)]);
    const right = [-forward[2], 0, forward[0]];
    let delta = [0,0,0];
    if (this.keys.has("KeyW")) delta = add3(delta, mul3(forward, speed));
    if (this.keys.has("KeyS")) delta = add3(delta, mul3(forward, -speed));
    if (this.keys.has("KeyA")) delta = add3(delta, mul3(right, -speed));
    if (this.keys.has("KeyD")) delta = add3(delta, mul3(right, speed));
    this.camera.target = add3(this.camera.target, delta);
  }

  updateMetrics() {
    const avg = this.frameSamples.reduce((a,b)=>a+b,0) / Math.max(1,this.frameSamples.length);
    const fps = avg > 0 ? 1000 / avg : 0;
    const gpu = this.gpuSamples.length ? this.gpuSamples.reduce((a,b)=>a+b,0)/this.gpuSamples.length : 0;
    if ($("metric-fps")) $("metric-fps").textContent = fps.toFixed(0);
    if ($("metric-gpu")) $("metric-gpu").textContent = gpu ? gpu.toFixed(1) : "n/a";
    if ($("metric-rays")) $("metric-rays").textContent = this.rayCount ? `${(this.rayCount/1000).toFixed(0)}k` : "—";
    if ($("metric-probes")) $("metric-probes").textContent = this.probeCounts.reduce((a,b)=>a+b,0) || "—";
    if ($("gi-resolution")) {
      const scaled = this.resolutionScale < 0.995 ? ` · ${Math.round(this.resolutionScale * 100)}% fill` : "";
      $("gi-resolution").textContent = `${this.giWidth}×${this.giHeight} · ${this.raysPerSample} spp${scaled}`;
    }
    const pass = this.passTimes;
    for (const name of ["frame","geometry","gi","composite"]) {
      const value = pass[name] || 0;
      if ($(`pass-${name}`)) $(`pass-${name}`).textContent = value ? `${value.toFixed(2)} ms` : "sampling";
      if ($(`bar-${name}`)) $(`bar-${name}`).style.width = `${clamp(value / 22 * 100, 2, 100)}%`;
    }
    if ($("dynamic-profiler-row")) {
      $("dynamic-profiler-row").hidden = !this.dynamicScene;
      if (this.dynamicScene && $("pass-dynamic")) {
        $("pass-dynamic").textContent = `${this.dynamicScene.instanceCount} inst · ${this.dynamicUpdateMs.toFixed(2)} ms CPU · ${(this.dynamicUploadBytes / 1024).toFixed(1)} KiB`;
        $("bar-dynamic").style.width = `${clamp(this.dynamicUpdateMs / 2 * 100, 2, 100)}%`;
      }
    }
    // Default tiers hold one Algorithm 3 owner per internal pixel, but adapt
    // that internal pixel budget on slower devices. This directly targets
    // sustained frame time instead of inferring performance from DPR or user
    // agent, and it never creates a sparse/subsampled ray lattice.
    const adaptiveTier = this.qualityName === "balanced" || this.qualityName === "performance";
    // React quickly to the first sustained result: on a 12 FPS device a
    // 90-frame window left the default workload visibly slow for 7.5 seconds.
    // Later adjustments use a longer window so the budget does not hunt.
    const firstAdjustment = this.lastAdaptiveAdjustmentFrame === 0
      && this.adaptivePixelBudgetScale >= 0.995;
    const requiredSamples = firstAdjustment ? 30 : 60;
    const adjustmentInterval = firstAdjustment ? 30 : 60;
    const enoughSamples = this.frameSamples.length >= requiredSamples;
    const adjustmentDue = this.frameIndex - this.lastAdaptiveAdjustmentFrame >= adjustmentInterval;
    if (
      adaptiveTier
      && !this.automaticTest
      && document.visibilityState === "visible"
      && enoughSamples
      && adjustmentDue
    ) {
      // CPU rAF timing is vsync-limited. Prefer GPU timestamps. On browsers
      // that do not expose them, a downscaled renderer sustaining a 60 Hz
      // cadence makes one conservative upscale probe every five seconds; a
      // missed refresh still triggers the fast downscale path after 60 frames.
      const vsyncRecoveryProbe = gpu <= 0
        && this.adaptivePixelBudgetScale < 0.995
        && avg >= 15
        && avg <= 17.8
        && this.frameIndex - this.lastAdaptiveAdjustmentFrame >= 300;
      const observedMs = gpu > 0 ? gpu : (vsyncRecoveryProbe ? 12.5 : avg);
      const nextScale = adaptiveBudgetScale(this.adaptivePixelBudgetScale, observedMs);
      if (Math.abs(nextScale - this.adaptivePixelBudgetScale) >= 0.025) {
        this.adaptivePixelBudgetScale = nextScale;
        this.lastAdaptiveAdjustmentFrame = this.frameIndex;
        this.width = this.height = 0;
        this.frameSamples.length = 0;
        this.gpuSamples.length = 0;
        this.createSizedResources();
      }
    }
  }

  updateSceneUI() {
    const info = SCENE_INFO[this.sceneIndex];
    $("scene-name").textContent = info.name;
    const totalTriangles = this.scene.geometry.triangleCount + (this.dynamicScene?.triangleCount || 0);
    $("scene-description").textContent = `${info.description} ${totalTriangles.toLocaleString()} ray-traced triangles.`;
    $("scene-index").textContent = `${String(this.sceneIndex+1).padStart(2,"0")} / ${SCENE_INFO.length}`;
    $("scene-select").value = String(this.sceneIndex);
    document.querySelectorAll(".scene-strip button").forEach((button, i) => button.classList.toggle("active", i === this.sceneIndex));
  }

  installUI() {
    const select = $("scene-select");
    const strip = $("scene-strip");
    // Startup is generation-safe, but keep DOM installation idempotent too so
    // HMR/manual restarts can never duplicate options or shortcut buttons.
    select.replaceChildren();
    strip.replaceChildren();
    $("quality").value = this.qualityName;
    SCENE_INFO.forEach((scene, index) => {
      const option = document.createElement("option");
      option.value = String(index);
      option.textContent = `${String(index+1).padStart(2,"0")} — ${scene.name}`;
      select.append(option);
      const button = document.createElement("button");
      button.textContent = String(index+1).padStart(2,"0");
      button.title = scene.name;
      button.addEventListener("click", () => this.loadScene(index));
      strip.append(button);
    });
    const on = (element, event, fn, options) => {
      element.addEventListener(event, fn, options);
      this.cleanup.push(() => element.removeEventListener(event, fn, options));
    };
    on(select, "change", () => this.loadScene(Number(select.value)));
    on($("prev-scene"), "click", () => this.loadScene(this.sceneIndex - 1));
    on($("next-scene"), "click", () => this.loadScene(this.sceneIndex + 1));
    on($("toggle-panel"), "click", () => $("control-panel").classList.toggle("closed"));
    on($("debug-view"), "change", (e) => { this.debugMode = Number(e.target.value); });
    on($("quality"), "change", (e) => {
      this.qualityName = e.target.value;
      this.adaptivePixelBudgetScale = 1;
      this.lastAdaptiveAdjustmentFrame = this.frameIndex;
      this.width = this.height = 0;
      this.createSizedResources();
    });
    on($("indirect-strength"), "input", (e) => {
      this.indirectStrength = Number(e.target.value);
      $("indirect-value").textContent = `${this.indirectStrength.toFixed(2)}×`;
    });
    on($("sun-speed"), "input", (e) => {
      this.sunSpeed = Number(e.target.value);
      $("sun-value").textContent = `${this.sunSpeed.toFixed(2)}×`;
    });
    on($("animate-camera"), "change", (e) => { this.animateCamera = e.target.checked; });
    on($("animate-lights"), "change", (e) => {
      this.animateLights = e.target.checked;
      // Toggling animation changes the lighting clock discontinuously (the
      // fixed audit/light pose is t=0.7). Never retain radiance from the old
      // pose under the longer fixed-light history weight.
      this.resetProbeHistory();
    });
    on($("temporal-stability"), "change", (e) => {
      this.temporalStability = e.target.checked;
      this.resetProbeHistory();
    });
    on($("show-profiler"), "change", (e) => { $("pass-profiler").hidden = !e.target.checked; });
    on($("run-validation"), "click", () => this.runValidation());
    on($("close-audit"), "click", () => { $("audit-card").hidden = true; });

    on(this.canvas, "pointerdown", (e) => {
      this.mouse = { down: true, x: e.clientX, y: e.clientY };
      this.canvas.setPointerCapture(e.pointerId);
      this.animateCamera = false;
      $("animate-camera").checked = false;
    });
    on(this.canvas, "pointermove", (e) => {
      if (!this.mouse.down) return;
      this.camera.azimuth -= (e.clientX - this.mouse.x) * 0.006;
      this.camera.elevation = clamp(this.camera.elevation + (e.clientY - this.mouse.y) * 0.004, -1.25, 1.25);
      this.mouse.x = e.clientX; this.mouse.y = e.clientY;
    });
    on(this.canvas, "pointerup", () => { this.mouse.down = false; });
    on(this.canvas, "wheel", (e) => {
      e.preventDefault();
      this.camera.distance = clamp(this.camera.distance * Math.exp(e.deltaY * 0.001), this.scene.radius * 0.08, this.scene.radius * 4);
    }, { passive: false });
    on(window, "keydown", (e) => {
      if (e.code === "Space") {
        e.preventDefault();
        this.animateCamera = !this.animateCamera;
        $("animate-camera").checked = this.animateCamera;
      }
      this.keys.add(e.code);
    });
    on(window, "keyup", (e) => this.keys.delete(e.code));
    on(window, "resize", () => { this.width = this.height = 0; });
  }

  async waitFrames(count) {
    const target = this.frameIndex + count;
    while (this.frameIndex < target && this.running) {
      await new Promise((resolve) => requestAnimationFrame(resolve));
    }
  }

  async readGpuBuffer(source, size, label) {
    const readback = createBuffer(this.device, label, size, GPU.COPY_DST | GPU.MAP_READ);
    const encoder = this.device.createCommandEncoder({ label });
    encoder.copyBufferToBuffer(source, 0, readback, 0, size);
    this.device.queue.submit([encoder.finish()]);
    await readback.mapAsync(MAP.READ);
    const bytes = readback.getMappedRange().slice(0);
    readback.unmap();
    readback.destroy();
    return bytes;
  }

  async measureWorldProbeStability() {
    this.readbackPause = true;
    let hashReadback = null;
    let irradianceReadback = null;
    let hash;
    let fields;
    let current;
    let previous;
    try {
      await this.device.queue.onSubmittedWorkDone();
      const hashBytes = K.totalHashSlots * K.hashFrames * 8;
      const irradianceBytes = K.probeCaps[0] * K.irradianceTexels * K.irradianceFrames * 16;
      // Capture both buffers and the history-frame index in one submission. Separate
      // asynchronous copies allowed animation frames to advance between copies,
      // occasionally pairing a hash frame with the wrong irradiance frame.
      current = (this.frameIndex - 1) & (K.hashFrames - 1);
      previous = (current + K.hashFrames - 1) & (K.hashFrames - 1);
      hashReadback = createBuffer(this.device, "stability hash readback", hashBytes, GPU.COPY_DST | GPU.MAP_READ);
      irradianceReadback = createBuffer(this.device, "stability irradiance readback", irradianceBytes, GPU.COPY_DST | GPU.MAP_READ);
      const encoder = this.device.createCommandEncoder({ label: "atomic world-probe stability snapshot" });
      encoder.copyBufferToBuffer(this.hashBuffer, 0, hashReadback, 0, hashBytes);
      encoder.copyBufferToBuffer(this.irradianceBuffer, 0, irradianceReadback, 0, irradianceBytes);
      this.device.queue.submit([encoder.finish()]);
      await Promise.all([hashReadback.mapAsync(MAP.READ), irradianceReadback.mapAsync(MAP.READ)]);
      hash = new Uint32Array(hashReadback.getMappedRange().slice(0));
      fields = new Float32Array(irradianceReadback.getMappedRange().slice(0));
    } finally {
      if (hashReadback?.mapState === "mapped") hashReadback.unmap();
      if (irradianceReadback?.mapState === "mapped") irradianceReadback.unmap();
      hashReadback?.destroy();
      irradianceReadback?.destroy();
      this.readbackPause = false;
    }
    const frameMap = (frame) => {
      const map = new Map();
      let validSlots = 0;
      const base = frame * K.totalHashSlots * 2;
      for (let slot = 0; slot < K.hashSizes[0]; slot++) {
        const key = hash[base + slot * 2];
        const index = hash[base + slot * 2 + 1];
        if (key !== 0xffffffff && index < K.probeCaps[0]) {
          validSlots++;
          map.set(key, index);
        }
      }
      return { map, validSlots };
    };
    const currentFrameMap = frameMap(current);
    const previousFrameMap = frameMap(previous);
    const currentMap = currentFrameMap.map;
    const previousMap = previousFrameMap.map;
    const luminance = (frame, probe) => {
      const base = (frame * K.probeCaps[0] * K.irradianceTexels + probe * K.irradianceTexels) * 4;
      let sum = 0;
      for (let texel = 0; texel < K.irradianceTexels; texel++) {
        const i = base + texel * 4;
        sum += fields[i] * 0.2126 + fields[i + 1] * 0.7152 + fields[i + 2] * 0.0722;
      }
      return sum / K.irradianceTexels;
    };
    const relative = [];
    const absolute = [];
    for (const [key, probe] of currentMap) {
      const oldProbe = previousMap.get(key);
      if (oldProbe == null) continue;
      const a = luminance(current, probe);
      const b = luminance(previous, oldProbe);
      const delta = Math.abs(a - b);
      absolute.push(delta);
      relative.push(delta / Math.max(0.05, (a + b) * 0.5));
    }
    relative.sort((a, b) => a - b);
    absolute.sort((a, b) => a - b);
    const percentile = (values, p) => values.length ? values[Math.min(values.length - 1, Math.floor((values.length - 1) * p))] : Infinity;
    return {
      currentProbes: currentMap.size,
      previousProbes: previousMap.size,
      currentValidSlots: currentFrameMap.validSlots,
      previousValidSlots: previousFrameMap.validSlots,
      matchedProbes: relative.length,
      medianRelative: percentile(relative, 0.5),
      p95Relative: percentile(relative, 0.95),
      p99Relative: percentile(relative, 0.99),
      p95Absolute: percentile(absolute, 0.95),
      maxAbsolute: absolute.at(-1) ?? Infinity,
    };
  }

  async runMotionStabilityAudit({ samples = 5, interval = 4, warmup = 32 } = {}) {
    const saved = {
      animateCamera: this.animateCamera,
      animateLights: this.animateLights,
      temporalStability: this.temporalStability,
      testTimeOverride: this.testTimeOverride,
      testFrameTime: this.testFrameTime,
      testFrameStep: this.testFrameStep,
    };
    try {
      this.animateCamera = true;
      this.animateLights = false;
      this.temporalStability = true;
      this.testTimeOverride = null;
      this.testFrameTime = 0.8;
      this.testFrameStep = 1 / 60;
      this.resetProbeHistory();
      await this.waitFrames(warmup);
      const measurements = [];
      for (let i = 0; i < samples; i++) {
        await this.waitFrames(interval);
        measurements.push(await this.measureWorldProbeStability());
      }
      const valid = measurements.filter((m) => Number.isFinite(m.p95Relative) && m.matchedProbes >= 16);
      const maximum = (name) => valid.length ? Math.max(...valid.map((m) => m[name])) : Infinity;
      const minimum = (name) => valid.length ? Math.min(...valid.map((m) => m[name])) : 0;
      return {
        samples: valid.length,
        matchedProbesMin: minimum("matchedProbes"),
        p95RelativeMax: maximum("p95Relative"),
        p99RelativeMax: maximum("p99Relative"),
        p95AbsoluteMax: maximum("p95Absolute"),
        // Dark probe fields can have a large percentage change with an
        // imperceptibly small absolute delta. Require the strict global
        // absolute ceiling plus either the relative ceiling or a 0.006
        // scene-linear dark-field ceiling (less than 1.6/255 before tone
        // mapping). Final-frame motion has its own stricter byte-domain gate.
        passed: valid.length === samples
          && maximum("p95Absolute") <= 0.012
          && (maximum("p95Relative") <= 0.015 || maximum("p95Absolute") <= 0.006),
        measurements,
      };
    } finally {
      Object.assign(this, saved);
      this.resetProbeHistory();
    }
  }

  compareFinalFrames(a, b) {
    if (a.width !== b.width || a.height !== b.height) {
      return { passed: false, reason: "capture dimensions changed" };
    }
    const differences = [];
    let squared = 0;
    let changed = 0;
    for (let y = 0; y < a.height; y++) {
      const rowA = y * a.bytesPerRow;
      const rowB = y * b.bytesPerRow;
      for (let x = 0; x < a.width; x++) {
        const pixelA = rowA + x * 4;
        const pixelB = rowB + x * 4;
        for (let channel = 0; channel < 3; channel++) {
          const delta = Math.abs(a.pixels[pixelA + channel] - b.pixels[pixelB + channel]);
          differences.push(delta);
          squared += delta * delta;
          if (delta > 0) changed++;
        }
      }
    }
    differences.sort((x, y) => x - y);
    const percentile = (p) => differences[Math.min(differences.length - 1, Math.floor((differences.length - 1) * p))] || 0;
    const rmse = Math.sqrt(squared / Math.max(1, differences.length));
    const diagnosticOverflows = Math.max(
      a.diagnosticOverflows || 0,
      b.diagnosticOverflows || 0,
    );
    return {
      p95ByteDelta: percentile(0.95),
      p99ByteDelta: percentile(0.99),
      maxByteDelta: differences.at(-1) || 0,
      rmseByteDelta: rmse,
      changedChannelRatio: changed / Math.max(1, differences.length),
      diagnosticOverflows,
      passed: percentile(0.95) <= 1
        && percentile(0.99) <= 2
        && rmse <= 0.75
        && (differences.at(-1) || 0) <= 48
        && diagnosticOverflows === 0,
    };
  }

  worldAt(frame, x, y) {
    const byteOffset = y * frame.worldBytesPerRow + x * 16;
    const index = byteOffset >> 2;
    return [
      frame.worldPixels[index],
      frame.worldPixels[index + 1],
      frame.worldPixels[index + 2],
      frame.worldPixels[index + 3],
    ];
  }

  normalAt(frame, x, y) {
    const byteOffset = y * frame.normalBytesPerRow + x * 8;
    const index = byteOffset >> 1;
    const octX = halfToFloat(frame.normalPixels[index]);
    const octY = halfToFloat(frame.normalPixels[index + 1]);
    const normal = [octX, octY, 1 - Math.abs(octX) - Math.abs(octY)];
    if (normal[2] < 0) {
      const oldX = normal[0];
      const oldY = normal[1];
      normal[0] = (1 - Math.abs(oldY)) * (oldX >= 0 ? 1 : -1);
      normal[1] = (1 - Math.abs(oldX)) * (oldY >= 0 ? 1 : -1);
    }
    const length = Math.hypot(...normal);
    return length > 1e-6 ? normal.map((component) => component / length) : [0, 0, 0];
  }

  compareReprojectedFrames(a, b, {
    pixelStep = 2,
    searchRadius = 1,
    worldToleranceScale = 0.045,
  } = {}) {
    if (a.width !== b.width || a.height !== b.height) {
      return { passed: false, reason: "capture dimensions changed" };
    }
    const differences = [];
    let squared = 0;
    let matchedPixels = 0;
    let surfacePixels = 0;
    let projectedPixels = 0;
    // Half-float positions need a small tolerance, but a wide tolerance can
    // falsely pair opposite sides of thin, high-contrast geometry at
    // disocclusion boundaries. Keep the match well below a c0 cell.
    const maximumWorldDelta = Math.max(0.02, a.baseSpacing * worldToleranceScale);
    const maximumWorldDeltaSquared = maximumWorldDelta * maximumWorldDelta;
    // Decode the destination fields once. A radius-four correspondence search
    // otherwise converts the same half-float world/normal texels up to 81
    // times per source pixel, turning a strict full-frame audit into minutes
    // of avoidable CPU work without making it more rigorous.
    const destinationPixelCount = b.width * b.height;
    const destinationWorld = new Float32Array(destinationPixelCount * 4);
    const destinationNormal = new Float32Array(destinationPixelCount * 3);
    const destinationValid = new Uint8Array(destinationPixelCount);
    for (let destinationY = 0; destinationY < b.height; destinationY++) {
      for (let destinationX = 0; destinationX < b.width; destinationX++) {
        const destinationPixel = destinationY * b.width + destinationX;
        const world = this.worldAt(b, destinationX, destinationY);
        if (!(world[3] >= 0.5) || !world.every(Number.isFinite)) continue;
        const normal = this.normalAt(b, destinationX, destinationY);
        if (!normal.every(Number.isFinite)) continue;
        destinationWorld.set(world, destinationPixel * 4);
        destinationNormal.set(normal, destinationPixel * 3);
        destinationValid[destinationPixel] = 1;
      }
    }
    for (let y = 0; y < a.height; y += pixelStep) {
      for (let x = 0; x < a.width; x += pixelStep) {
        const world = this.worldAt(a, x, y);
        if (!(world[3] >= 0.5) || !world.every(Number.isFinite)) continue;
        const normal = this.normalAt(a, x, y);
        if (!normal.every(Number.isFinite)) continue;
        surfacePixels++;
        const clip = transformPoint(b.viewProjection, world);
        if (!(clip[3] > 1e-6)) continue;
        const projectedX = (clip[0] / clip[3] * 0.5 + 0.5) * b.width;
        const projectedY = (0.5 - clip[1] / clip[3] * 0.5) * b.height;
        const centerX = Math.round(projectedX - 0.5);
        const centerY = Math.round(projectedY - 0.5);
        if (centerX < 0 || centerY < 0 || centerX >= b.width || centerY >= b.height) continue;
        projectedPixels++;
        let bestX = -1;
        let bestY = -1;
        let bestDistance = Infinity;
        for (let offsetY = -searchRadius; offsetY <= searchRadius; offsetY++) {
          for (let offsetX = -searchRadius; offsetX <= searchRadius; offsetX++) {
            const candidateX = centerX + offsetX;
            const candidateY = centerY + offsetY;
            if (candidateX < 0 || candidateY < 0 || candidateX >= b.width || candidateY >= b.height) continue;
            const candidatePixel = candidateY * b.width + candidateX;
            if (!destinationValid[candidatePixel]) continue;
            const worldOffset = candidatePixel * 4;
            const normalOffset = candidatePixel * 3;
            const normalAgreement = normal[0] * destinationNormal[normalOffset]
              + normal[1] * destinationNormal[normalOffset + 1]
              + normal[2] * destinationNormal[normalOffset + 2];
            if (normalAgreement < 0.88) continue;
            const dx = destinationWorld[worldOffset] - world[0];
            const dy = destinationWorld[worldOffset + 1] - world[1];
            const dz = destinationWorld[worldOffset + 2] - world[2];
            const distance = dx * dx + dy * dy + dz * dz;
            if (distance < bestDistance) {
              bestDistance = distance;
              bestX = candidateX;
              bestY = candidateY;
            }
          }
        }
        if (bestDistance > maximumWorldDeltaSquared || bestX < 0) continue;
        matchedPixels++;
        const pixelA = y * a.bytesPerRow + x * 4;
        const pixelB = bestY * b.bytesPerRow + bestX * 4;
        for (let channel = 0; channel < 3; channel++) {
          const difference = Math.abs(a.pixels[pixelA + channel] - b.pixels[pixelB + channel]);
          differences.push(difference);
          squared += difference * difference;
        }
      }
    }
    differences.sort((left, right) => left - right);
    const percentile = (p) => differences.length
      ? differences[Math.min(differences.length - 1, Math.floor((differences.length - 1) * p))]
      : Infinity;
    const rmse = Math.sqrt(squared / Math.max(1, differences.length));
    // A handful of disocclusion-edge correspondences can be geometrically
    // ambiguous even after position/normal rejection. Keep the raw RMSE and
    // maximum in the report, but gate temporal shimmer on a 99.5%-trimmed RMSE.
    const trimmedCount = Math.max(1, Math.floor(differences.length * 0.995));
    let trimmedSquared = 0;
    for (let i = 0; i < trimmedCount; i++) trimmedSquared += differences[i] ** 2;
    const trimmedRmse = Math.sqrt(trimmedSquared / trimmedCount);
    const matchedPixelRatio = matchedPixels / Math.max(1, surfacePixels);
    const largeDeltaRatio = differences.filter((difference) => difference > 32).length
      / Math.max(1, differences.length);
    const diagnosticOverflows = Math.max(
      a.diagnosticOverflows || 0,
      b.diagnosticOverflows || 0,
    );
    return {
      matchedPixels,
      surfacePixels,
      projectedPixels,
      matchedPixelRatio,
      p95ByteDelta: percentile(0.95),
      p99ByteDelta: percentile(0.99),
      p999ByteDelta: percentile(0.999),
      maxByteDelta: differences.at(-1) ?? Infinity,
      rmseByteDelta: rmse,
      trimmedRmseByteDelta: trimmedRmse,
      largeDeltaRatio,
      diagnosticOverflows,
      passed: matchedPixelRatio >= 0.35
        && percentile(0.95) <= 3
        && percentile(0.99) <= 9
        && trimmedRmse <= 3
        && diagnosticOverflows === 0,
    };
  }

  async runViewDistanceInvarianceAudit({
    farScale = 1.05,
    nearScale = 0.98,
    warmup = 48,
  } = {}) {
    const saved = {
      camera: {
        ...this.camera,
        target: [...this.camera.target],
      },
      animateCamera: this.animateCamera,
      animateLights: this.animateLights,
      temporalStability: this.temporalStability,
      debugMode: this.debugMode,
      testTimeOverride: this.testTimeOverride,
    };
    try {
      this.animateCamera = false;
      this.animateLights = false;
      this.temporalStability = true;
      this.debugMode = 1;
      this.testTimeOverride = 0.7;
      const baseCamera = this.sceneIndex === 1
        ? {
          target: [7.0, 2.6, -0.5],
          distance: 20.5,
          azimuth: Math.PI,
          elevation: 0.025,
        }
        : {
          ...this.camera,
          target: [...this.camera.target],
        };
      const captureDistance = async (scale) => {
        this.camera = {
          ...baseCamera,
          target: [...baseCamera.target],
          distance: baseCamera.distance * scale,
        };
        this.resetProbeHistory();
        await this.waitFrames(warmup);
        return this.captureFinalFrame();
      };
      const far = await captureDistance(farScale);
      const near = await captureDistance(nearScale);
      const comparison = this.compareReprojectedFrames(far, near, {
        pixelStep: 1,
        searchRadius: 4,
        worldToleranceScale: 0.4,
      });
      // This deliberately spans a large dolly, not adjacent frames. It catches
      // view-sized sparse surfaces (the reported Sponza lion failure) that can
      // disappear from percentile-only short-motion tests.
      const passed = comparison.matchedPixelRatio >= 0.12
        && comparison.p95ByteDelta <= 12
        && comparison.p99ByteDelta <= 36
        && comparison.p999ByteDelta <= 96
        && comparison.largeDeltaRatio <= 0.02;
      return {
        scene: this.sceneIndex,
        farScale,
        nearScale,
        warmup,
        ...comparison,
        passed,
      };
    } finally {
      Object.assign(this, saved);
      this.camera = saved.camera;
      this.resetProbeHistory();
    }
  }

  async runLongTranslationCacheAudit({
    warmup = 48,
    motionFrames = 180,
    captureInterval = 6,
    translationPerFrame = 0.12,
    recoveryWarmup = 48,
    preservePose = false,
  } = {}) {
    const saved = {
      camera: {
        ...this.camera,
        target: [...this.camera.target],
      },
      animateCamera: this.animateCamera,
      animateLights: this.animateLights,
      temporalStability: this.temporalStability,
      debugMode: this.debugMode,
      testTimeOverride: this.testTimeOverride,
    };
    try {
      this.animateCamera = false;
      this.animateLights = false;
      this.temporalStability = true;
      this.debugMode = 1;
      this.testTimeOverride = 0.7;
      this.camera = this.sceneIndex === 1
        ? {
          target: [5.0, 1.75, -0.5],
          distance: 14.3,
          azimuth: Math.PI,
          elevation: 0.06,
        }
        : {
          ...this.camera,
          target: [...this.camera.target],
          elevation: Math.min(this.camera.elevation, 0.08),
        };
      this.resetProbeHistory();
      await this.waitFrames(warmup);

      const samples = [];
      let previous = await this.captureFinalFrame();
      for (let frame = 0; frame < motionFrames; frame++) {
        const forward = [-Math.cos(this.camera.azimuth), 0, -Math.sin(this.camera.azimuth)];
        this.camera.target = add3(this.camera.target, mul3(forward, translationPerFrame));
        await this.waitFrames(1);
        if ((frame + 1) % captureInterval !== 0 && frame + 1 !== motionFrames) continue;
        const current = await this.captureFinalFrame();
        const comparison = this.compareReprojectedFrames(previous, current, {
          pixelStep: 1,
          searchRadius: 4,
          worldToleranceScale: 0.4,
        });
        // These checkpoints are captureInterval translated frames apart, not
        // adjacent frames. Allow the real parallax/visibility change produced
        // by that displacement while still rejecting the large block
        // corruption this audit was built from. The existing continuous-motion
        // audit independently applies the stricter adjacent-frame thresholds.
        const checkpointPassed = comparison.matchedPixelRatio >= 0.35
          && comparison.p95ByteDelta <= 12
          && comparison.p99ByteDelta <= 28
          && comparison.p999ByteDelta <= 128
          && comparison.trimmedRmseByteDelta <= 6
          && comparison.largeDeltaRatio <= 0.01
          && comparison.diagnosticOverflows === 0;
        samples.push({
          frame: frame + 1,
          target: [...this.camera.target],
          probes: [...current.diagnostics.slice(0, 4)],
          overflows: current.diagnosticOverflows,
          matchedPixelRatio: comparison.matchedPixelRatio,
          p95ByteDelta: comparison.p95ByteDelta,
          p99ByteDelta: comparison.p99ByteDelta,
          p999ByteDelta: comparison.p999ByteDelta,
          trimmedRmseByteDelta: comparison.trimmedRmseByteDelta,
          largeDeltaRatio: comparison.largeDeltaRatio,
          passed: checkpointPassed,
        });
        previous = current;
      }

      await this.waitFrames(recoveryWarmup);
      const accumulated = await this.captureFinalFrame();
      this.resetProbeHistory();
      await this.waitFrames(recoveryWarmup);
      const recovered = await this.captureFinalFrame();
      const recoveryDifference = this.compareReprojectedFrames(accumulated, recovered, {
        pixelStep: 1,
        searchRadius: 0,
        worldToleranceScale: 0.1,
      });
      const maximumOverflows = Math.max(
        accumulated.diagnosticOverflows,
        recovered.diagnosticOverflows,
        ...samples.map((sample) => sample.overflows),
      );
      const failedMotionSamples = samples.filter((sample) => !sample.passed).length;
      const accumulatedProbeCounts = [...accumulated.diagnostics.slice(0, 4)];
      const recoveredProbeCounts = [...recovered.diagnostics.slice(0, 4)];
      const sparsePopulationMatched = accumulatedProbeCounts.every(
        (count, cascade) => count === recoveredProbeCounts[cascade],
      );
      const report = {
        scene: this.sceneIndex,
        warmup,
        motionFrames,
        captureInterval,
        translationPerFrame,
        recoveryWarmup,
        samples,
        accumulatedDiagnostics: [...accumulated.diagnostics],
        recoveredDiagnostics: [...recovered.diagnostics],
        accumulatedProbeCounts,
        recoveredProbeCounts,
        sparsePopulationMatched,
        maximumOverflows,
        failedMotionSamples,
        recoveryDifference,
      };
      report.passed = maximumOverflows === 0
        && failedMotionSamples === 0
        && sparsePopulationMatched
        && recoveryDifference.p95ByteDelta <= 4
        && recoveryDifference.p99ByteDelta <= 12
        && recoveryDifference.trimmedRmseByteDelta <= 4;
      return report;
    } finally {
      if (!preservePose) {
        Object.assign(this, saved);
        this.camera = saved.camera;
        this.resetProbeHistory();
      }
    }
  }

  async runFloorRoundTripAudit({
    warmup = 48,
    steps = 8,
    translationPerFrame = 0.06,
    recoveryWarmup = 48,
    movingLights = false,
    debugMode = 1,
    preservePose = false,
  } = {}) {
    const saved = {
      camera: { ...this.camera, target: [...this.camera.target] },
      animateCamera: this.animateCamera,
      animateLights: this.animateLights,
      temporalStability: this.temporalStability,
      debugMode: this.debugMode,
      testTimeOverride: this.testTimeOverride,
    };
    try {
      this.animateCamera = false;
      this.animateLights = movingLights;
      this.temporalStability = true;
      this.debugMode = debugMode;
      this.testTimeOverride = 0.7;
      this.camera = this.sceneIndex === 1
        ? {
            target: [5.0, 1.15, -0.5],
            distance: 13.8,
            azimuth: Math.PI,
            elevation: 0.015,
          }
        : {
            ...this.camera,
            target: [...this.camera.target],
            elevation: Math.min(this.camera.elevation, 0.04),
          };
      const originalCamera = { ...this.camera, target: [...this.camera.target] };
      this.resetProbeHistory();
      await this.waitFrames(warmup);
      const captures = [await this.captureFinalFrame()];
      const trajectory = [];
      const forward = [-Math.cos(this.camera.azimuth), 0, -Math.sin(this.camera.azimuth)];
      for (const direction of [1, -1]) {
        for (let step = 0; step < steps; step++) {
          this.camera.target = add3(
            this.camera.target,
            mul3(forward, translationPerFrame * direction),
          );
          if (movingLights) this.testTimeOverride += 1 / 60;
          await this.waitFrames(1);
          const current = await this.captureFinalFrame();
          const previous = captures.at(-1);
          const comparison = this.compareReprojectedFrames(previous, current, {
            pixelStep: 1,
            searchRadius: 4,
            worldToleranceScale: 0.25,
          });
          trajectory.push({
            leg: direction > 0 ? "forward" : "backward",
            step: step + 1,
            target: [...this.camera.target],
            ...comparison,
          });
          captures.push(current);
        }
      }
      const returned = captures.at(-1);
      this.camera = { ...originalCamera, target: [...originalCamera.target] };
      this.resetProbeHistory();
      await this.waitFrames(recoveryWarmup);
      const clean = await this.captureFinalFrame();
      const loopClosure = this.compareReprojectedFrames(returned, clean, {
        pixelStep: 1,
        searchRadius: 0,
        worldToleranceScale: 0.08,
      });
      const maximum = (field) => Math.max(...trajectory.map((sample) => sample[field] ?? Infinity));
      const minimum = (field) => Math.min(...trajectory.map((sample) => sample[field] ?? 0));
      const coverageExact = debugMode === 4
        ? maximum("maxByteDelta") === 0 && loopClosure.maxByteDelta === 0
        : null;
      return {
        scene: this.sceneIndex,
        movingLights,
        debugMode,
        preservePose,
        warmup,
        steps,
        translationPerFrame,
        matchedPixelRatioMin: minimum("matchedPixelRatio"),
        p95ByteDeltaMax: maximum("p95ByteDelta"),
        p99ByteDeltaMax: maximum("p99ByteDelta"),
        maxByteDeltaMax: maximum("maxByteDelta"),
        trimmedRmseByteDeltaMax: maximum("trimmedRmseByteDelta"),
        largeDeltaRatioMax: maximum("largeDeltaRatio"),
        loopClosure,
        coverageExact,
        passed: trajectory.length === steps * 2
          && minimum("matchedPixelRatio") >= 0.72
          && maximum("p95ByteDelta") <= 3
          && maximum("p99ByteDelta") <= 10
          && maximum("trimmedRmseByteDelta") <= 1.9
          && maximum("largeDeltaRatio") <= 0.001
          && trajectory.every((sample) => sample.diagnosticOverflows === 0)
          && (debugMode !== 4 || coverageExact)
          && loopClosure.p95ByteDelta <= 3
          && loopClosure.p99ByteDelta <= 8
          && loopClosure.trimmedRmseByteDelta <= 2,
        trajectory,
      };
    } finally {
      if (!preservePose) {
        Object.assign(this, saved);
        this.camera = saved.camera;
        this.resetProbeHistory();
      }
    }
  }

  async runDynamicRoundTripAudit({
    startTime = 0.8,
    movedTime = 3.15,
    warmup = 72,
    movedWarmup = 24,
    recoveryWarmup = 72,
  } = {}) {
    const saved = {
      camera: { ...this.camera, target: [...this.camera.target] },
      animateCamera: this.animateCamera,
      animateLights: this.animateLights,
      temporalStability: this.temporalStability,
      debugMode: this.debugMode,
      testTimeOverride: this.testTimeOverride,
    };
    try {
      this.animateCamera = false;
      this.animateLights = false;
      this.temporalStability = true;
      this.debugMode = 1;
      this.testTimeOverride = startTime;
      this.resetProbeHistory();
      await this.waitFrames(warmup);
      const start = await this.captureFinalFrame();
      this.testTimeOverride = movedTime;
      await this.waitFrames(movedWarmup);
      const moved = await this.captureFinalFrame();
      this.testTimeOverride = startTime;
      await this.waitFrames(recoveryWarmup);
      const returned = await this.captureFinalFrame();
      this.resetProbeHistory();
      await this.waitFrames(warmup);
      const clean = await this.captureFinalFrame();
      const movement = this.compareReprojectedFrames(start, moved, {
        pixelStep: 1,
        searchRadius: 0,
        worldToleranceScale: 0.08,
      });
      const closure = this.compareReprojectedFrames(returned, clean, {
        pixelStep: 1,
        searchRadius: 0,
        worldToleranceScale: 0.08,
      });
      return {
        startTime,
        movedTime,
        movement,
        closure,
        changed: movement.rmseByteDelta > 0.5,
        passed: movement.rmseByteDelta > 0.5
          && closure.matchedPixelRatio >= 0.98
          && closure.p95ByteDelta <= 3
          && closure.p99ByteDelta <= 9
          && closure.trimmedRmseByteDelta <= 1.75,
      };
    } finally {
      Object.assign(this, saved);
      this.camera = saved.camera;
      this.resetProbeHistory();
    }
  }

  async runDynamicEmitterResponseAudit({
    time = 2.0,
    warmup = 72,
  } = {}) {
    const savedEmissionScale = this.dynamicScene?.emissionScale ?? 1;
    const saved = {
      camera: { ...this.camera, target: [...this.camera.target] },
      animateCamera: this.animateCamera,
      animateLights: this.animateLights,
      temporalStability: this.temporalStability,
      debugMode: this.debugMode,
      testTimeOverride: this.testTimeOverride,
    };
    try {
      if (!this.dynamicScene || this.dynamicScene.emissiveInstanceCount < 1) {
        return { applicable: false, passed: false, reason: "No dynamic mesh lights." };
      }
      this.animateCamera = false;
      this.animateLights = false;
      this.temporalStability = true;
      this.debugMode = 1;
      this.testTimeOverride = time;

      this.dynamicScene.emissionScale = 0;
      this.resetProbeHistory();
      await this.waitFrames(warmup);
      const lightsOff = await this.captureFinalFrame();

      this.dynamicScene.emissionScale = 1;
      this.resetProbeHistory();
      await this.waitFrames(warmup);
      const lightsOn = await this.captureFinalFrame();
      const signal = this.compareFinalFrames(lightsOff, lightsOn);
      return {
        applicable: true,
        time,
        warmup,
        signal,
        // This is indirect-only at one fixed camera and geometry pose. A
        // measurable difference can therefore only come from transported
        // radiance emitted by dynamic mesh-light records and their TLAS.
        // Mesh lights are deliberately small and local, so a whole-frame p95
        // is expected to be zero. Require a spatially non-vacuous affected
        // region plus both RMS and peak energy instead of rewarding a source
        // that brightens every pixel by an imperceptible amount.
        passed: signal.rmseByteDelta >= 0.05
          && signal.changedChannelRatio >= 0.005
          && signal.maxByteDelta >= 2,
      };
    } finally {
      if (this.dynamicScene) this.dynamicScene.emissionScale = savedEmissionScale;
      Object.assign(this, saved);
      this.camera = saved.camera;
      this.resetProbeHistory();
    }
  }

  async runContinuousMotionAudit({
    frames = 24,
    warmup = 64,
    startTime = 0.8,
    timeStep = 0.05,
    movingLights = false,
  } = {}) {
    const saved = {
      animateCamera: this.animateCamera,
      animateLights: this.animateLights,
      temporalStability: this.temporalStability,
      debugMode: this.debugMode,
      testTimeOverride: this.testTimeOverride,
    };
    try {
      this.animateCamera = true;
      this.animateLights = movingLights;
      this.temporalStability = true;
      this.debugMode = 1;
      this.testTimeOverride = startTime;
      this.resetProbeHistory();
      await this.waitFrames(warmup);
      const captures = [];
      for (let frame = 0; frame < frames; frame++) {
        this.testTimeOverride = startTime + frame * timeStep;
        captures.push(await this.captureFinalFrame());
      }
      const comparisons = [];
      for (let frame = 1; frame < captures.length; frame++) {
        const comparison = this.compareReprojectedFrames(captures[frame - 1], captures[frame]);
        comparisons.push({
          ...comparison,
          motionPassed: comparison.passed,
        });
      }
      const maximum = (field) => Math.max(...comparisons.map((comparison) => comparison[field] ?? Infinity));
      const minimum = (field) => Math.min(...comparisons.map((comparison) => comparison[field] ?? 0));
      return {
        movingLights,
        frames,
        comparisons: comparisons.length,
        matchedPixelRatioMin: minimum("matchedPixelRatio"),
        p95ByteDeltaMax: maximum("p95ByteDelta"),
        p99ByteDeltaMax: maximum("p99ByteDelta"),
        maxByteDelta: maximum("maxByteDelta"),
        rmseByteDeltaMax: maximum("rmseByteDelta"),
        trimmedRmseByteDeltaMax: maximum("trimmedRmseByteDelta"),
        passed: comparisons.length === frames - 1
          && comparisons.every((comparison) => comparison.motionPassed),
        details: comparisons,
      };
    } finally {
      Object.assign(this, saved);
      this.resetProbeHistory();
    }
  }

  async runMovingLightResponseAudit({
    timeA = 0.7,
    timeB = 7.7,
    warmup = 72,
    adaptationFrames = 56,
  } = {}) {
    const saved = {
      animateCamera: this.animateCamera,
      animateLights: this.animateLights,
      temporalStability: this.temporalStability,
      debugMode: this.debugMode,
      testTimeOverride: this.testTimeOverride,
    };
    try {
      this.animateCamera = false;
      this.animateLights = true;
      this.temporalStability = true;
      this.debugMode = 1;
      const convergedAt = async (time) => {
        this.testTimeOverride = time;
        this.resetProbeHistory();
        await this.waitFrames(warmup);
        return this.captureFinalFrame();
      };
      const targetA = await convergedAt(timeA);
      const targetB = await convergedAt(timeB);
      this.testTimeOverride = timeA;
      this.resetProbeHistory();
      await this.waitFrames(warmup);
      this.testTimeOverride = timeB;
      const firstStep = await this.captureFinalFrame();
      await this.waitFrames(Math.max(1, adaptationFrames - 1));
      const adapted = await this.captureFinalFrame();
      const signal = this.compareFinalFrames(targetA, targetB);
      const initialError = this.compareFinalFrames(firstStep, targetB);
      const adaptedError = this.compareFinalFrames(adapted, targetB);
      const responseRatio = adaptedError.rmseByteDelta
        / Math.max(0.25, initialError.rmseByteDelta);
      const signalPresent = signal.rmseByteDelta >= 0.5
        && signal.changedChannelRatio >= 0.005;
      return {
        scene: this.sceneIndex,
        timeA,
        timeB,
        warmup,
        adaptationFrames,
        signal,
        initialError,
        adaptedError,
        responseRatio,
        passed: signalPresent
          && responseRatio <= 0.72
          && adaptedError.p95ByteDelta <= Math.max(5, initialError.p95ByteDelta * 0.72)
          && adaptedError.rmseByteDelta <= initialError.rmseByteDelta - 0.25,
      };
    } finally {
      Object.assign(this, saved);
      this.resetProbeHistory();
    }
  }

  async runFinalFrameRepeatabilityAudit({
    poses = 6,
    warmup = 64,
    holdFrames = 2,
  } = {}) {
    const saved = {
      animateCamera: this.animateCamera,
      animateLights: this.animateLights,
      temporalStability: this.temporalStability,
      testTimeOverride: this.testTimeOverride,
    };
    try {
      this.animateCamera = true;
      this.animateLights = false;
      this.temporalStability = true;
      const times = Array.from({ length: poses }, (_, i) => 1.25 + i * 0.55);
      const captureTrajectory = async () => {
        this.testTimeOverride = 0.7;
        this.resetProbeHistory();
        await this.waitFrames(warmup);
        const captures = [];
        for (const time of times) {
          this.testTimeOverride = time;
          await this.waitFrames(holdFrames);
          captures.push(await this.captureFinalFrame());
        }
        return captures;
      };
      const first = await captureTrajectory();
      const second = await captureTrajectory();
      const comparisons = first.map((frame, i) => this.compareFinalFrames(frame, second[i]));
      const maximum = (field) => Math.max(...comparisons.map((result) => result[field] ?? Infinity));
      return {
        poses,
        p95ByteDeltaMax: maximum("p95ByteDelta"),
        p99ByteDeltaMax: maximum("p99ByteDelta"),
        maxByteDelta: maximum("maxByteDelta"),
        rmseByteDeltaMax: maximum("rmseByteDelta"),
        changedChannelRatioMax: maximum("changedChannelRatio"),
        passed: comparisons.every((result) => result.passed),
        comparisons,
      };
    } finally {
      Object.assign(this, saved);
      this.resetProbeHistory();
    }
  }

  async runShadowMapAudit({ width = 80, height = 45, warmup = 8, time = 0.7 } = {}) {
    const saved = {
      animateCamera: this.animateCamera,
      animateLights: this.animateLights,
      temporalStability: this.temporalStability,
      testTimeOverride: this.testTimeOverride,
    };
    let auditBuffer;
    let auditPassBuffer;
    try {
      this.animateCamera = false;
      this.animateLights = true;
      this.temporalStability = true;
      this.testTimeOverride = time;
      await this.waitFrames(warmup);
      await this.device.queue.onSubmittedWorkDone();

      const byteLength = width * height * 6 * 4;
      auditBuffer = createBuffer(
        this.device,
        "raster shadow maps versus BVH visibility",
        byteLength,
        GPU.STORAGE | GPU.COPY_SRC | GPU.COPY_DST,
      );
      auditPassBuffer = createBuffer(
        this.device,
        "shadow map audit parameters",
        32,
        GPU.UNIFORM | GPU.COPY_DST,
        new Uint32Array([0, 0, width, height, 0, 0, 0, 0]),
      );
      const bindGroup = this.device.createBindGroup({
        label: "shadow-map correctness audit bind group",
        layout: this.computeLayout,
        entries: [
          { binding: 0, resource: { buffer: this.frameBuffer } },
          { binding: 1, resource: this.worldTexture.createView() },
          { binding: 2, resource: this.normalTexture.createView() },
          { binding: 3, resource: { buffer: this.hashBuffer } },
          { binding: 4, resource: { buffer: this.stateBuffer } },
          { binding: 5, resource: { buffer: this.probeMetaBuffer } },
          { binding: 6, resource: { buffer: auditBuffer } },
          { binding: 7, resource: { buffer: this.coneBuffer } },
          { binding: 8, resource: { buffer: this.irradianceBuffer } },
          { binding: 9, resource: { buffer: this.bvhNodeBuffer } },
          { binding: 10, resource: { buffer: this.triangleBuffer } },
          { binding: 11, resource: { buffer: auditPassBuffer } },
          { binding: 12, resource: this.materialAtlasView },
          { binding: 13, resource: this.materialSampler },
          { binding: 14, resource: this.irradianceAtlasWrite.createView() },
          { binding: 15, resource: this.irradianceAtlas.createView() },
          { binding: 16, resource: this.irradianceSampler },
          { binding: 17, resource: this.pointShadowArrayView },
          { binding: 18, resource: this.pointShadowSampler },
          { binding: 19, resource: this.sunShadowArrayView },
          { binding: 20, resource: this.shadowSampler },
          { binding: 21, resource: { buffer: this.sunShadowDataBuffer } },
        ],
      });
      const encoder = this.device.createCommandEncoder({ label: "shadow-map correctness audit" });
      encoder.clearBuffer(auditBuffer);
      const pass = encoder.beginComputePass({ label: "depth maps versus exact BVH visibility" });
      pass.setPipeline(this.computePipelines.validateShadowMaps);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
      pass.end();
      this.device.queue.submit([encoder.finish()]);
      const fields = new Uint32Array(await this.readGpuBuffer(
        auditBuffer,
        byteLength,
        "shadow map audit readback",
      ));

      const summarize = (referenceOffset, rasterOffset, validOffset) => {
        const absolute = [];
        let sum = 0;
        let referenceSum = 0;
        let rasterSum = 0;
        let classificationMismatches = 0;
        let severeMismatches = 0;
        let falseShadowPixels = 0;
        let lightLeakPixels = 0;
        const faceSamples = Array(6).fill(0);
        const faceMismatches = Array(6).fill(0);
        for (let pixel = 0; pixel < width * height; pixel++) {
          const base = pixel * 6;
          if (!fields[base + validOffset]) continue;
          const face = validOffset === 4 ? fields[base + validOffset] - 1 : -1;
          const reference = fields[base + referenceOffset] / 65535;
          const raster = fields[base + rasterOffset] / 65535;
          const error = Math.abs(reference - raster);
          absolute.push(error);
          sum += error;
          referenceSum += reference;
          rasterSum += raster;
          if ((reference >= 0.5) !== (raster >= 0.5)) {
            classificationMismatches++;
            if (face >= 0 && face < 6) faceMismatches[face]++;
            if (reference >= 0.5) falseShadowPixels++;
            else lightLeakPixels++;
          }
          if (face >= 0 && face < 6) faceSamples[face]++;
          if (error > 0.75) severeMismatches++;
        }
        absolute.sort((a, b) => a - b);
        const percentile = (p) => absolute.length
          ? absolute[Math.min(absolute.length - 1, Math.floor((absolute.length - 1) * p))]
          : Infinity;
        return {
          samples: absolute.length,
          meanAbsolute: sum / Math.max(1, absolute.length),
          meanReferenceVisibility: referenceSum / Math.max(1, absolute.length),
          meanRasterVisibility: rasterSum / Math.max(1, absolute.length),
          p95Absolute: percentile(0.95),
          p99Absolute: percentile(0.99),
          classificationMismatchRatio: classificationMismatches / Math.max(1, absolute.length),
          falseShadowRatio: falseShadowPixels / Math.max(1, absolute.length),
          lightLeakRatio: lightLeakPixels / Math.max(1, absolute.length),
          perFace: faceSamples.map((samples, face) => ({
            face,
            samples,
            mismatchRatio: faceMismatches[face] / Math.max(1, samples),
          })),
          severeMismatchRatio: severeMismatches / Math.max(1, absolute.length),
        };
      };
      const point = summarize(0, 1, 4);
      const sun = summarize(2, 3, 5);
      const lightPassed = (metric, minimumSamples) =>
        metric.samples >= minimumSamples
        && metric.meanAbsolute <= 0.08
        && metric.p95Absolute <= 0.35
        && metric.classificationMismatchRatio <= 0.06
        && metric.severeMismatchRatio <= 0.035;
      // The Cornell audit camera sees all six point-light cube faces. Require
      // explicit face coverage there so an accidental face mapping omission
      // cannot hide behind a good aggregate error from the other five faces.
      const pointFaceCoverageRequired = this.sceneIndex === 10;
      const pointFaceCoveragePassed = !pointFaceCoverageRequired
        || point.perFace.every((face) => face.samples >= 4);
      const pointApplicable = this.pointShadowsEnabled;
      const pointPassed = pointApplicable
        ? (lightPassed(point, 32) && pointFaceCoveragePassed)
        : null;
      const sunPassed = lightPassed(sun, 32);
      return {
        resolution: [width, height],
        scene: this.sceneIndex,
        time,
        point,
        sun,
        pointFaceCoverageRequired,
        pointFaceCoveragePassed,
        pointApplicable,
        pointPassed,
        sunPassed,
        passed: (pointPassed ?? true) && sunPassed,
      };
    } finally {
      auditBuffer?.destroy();
      auditPassBuffer?.destroy();
      Object.assign(this, saved);
    }
  }

  async runSunShadowSweepAudit({
    times = [0.7, 7.2, 13.7, 20.2, 26.7, 33.2, 39.7, 46.2],
    width = 80,
    height = 45,
    warmup = 3,
  } = {}) {
    const samples = [];
    for (const time of times) {
      const result = await this.runShadowMapAudit({ width, height, warmup, time });
      samples.push({ time, ...result.sun, passed: result.sunPassed });
    }
    const maximum = (field) => Math.max(...samples.map((sample) => sample[field] ?? Infinity));
    return {
      scene: this.sceneIndex,
      resolution: [width, height],
      times,
      classificationMismatchRatioMax: maximum("classificationMismatchRatio"),
      severeMismatchRatioMax: maximum("severeMismatchRatio"),
      falseShadowRatioMax: maximum("falseShadowRatio"),
      lightLeakRatioMax: maximum("lightLeakRatio"),
      meanAbsoluteMax: maximum("meanAbsolute"),
      passed: samples.every((sample) => sample.passed),
      samples,
    };
  }

  summarizePlanarContinuity(frame) {
    const deltas = [];
    const strongestEdges = [];
    let planarTriplets = 0;
    let abruptSteps = 0;
    let maximumStepContrast = 0;
    let severeEdges = 0;
    let samples = 0;
    const luminance = (x, y) => {
      const offset = y * frame.bytesPerRow + x * 4;
      return frame.pixels[offset] * 0.2126
        + frame.pixels[offset + 1] * 0.7152
        + frame.pixels[offset + 2] * 0.0722;
    };
    const emissionEncoded = (x, y) => {
      if (this.worldAt(frame, x, y)[3] > 1.25) return 1;
      const index = (y * frame.normalBytesPerRow + x * 8) >> 1;
      return Math.max(
        halfToFloat(frame.normalPixels[index + 2]),
        halfToFloat(frame.normalPixels[index + 3]),
      );
    };
    const compare = (x0, y0, x1, y1) => {
      const worldA = this.worldAt(frame, x0, y0);
      const worldB = this.worldAt(frame, x1, y1);
      if (!(worldA[3] >= 0.5) || !(worldB[3] >= 0.5)) return;
      const normalA = this.normalAt(frame, x0, y0);
      const normalB = this.normalAt(frame, x1, y1);
      if (dot3(normalA, normalB) < 0.995) return;
      // The emitter silhouette is an intentional material boundary, not a
      // probe-grid discontinuity. Green/blue HDR emission are packed in the
      // two spare normal channels and identify it without scene coordinates.
      if (emissionEncoded(x0, y0) > 1e-5 || emissionEncoded(x1, y1) > 1e-5) return;
      const distance = Math.hypot(
        worldA[0] - worldB[0],
        worldA[1] - worldB[1],
        worldA[2] - worldB[2],
      );
      // Reject silhouette/disocclusion pairs. Neighboring samples on one
      // planar surface are much closer than a fifth of a base cell.
      if (distance > frame.baseSpacing * 0.2) return;
      const delta = Math.abs(luminance(x0, y0) - luminance(x1, y1));
      deltas.push(delta);
      if (delta > 32) {
        severeEdges++;
        strongestEdges.push({
          a: [x0, y0],
          b: [x1, y1],
          delta,
          luminance: [luminance(x0, y0), luminance(x1, y1)],
          worldA: worldA.slice(0, 3),
          worldB: worldB.slice(0, 3),
          normalA,
          normalB,
        });
      }
      samples++;
    };
    for (let y = 0; y < frame.height; y++) {
      for (let x = 0; x < frame.width; x++) {
        if (x + 1 < frame.width) compare(x, y, x + 1, y);
        if (y + 1 < frame.height) compare(x, y, x, y + 1);
      }
    }
    const planarSample = (x, y) => {
      const world = this.worldAt(frame, x, y);
      if (!(world[3] >= 0.5)) return null;
      if (emissionEncoded(x, y) > 1e-5) return null;
      return { world, normal: this.normalAt(frame, x, y), luminance: luminance(x, y) };
    };
    const compareTriplet = (x0, y0, x1, y1, x2, y2) => {
      const a = planarSample(x0, y0);
      const b = planarSample(x1, y1);
      const c = planarSample(x2, y2);
      if (!a || !b || !c) return;
      if (dot3(a.normal, b.normal) < 0.995 || dot3(b.normal, c.normal) < 0.995) return;
      const adjacentDistance = (left, right) => Math.hypot(
        left.world[0] - right.world[0],
        left.world[1] - right.world[1],
        left.world[2] - right.world[2],
      );
      if (adjacentDistance(a, b) > frame.baseSpacing * 0.2
        || adjacentDistance(b, c) > frame.baseSpacing * 0.2) return;
      const leftSlope = Math.abs(b.luminance - a.luminance);
      const rightSlope = Math.abs(c.luminance - b.luminance);
      const contrast = Math.max(leftSlope, rightSlope);
      const continuation = Math.min(leftSlope, rightSlope);
      planarTriplets++;
      maximumStepContrast = Math.max(maximumStepContrast, contrast);
      // Angular-bin aliasing is a plateau separated by a single large jump.
      // A true close area light can have a much larger gradient, but it
      // changes continuously across successive samples instead of flattening
      // immediately on one side of the edge.
      if (contrast > 24 && continuation < contrast * 0.1) abruptSteps++;
    };
    for (let y = 1; y + 1 < frame.height; y++) {
      for (let x = 1; x + 1 < frame.width; x++) {
        compareTriplet(x - 1, y, x, y, x + 1, y);
        compareTriplet(x, y - 1, x, y, x, y + 1);
      }
    }
    deltas.sort((a, b) => a - b);
    strongestEdges.sort((a, b) => b.delta - a.delta);
    const percentile = (p) => deltas.length
      ? deltas[Math.min(deltas.length - 1, Math.floor((deltas.length - 1) * p))]
      : Infinity;
    return {
      samples,
      p95ByteDelta: percentile(0.95),
      p99ByteDelta: percentile(0.99),
      p999ByteDelta: percentile(0.999),
      maximumByteDelta: deltas.at(-1) ?? Infinity,
      severeEdgeRatio: severeEdges / Math.max(1, samples),
      planarTriplets,
      abruptStepRatio: abruptSteps / Math.max(1, planarTriplets),
      maximumStepContrast,
      strongestEdges: strongestEdges.slice(0, 8),
    };
  }

  async runCornellArtifactAudit({
    warmup = 128,
    referenceSamples = 512,
    preservePose = false,
  } = {}) {
    const saved = {
      sceneIndex: this.sceneIndex,
      camera: { ...this.camera, target: [...this.camera.target] },
      animateCamera: this.animateCamera,
      animateLights: this.animateLights,
      temporalStability: this.temporalStability,
      debugMode: this.debugMode,
      testTimeOverride: this.testTimeOverride,
    };
    try {
      if (this.sceneIndex !== 10) await this.loadScene(10);
      this.animateCamera = false;
      this.animateLights = false;
      this.temporalStability = true;
      // Audit the reconstructed irradiance itself. The regular indirect-only
      // view multiplies by albedo, so a real material boundary on one plane
      // can otherwise be misclassified as a probe-grid discontinuity.
      this.debugMode = 6;
      this.testTimeOverride = 0.7;
      const results = [];
      for (const [name, pose] of Object.entries(CORNELL_ARTIFACT_POSES)) {
        this.setCameraPose(pose.position, pose.target);
        this.resetProbeHistory();
        await this.waitFrames(warmup);
        const baseline = await this.captureFinalFrame();
        const continuity = this.summarizePlanarContinuity(baseline);

        const trajectory = [];
        let previous = baseline;
        const step = [0.022, 0.006, -0.014];
        for (const multiplier of [1, 2, 1, 0]) {
          this.setCameraPose(
            add3(pose.position, mul3(step, multiplier)),
            add3(pose.target, mul3(step, multiplier)),
          );
          await this.waitFrames(1);
          const current = await this.captureFinalFrame();
          trajectory.push(this.compareReprojectedFrames(previous, current, {
            pixelStep: 1,
            searchRadius: 4,
            worldToleranceScale: 0.2,
          }));
          previous = current;
        }
        this.setCameraPose(pose.position, pose.target);
        this.resetProbeHistory();
        await this.waitFrames(warmup);
        const clean = await this.captureFinalFrame();
        const loopClosure = this.compareFinalFrames(previous, clean);
        const reference = await this.runPathTracedReferenceAudit({
          width: 80,
          height: 45,
          samples: referenceSamples,
          warmup,
        });
        const maximum = (field) => Math.max(
          ...trajectory.map((sample) => sample[field] ?? Infinity),
        );
        const minimum = (field) => Math.min(
          ...trajectory.map((sample) => sample[field] ?? 0),
        );
        const metrics = this.metricsSnapshot();
        const result = {
          name,
          position: [...pose.position],
          target: [...pose.target],
          continuity,
          motion: {
            samples: trajectory.length,
            matchedPixelRatioMin: minimum("matchedPixelRatio"),
            p95ByteDeltaMax: maximum("p95ByteDelta"),
            p99ByteDeltaMax: maximum("p99ByteDelta"),
            p999ByteDeltaMax: maximum("p999ByteDelta"),
            trimmedRmseByteDeltaMax: maximum("trimmedRmseByteDelta"),
            largeDeltaRatioMax: maximum("largeDeltaRatio"),
          },
          loopClosure,
          reference,
          metrics,
        };
        result.passed = continuity.samples >= 10_000
          && continuity.p99ByteDelta <= 4
          && continuity.p999ByteDelta <= 32
          && continuity.maximumByteDelta <= 96
          && continuity.severeEdgeRatio <= 0.001
          && result.motion.matchedPixelRatioMin >= 0.7
          && result.motion.p95ByteDeltaMax <= 3
          && result.motion.p99ByteDeltaMax <= 10
          && result.motion.trimmedRmseByteDeltaMax <= 2
          && result.motion.largeDeltaRatioMax <= 0.001
          && loopClosure.p95ByteDelta <= 2
          && loopClosure.p99ByteDelta <= 6
          && reference.activePixels >= 64
          && reference.p95Absolute <= 0.12
          && reference.p99Absolute <= 0.30
          && reference.severeUnderlitRatio <= 0.02
          && reference.severeOverlitRatio <= 0.04
          && metrics.gpuMs != null
          && metrics.gpuMs <= 16.6
          && metrics.overflows === 0
          && !metrics.gpuError;
        results.push(result);
      }
      return {
        scene: 10,
        name: "Cornell reported-artifact regressions",
        universalGI: true,
        warmup,
        referenceSamples,
        passed: results.every((result) => result.passed),
        poses: results,
      };
    } finally {
      if (!preservePose) {
        if (this.sceneIndex !== saved.sceneIndex) await this.loadScene(saved.sceneIndex);
        Object.assign(this, saved);
        this.camera = saved.camera;
        this.resetProbeHistory();
      }
    }
  }

  async runDoorZoomContinuityAudit({ warmup = 64, settleFrames = 1, preservePose = false } = {}) {
    const saved = {
      sceneIndex: this.sceneIndex,
      camera: { ...this.camera, target: [...this.camera.target] },
      animateCamera: this.animateCamera,
      animateLights: this.animateLights,
      temporalStability: this.temporalStability,
      debugMode: this.debugMode,
      testTimeOverride: this.testTimeOverride,
    };
    const target = [0, 2.2, 3.5];
    const distances = [8.6, 8.3, 8.0, 7.7, 7.4, 7.1, 6.8, 6.5];
    const luminanceStatistics = (frame) => {
      const values = [];
      let maximum = 0;
      for (let y = 0; y < frame.height; y++) {
        const pixelRow = y * frame.bytesPerRow;
        const worldRow = y * (frame.worldBytesPerRow / 4);
        for (let x = 0; x < frame.width; x++) {
          if (frame.worldPixels[worldRow + x * 4 + 3] < 0.5) continue;
          const pixel = pixelRow + x * 4;
          const value = frame.pixels[pixel] * 0.2126
            + frame.pixels[pixel + 1] * 0.7152
            + frame.pixels[pixel + 2] * 0.0722;
          values.push(value);
          maximum = Math.max(maximum, value);
        }
      }
      values.sort((a, b) => a - b);
      const percentile = (p) => values[
        Math.min(values.length - 1, Math.floor((values.length - 1) * p))
      ] || 0;
      return {
        surfacePixels: values.length,
        p50LuminanceByte: percentile(0.5),
        p95LuminanceByte: percentile(0.95),
        p99LuminanceByte: percentile(0.99),
        maximumLuminanceByte: maximum,
      };
    };
    try {
      if (this.sceneIndex !== 8) await this.loadScene(8);
      this.animateCamera = false;
      this.animateLights = false;
      this.temporalStability = true;
      this.debugMode = 1;
      this.testTimeOverride = 4.0;
      this.setCameraPose([0, 2.7, distances[0]], target);
      this.resetProbeHistory();
      await this.waitFrames(warmup);
      let previous = await this.captureFinalFrame();
      const samples = [{
        cameraZ: distances[0],
        environmentAccess: previous.diagnostics[8] !== 0,
        luminance: luminanceStatistics(previous),
      }];
      for (const cameraZ of distances.slice(1)) {
        this.setCameraPose([0, 2.7, cameraZ], target);
        await this.waitFrames(settleFrames);
        const current = await this.captureFinalFrame();
        samples.push({
          cameraZ,
          environmentAccess: current.diagnostics[8] !== 0,
          luminance: luminanceStatistics(current),
          transition: this.compareReprojectedFrames(previous, current, {
            pixelStep: 1,
            searchRadius: 5,
            worldToleranceScale: 0.3,
          }),
        });
        previous = current;
      }

      // The same universal classifier must make the complementary decision
      // for the closed topology. No scene-specific radiance threshold or
      // lighting override participates in rendering this frame.
      this.testTimeOverride = 0.7;
      this.setCameraPose([0, 2.7, 3.5], [0, 2.2, 0]);
      this.resetProbeHistory();
      await this.waitFrames(warmup);
      const closed = await this.captureFinalFrame();
      const closedDoor = {
        environmentAccess: closed.diagnostics[8] !== 0,
        luminance: luminanceStatistics(closed),
        overflows: closed.diagnosticOverflows,
      };
      const transitions = samples.slice(1).map((sample) => sample.transition);
      const report = {
        scene: 8,
        pose: "open doorway dolly through the former viewport-coverage boundary",
        warmup,
        settleFrames,
        samples,
        closedDoor,
        metrics: this.metricsSnapshot(),
      };
      report.passed = samples.every((sample) => sample.environmentAccess)
        && samples.every((sample) => sample.luminance.p99LuminanceByte >= 4)
        && Math.min(...transitions.map((sample) => sample.matchedPixelRatio)) >= 0.45
        && Math.max(...transitions.map((sample) => sample.p95ByteDelta)) <= 14
        && Math.max(...transitions.map((sample) => sample.p99ByteDelta)) <= 42
        && Math.max(...transitions.map((sample) => sample.largeDeltaRatio)) <= 0.02
        && !closedDoor.environmentAccess
        && closedDoor.luminance.maximumLuminanceByte <= 1
        && closedDoor.overflows === 0
        && !report.metrics.gpuError;
      return report;
    } finally {
      if (!preservePose) {
        if (this.sceneIndex !== saved.sceneIndex) await this.loadScene(saved.sceneIndex);
        Object.assign(this, saved);
        this.camera = saved.camera;
        this.resetProbeHistory();
      }
    }
  }

  async runEnclosureLeakAudit({ warmup = 128, samples = 512, preservePose = false } = {}) {
    const saved = {
      sceneIndex: this.sceneIndex,
      camera: { ...this.camera, target: [...this.camera.target] },
      animateCamera: this.animateCamera,
      animateLights: this.animateLights,
      temporalStability: this.temporalStability,
      debugMode: this.debugMode,
      testTimeOverride: this.testTimeOverride,
    };
    const insidePosition = [-2.2, 1.25, 0.5];
    const insideTarget = [-1.3, 1.72, -0.42];
    try {
      if (this.sceneIndex !== 0) await this.loadScene(0);
      this.animateCamera = false;
      this.animateLights = false;
      this.temporalStability = true;
      this.debugMode = 1;
      this.testTimeOverride = 0.7;
      this.setCameraPose(insidePosition, insideTarget);
      this.resetProbeHistory();
      const reference = await this.runPathTracedReferenceAudit({
        width: 80,
        height: 45,
        samples,
        warmup,
      });

      // Move the camera and target together inside the closed box, then close
      // the loop. World-space probe identities must remain stable and the
      // returned image must match a clean reconstruction at the same pose.
      this.setCameraPose(insidePosition, insideTarget);
      this.resetProbeHistory();
      await this.waitFrames(warmup);
      const origin = await this.captureFinalFrame();
      const luminanceBytes = [];
      let brightPixels = 0;
      let severePixels = 0;
      let maximumLuminance = 0;
      let maximumPixel = [0, 0];
      const severeSamples = [];
      for (let y = 0; y < origin.height; y++) {
        const row = y * origin.bytesPerRow;
        for (let x = 0; x < origin.width; x++) {
          const pixel = row + x * 4;
          const luminance = origin.pixels[pixel] * 0.2126
            + origin.pixels[pixel + 1] * 0.7152
            + origin.pixels[pixel + 2] * 0.0722;
          luminanceBytes.push(luminance);
          if (luminance > 12) brightPixels++;
          if (luminance > 32) {
            severePixels++;
            if (severeSamples.length < 16) severeSamples.push([x, y, luminance]);
          }
          if (luminance > maximumLuminance) {
            maximumLuminance = luminance;
            maximumPixel = [x, y];
          }
        }
      }
      luminanceBytes.sort((a, b) => a - b);
      const displayPercentile = (p) => luminanceBytes[
        Math.min(luminanceBytes.length - 1, Math.floor((luminanceBytes.length - 1) * p))
      ] || 0;
      const displayLeak = {
        p95LuminanceByte: displayPercentile(0.95),
        p99LuminanceByte: displayPercentile(0.99),
        p999LuminanceByte: displayPercentile(0.999),
        maximumLuminanceByte: maximumLuminance,
        maximumPixel,
        maximumWorld: this.worldAt(origin, ...maximumPixel),
        maximumNormal: this.normalAt(origin, ...maximumPixel),
        severeSamples,
        brightPixelRatio: brightPixels / Math.max(1, origin.width * origin.height),
        severePixelRatio: severePixels / Math.max(1, origin.width * origin.height),
      };
      const trajectory = [];
      const step = [0.045, 0.018, -0.035];
      let previous = origin;
      for (const multiplier of [1, 2, 3, 2, 1, 0]) {
        const position = add3(insidePosition, mul3(step, multiplier));
        const target = add3(insideTarget, mul3(step, multiplier));
        this.setCameraPose(position, target);
        await this.waitFrames(1);
        const current = await this.captureFinalFrame();
        trajectory.push(this.compareReprojectedFrames(previous, current, {
          pixelStep: 1,
          searchRadius: 4,
          worldToleranceScale: 0.2,
        }));
        previous = current;
      }
      this.setCameraPose(insidePosition, insideTarget);
      this.resetProbeHistory();
      await this.waitFrames(warmup);
      const clean = await this.captureFinalFrame();
      const loopClosure = this.compareFinalFrames(previous, clean);
      const maximum = (field) => Math.max(...trajectory.map((sample) => sample[field] ?? Infinity));
      const minimum = (field) => Math.min(...trajectory.map((sample) => sample[field] ?? 0));
      const metrics = this.metricsSnapshot();
      const report = {
        scene: 0,
        pose: "inside closed white box",
        reference,
        trajectory: {
          samples: trajectory.length,
          matchedPixelRatioMin: minimum("matchedPixelRatio"),
          p95ByteDeltaMax: maximum("p95ByteDelta"),
          p99ByteDeltaMax: maximum("p99ByteDelta"),
          largeDeltaRatioMax: maximum("largeDeltaRatio"),
        },
        loopClosure,
        displayLeak,
        metrics,
      };
      report.passed = reference.activePixels >= 64
        && report.trajectory.matchedPixelRatioMin >= 0.68
        && report.trajectory.p95ByteDeltaMax <= 4
        && report.trajectory.p99ByteDeltaMax <= 16
        && report.trajectory.largeDeltaRatioMax <= 0.005
        && loopClosure.p95ByteDelta <= 2
        && loopClosure.p99ByteDelta <= 6
        && displayLeak.p99LuminanceByte <= 6
        && displayLeak.maximumLuminanceByte <= 1
        && displayLeak.severePixelRatio === 0
        && metrics.overflows === 0
        && !metrics.gpuError;
      return report;
    } finally {
      if (!preservePose) {
        if (this.sceneIndex !== saved.sceneIndex) await this.loadScene(saved.sceneIndex);
        Object.assign(this, saved);
        this.camera = saved.camera;
        this.resetProbeHistory();
      }
    }
  }

  renderReferenceComparison(fields, width, height) {
    const canvas = $("reference-comparison");
    const container = $("reference-visual");
    if (!canvas || !container) return;
    canvas.width = width * 3;
    canvas.height = height;
    const context = canvas.getContext("2d");
    const image = context.createImageData(canvas.width, height);
    const aces = (value) => clamp(
      (value * (2.51 * value + 0.03)) / (value * (2.43 * value + 0.59) + 0.14),
      0,
      1,
    );
    const display = (value) => Math.round(Math.pow(aces(value * 1.35), 1 / 2.2) * 255);
    for (let pixel = 0; pixel < width * height; pixel++) {
      const source = pixel * 8;
      const x = pixel % width;
      const y = Math.floor(pixel / width);
      const valid = fields[source + 6] && fields[source + 7];
      const reference = valid
        ? [0, 1, 2].map((channel) => fields[source + channel] / (fields[source + 7] * 65536))
        : [0, 0, 0];
      const current = valid
        ? [3, 4, 5].map((channel) => fields[source + channel] / 65536)
        : [0, 0, 0];
      for (let panel = 0; panel < 3; panel++) {
        const destination = (y * canvas.width + panel * width + x) * 4;
        if (panel < 2) {
          const rgb = panel === 0 ? reference : current;
          for (let channel = 0; channel < 3; channel++) image.data[destination + channel] = display(rgb[channel]);
        } else {
          const error = Math.max(...reference.map((value, channel) => Math.abs(value - current[channel])));
          const heat = clamp(error / 0.4, 0, 1);
          image.data[destination] = Math.round(255 * heat);
          image.data[destination + 1] = Math.round(115 * heat * heat);
          image.data[destination + 2] = Math.round(28 * heat * heat * heat);
        }
        image.data[destination + 3] = 255;
      }
    }
    context.putImageData(image, 0, 0);
    container.hidden = false;
  }

  async runPathTracedReferenceAudit({ width = 64, height = 36, samples = 512, warmup = 96 } = {}) {
    const saved = {
      animateCamera: this.animateCamera,
      animateLights: this.animateLights,
      temporalStability: this.temporalStability,
      testTimeOverride: this.testTimeOverride,
    };
    let auditBuffer;
    let auditPassBuffer;
    try {
      this.animateCamera = false;
      this.animateLights = false;
      this.temporalStability = true;
      this.testTimeOverride = 0.7;
      this.resetProbeHistory();
      await this.waitFrames(warmup);
      await this.device.queue.onSubmittedWorkDone();

      const byteLength = width * height * 8 * 4;
      auditBuffer = createBuffer(
        this.device,
        "path-traced reference audit accumulation",
        byteLength,
        GPU.STORAGE | GPU.COPY_SRC | GPU.COPY_DST,
      );
      auditPassBuffer = createBuffer(
        this.device,
        "path-traced reference audit parameters",
        32,
        GPU.UNIFORM | GPU.COPY_DST,
        new Uint32Array([0, samples, width, height, 0, 0, 0, 0]),
      );
      const bindGroup = this.device.createBindGroup({
        label: "path-traced reference audit bind group",
        layout: this.computeLayout,
        entries: [
          { binding: 0, resource: { buffer: this.frameBuffer } },
          { binding: 1, resource: this.worldTexture.createView() },
          { binding: 2, resource: this.normalTexture.createView() },
          { binding: 3, resource: { buffer: this.hashBuffer } },
          { binding: 4, resource: { buffer: this.stateBuffer } },
          { binding: 5, resource: { buffer: this.probeMetaBuffer } },
          { binding: 6, resource: { buffer: auditBuffer } },
          { binding: 7, resource: { buffer: this.coneBuffer } },
          { binding: 8, resource: { buffer: this.irradianceBuffer } },
          { binding: 9, resource: { buffer: this.bvhNodeBuffer } },
          { binding: 10, resource: { buffer: this.triangleBuffer } },
          { binding: 11, resource: { buffer: auditPassBuffer } },
          { binding: 12, resource: this.materialAtlasView },
          { binding: 13, resource: this.materialSampler },
          { binding: 14, resource: this.irradianceAtlasWrite.createView() },
          { binding: 15, resource: this.irradianceAtlas.createView() },
          { binding: 16, resource: this.irradianceSampler },
          { binding: 17, resource: this.pointShadowArrayView },
          { binding: 18, resource: this.pointShadowSampler },
          { binding: 19, resource: this.sunShadowArrayView },
          { binding: 20, resource: this.shadowSampler },
          { binding: 21, resource: { buffer: this.sunShadowDataBuffer } },
        ],
      });
      const encoder = this.device.createCommandEncoder({ label: "path-traced reference audit" });
      encoder.clearBuffer(auditBuffer);
      const pass = encoder.beginComputePass({ label: `${samples} spp cosine reference` });
      pass.setPipeline(this.computePipelines.validateReference);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8), samples);
      pass.end();
      this.device.queue.submit([encoder.finish()]);
      const fields = new Uint32Array(await this.readGpuBuffer(auditBuffer, byteLength, "reference audit readback"));
      this.renderReferenceComparison(fields, width, height);
      const diagnosticFrame = await this.captureFinalFrame();

      const relative = [];
      const absolute = [];
      let squaredError = 0;
      let referenceEnergy = 0;
      let signedBias = 0;
      let referenceLuminance = 0;
      let currentLuminance = 0;
      let currentReferenceDot = 0;
      let currentEnergy = 0;
      let channelCount = 0;
      let activePixels = 0;
      let severeUnderlitPixels = 0;
      let severeOverlitPixels = 0;
      const pairs = [];
      const pixelPairs = new Array(width * height);
      const luminance = (rgb) => rgb[0] * 0.2126 + rgb[1] * 0.7152 + rgb[2] * 0.0722;
      for (let pixel = 0; pixel < width * height; pixel++) {
        const base = pixel * 8;
        if (!fields[base + 6] || !fields[base + 7]) continue;
        const reference = [0, 1, 2].map((channel) => fields[base + channel] / (fields[base + 7] * 65536));
        const current = [3, 4, 5].map((channel) => fields[base + channel] / 65536);
        const refLum = luminance(reference);
        const currentLum = luminance(current);
        const delta = Math.abs(refLum - currentLum);
        absolute.push(delta);
        relative.push(delta / Math.max(0.08, (refLum + currentLum) * 0.5));
        if (refLum >= 0.08 && currentLum < refLum * 0.35) severeUnderlitPixels++;
        if (currentLum >= 0.08 && currentLum > refLum * 2.5 + 0.04) severeOverlitPixels++;
        signedBias += currentLum - refLum;
        referenceLuminance += refLum;
        currentLuminance += currentLum;
        for (let channel = 0; channel < 3; channel++) {
          const error = current[channel] - reference[channel];
          squaredError += error * error;
          referenceEnergy += reference[channel] * reference[channel];
          currentReferenceDot += current[channel] * reference[channel];
          currentEnergy += current[channel] * current[channel];
          channelCount++;
        }
        const pair = { reference, current, pixel };
        pairs.push(pair);
        pixelPairs[pixel] = pair;
        activePixels++;
      }
      relative.sort((a, b) => a - b);
      absolute.sort((a, b) => a - b);
      const percentile = (values, p) => values.length
        ? values[Math.min(values.length - 1, Math.floor((values.length - 1) * p))]
        : Infinity;
      const nrmse = Math.sqrt(squaredError / Math.max(1, channelCount))
        / Math.max(0.08, Math.sqrt(referenceEnergy / Math.max(1, channelCount)));
      const optimalScale = currentReferenceDot / Math.max(1e-8, currentEnergy);
      let scaledSquaredError = 0;
      for (const pair of pairs) {
        for (let channel = 0; channel < 3; channel++) {
          const error = pair.current[channel] * optimalScale - pair.reference[channel];
          scaledSquaredError += error * error;
        }
      }
      const scaleInvariantNrmse = Math.sqrt(scaledSquaredError / Math.max(1, channelCount))
        / Math.max(0.08, Math.sqrt(referenceEnergy / Math.max(1, channelCount)));
      // A single low-resolution pixel whose cosine ray happens to hit a tiny,
      // bright emitter can dominate raw NRMSE even though the field is accurate
      // everywhere else. Preserve raw NRMSE, but also report a fixed 99%-trimmed
      // metric. The percentile/dark/leak gates below still cover every pixel
      // class and prevent the trim from hiding broad artifacts.
      const rankedPairs = pairs.map((pair) => {
        let errorEnergy = 0;
        let pairReferenceEnergy = 0;
        for (let channel = 0; channel < 3; channel++) {
          errorEnergy += (pair.current[channel] - pair.reference[channel]) ** 2;
          pairReferenceEnergy += pair.reference[channel] ** 2;
        }
        return { ...pair, errorEnergy, pairReferenceEnergy };
      }).sort((a, b) => a.errorEnergy - b.errorEnergy);
      const trimmedCount = Math.max(1, Math.floor(rankedPairs.length * 0.99));
      let trimmedSquaredError = 0;
      let trimmedReferenceEnergy = 0;
      for (let index = 0; index < trimmedCount; index++) {
        trimmedSquaredError += rankedPairs[index].errorEnergy;
        trimmedReferenceEnergy += rankedPairs[index].pairReferenceEnergy;
      }
      const trimmedNrmse99 = Math.sqrt(trimmedSquaredError / (trimmedCount * 3))
        / Math.max(0.08, Math.sqrt(trimmedReferenceEnergy / (trimmedCount * 3)));
      const maximumErrorPair = rankedPairs.at(-1);
      // Split RC deliberately reconstructs a low-frequency irradiance field.
      // Keep the raw pixel metric above, but also compare both images through
      // one 3x3 low-pass footprint so raster edge aliasing is not mislabeled
      // as a GI error.
      const lowFrequencyPairs = [];
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          if (!pixelPairs[y * width + x]) continue;
          const reference = [0, 0, 0];
          const current = [0, 0, 0];
          let contributors = 0;
          for (let oy = -1; oy <= 1; oy++) {
            for (let ox = -1; ox <= 1; ox++) {
              const sx = x + ox;
              const sy = y + oy;
              if (sx < 0 || sy < 0 || sx >= width || sy >= height) continue;
              const neighbor = pixelPairs[sy * width + sx];
              if (!neighbor) continue;
              for (let channel = 0; channel < 3; channel++) {
                reference[channel] += neighbor.reference[channel];
                current[channel] += neighbor.current[channel];
              }
              contributors++;
            }
          }
          if (!contributors) continue;
          for (let channel = 0; channel < 3; channel++) {
            reference[channel] /= contributors;
            current[channel] /= contributors;
          }
          lowFrequencyPairs.push({ reference, current });
        }
      }
      let lowReferenceEnergy = 0;
      let lowCurrentEnergy = 0;
      let lowCurrentReferenceDot = 0;
      for (const pair of lowFrequencyPairs) {
        for (let channel = 0; channel < 3; channel++) {
          lowReferenceEnergy += pair.reference[channel] ** 2;
          lowCurrentEnergy += pair.current[channel] ** 2;
          lowCurrentReferenceDot += pair.current[channel] * pair.reference[channel];
        }
      }
      const lowScale = lowCurrentReferenceDot / Math.max(1e-8, lowCurrentEnergy);
      let lowScaledSquaredError = 0;
      for (const pair of lowFrequencyPairs) {
        for (let channel = 0; channel < 3; channel++) {
          const error = pair.current[channel] * lowScale - pair.reference[channel];
          lowScaledSquaredError += error * error;
        }
      }
      const lowChannelCount = lowFrequencyPairs.length * 3;
      const lowFrequencyScaleInvariantNrmse = Math.sqrt(
        lowScaledSquaredError / Math.max(1, lowChannelCount),
      ) / Math.max(0.08, Math.sqrt(lowReferenceEnergy / Math.max(1, lowChannelCount)));
      const rankedLowFrequencyPairs = lowFrequencyPairs.map((pair) => {
        let errorEnergy = 0;
        let pairReferenceEnergy = 0;
        for (let channel = 0; channel < 3; channel++) {
          const error = pair.current[channel] * lowScale - pair.reference[channel];
          errorEnergy += error * error;
          pairReferenceEnergy += pair.reference[channel] ** 2;
        }
        return { errorEnergy, pairReferenceEnergy };
      }).sort((a, b) => a.errorEnergy - b.errorEnergy);
      const trimmedLowCount = Math.max(1, Math.floor(rankedLowFrequencyPairs.length * 0.99));
      let trimmedLowSquaredError = 0;
      let trimmedLowReferenceEnergy = 0;
      for (let index = 0; index < trimmedLowCount; index++) {
        trimmedLowSquaredError += rankedLowFrequencyPairs[index].errorEnergy;
        trimmedLowReferenceEnergy += rankedLowFrequencyPairs[index].pairReferenceEnergy;
      }
      const trimmedLowFrequencyScaleInvariantNrmse99 = Math.sqrt(
        trimmedLowSquaredError / (trimmedLowCount * 3),
      ) / Math.max(0.08, Math.sqrt(trimmedLowReferenceEnergy / (trimmedLowCount * 3)));
      const report = {
        resolution: [width, height],
        samples,
        activePixels,
        nrmse,
        trimmedNrmse99,
        optimalEnergyScale: optimalScale,
        scaleInvariantNrmse,
        lowFrequencyOptimalEnergyScale: lowScale,
        lowFrequencyScaleInvariantNrmse,
        trimmedLowFrequencyScaleInvariantNrmse99,
        meanReferenceLuminance: referenceLuminance / Math.max(1, activePixels),
        meanSplitRCLuminance: currentLuminance / Math.max(1, activePixels),
        meanSignedLuminanceBias: signedBias / Math.max(1, activePixels),
        medianRelative: percentile(relative, 0.5),
        p95Relative: percentile(relative, 0.95),
        p95Absolute: percentile(absolute, 0.95),
        p99Absolute: percentile(absolute, 0.99),
        maximumAbsolute: absolute.at(-1) ?? Infinity,
        maximumErrorPixel: maximumErrorPair
          ? [maximumErrorPair.pixel % width, Math.floor(maximumErrorPair.pixel / width)]
          : null,
        maximumErrorReference: maximumErrorPair?.reference || null,
        maximumErrorSplitRC: maximumErrorPair?.current || null,
        maximumErrorWorld: maximumErrorPair
          ? this.worldAt(
            diagnosticFrame,
            Math.min(diagnosticFrame.width - 1, Math.floor(
              ((maximumErrorPair.pixel % width) * diagnosticFrame.width + width / 2) / width,
            )),
            Math.min(diagnosticFrame.height - 1, Math.floor(
              (Math.floor(maximumErrorPair.pixel / width) * diagnosticFrame.height + height / 2) / height,
            )),
          )
          : null,
        maximumErrorNormal: maximumErrorPair
          ? this.normalAt(
            diagnosticFrame,
            Math.min(diagnosticFrame.width - 1, Math.floor(
              ((maximumErrorPair.pixel % width) * diagnosticFrame.width + width / 2) / width,
            )),
            Math.min(diagnosticFrame.height - 1, Math.floor(
              (Math.floor(maximumErrorPair.pixel / width) * diagnosticFrame.height + height / 2) / height,
            )),
          )
          : null,
        severeUnderlitRatio: severeUnderlitPixels / Math.max(1, activePixels),
        severeOverlitRatio: severeOverlitPixels / Math.max(1, activePixels),
      };
      const frozenBaseline = REFERENCE_BASELINES[this.sceneIndex];
      report.frozenBaselineCeilings = frozenBaseline || null;
      report.frozenBaselinePassed = !frozenBaseline || Object.entries(frozenBaseline)
        .every(([metric, ceiling]) => report[metric] <= ceiling);
      report.paperSceneStrictPassed = this.sceneIndex !== 1
        || (
          report.nrmse <= 0.25
          && report.trimmedNrmse99 <= 0.22
          && report.lowFrequencyScaleInvariantNrmse <= 0.15
          && report.p95Absolute <= 0.06
          && report.p99Absolute <= 0.10
          && report.severeOverlitRatio <= 0.01
        );
      report.passed = activePixels >= 64
        && report.frozenBaselinePassed
        && report.trimmedNrmse99 <= 0.34
        && report.trimmedLowFrequencyScaleInvariantNrmse99 <= 0.36
        && report.p95Absolute <= 0.15
        && report.p99Absolute <= 0.40
        && report.severeUnderlitRatio <= 0.01
        && report.severeOverlitRatio <= 0.04
        && Math.abs(report.meanSignedLuminanceBias) <= 0.02
        && report.paperSceneStrictPassed;
      return report;
    } finally {
      auditBuffer?.destroy();
      auditPassBuffer?.destroy();
      Object.assign(this, saved);
      this.resetProbeHistory();
    }
  }

  metricsSnapshot() {
    const avg = this.frameSamples.slice(-60).reduce((a,b)=>a+b,0) / Math.max(1,Math.min(60,this.frameSamples.length));
    const gpu = this.gpuSamples.length ? this.gpuSamples.slice(-8).reduce((a,b)=>a+b,0)/Math.min(8,this.gpuSamples.length) : null;
    return {
      scene: this.sceneIndex,
      name: SCENE_INFO[this.sceneIndex].name,
      fps: avg ? 1000/avg : 0,
      gpuMs: gpu,
      triangles: (this.scene?.geometry.triangleCount || 0)
        + (this.dynamicScene?.triangleCount || 0),
      dynamicInstances: this.dynamicScene?.instanceCount || 0,
      dynamicEmissiveInstances: this.dynamicScene?.emissiveInstanceCount || 0,
      dynamicTlasNodes: this.dynamicScene?.tlasNodeCount || 0,
      dynamicSweptNodes: this.dynamicScene?.sweptTlasNodeCount || 0,
      dynamicUpdateP95Ms: this.dynamicUpdateMs || 0,
      dynamicUploadBytes: this.dynamicUploadBytes || 0,
      probes: [...this.probeCounts],
      rays: this.rayCount,
      hitRate: this.rayCount ? this.hitCount/this.rayCount : 0,
      environmentAccess: this.environmentAccess ?? null,
      overflows: this.overflowCount,
      gpuError: this.lastGpuError || null,
    };
  }

  exposeTestReport(report) {
    globalThis.__RC_TEST_REPORT__ = report;
    document.documentElement.dataset.audit = report.passed ? "passed" : "warning";
    let hidden = document.getElementById("test-report");
    if (!hidden) {
      hidden = document.createElement("pre");
      hidden.id = "test-report";
      hidden.hidden = true;
      document.body.append(hidden);
    }
    hidden.textContent = JSON.stringify(report);
    if (new URLSearchParams(location.search).has("autotest")) {
      const card = $("audit-card");
      if (card) card.hidden = false;
      if ($("audit-title")) $("audit-title").textContent = report.passed
        ? "Automated audit passed"
        : "Automated audit warning";
      if ($("audit-progress")) $("audit-progress").style.width = "100%";
      if ($("audit-report")) $("audit-report").textContent = JSON.stringify(report, null, 2);
      hideStatus();
    }
    console.info("[Split RC] validation-complete", report);
  }

  async runValidation({ framesPerScene = 72 } = {}) {
    const card = $("audit-card");
    card.hidden = false;
    $("audit-report").textContent = `Beginning deterministic ${SCENE_INFO.length}-scene audit…`;
    const previousCamera = this.animateCamera;
    const previousLights = this.animateLights;
    this.animateCamera = true;
    this.animateLights = true;
    this.lastGpuError = null;
    const results = [];
    for (let i = 0; i < SCENE_INFO.length; i++) {
      $("audit-title").textContent = `Running scene ${i+1} of ${SCENE_INFO.length}`;
      $("audit-progress").style.width = `${i/SCENE_INFO.length*100}%`;
      await this.loadScene(i);
      await this.waitFrames(framesPerScene);
      const result = this.metricsSnapshot();
      // The fixed-light field switches to its long .98 history after a short
      // bootstrap. Measure steady-state variance only after that transition;
      // pixel-space motion audits below separately cover the startup path.
      result.motionStability = await this.runMotionStabilityAudit({
        samples: 4,
        interval: 3,
        warmup: 384,
      });
      result.finalFrameRepeatability = await this.runFinalFrameRepeatabilityAudit({
        poses: 4,
        warmup: 48,
        holdFrames: 2,
      });
      result.continuousMotion = await this.runContinuousMotionAudit({
        frames: 32,
        warmup: 192,
      });
      result.movingLightContinuousMotion = await this.runContinuousMotionAudit({
        frames: 32,
        warmup: 192,
        movingLights: true,
        timeStep: 1 / 60,
      });
      if (i === 1 || i === 10) {
        result.movingLightResponse = await this.runMovingLightResponseAudit();
      }
      if ([0, 10, 11].includes(i)) {
        result.shadowMapCorrectness = await this.runShadowMapAudit({
          width: 80,
          height: 45,
          warmup: 8,
        });
      }
      if (i === 1) {
        result.cacheMotionRecovery = await this.runLongTranslationCacheAudit({
          motionFrames: 54,
          captureInterval: 6,
        });
      }
      if ([0, 1, 10, 11, 12].includes(i)) {
        $("audit-title").textContent = `Comparing ${SCENE_INFO[i].short} with a path-traced reference`;
        result.pathTracedReference = await this.runPathTracedReferenceAudit({
          width: 64,
          height: 36,
          samples: 512,
          warmup: 192,
        });
      }
      if (i === 0) {
        $("audit-title").textContent = "Testing closed-volume light-leak rejection";
        result.enclosureLeak = await this.runEnclosureLeakAudit();
      }
      if (i === 8) {
        $("audit-title").textContent = "Testing open-door zoom continuity and closed-room darkness";
        result.doorZoomContinuity = await this.runDoorZoomContinuityAudit();
      }
      if (i === 10) {
        $("audit-title").textContent = "Replaying reported Cornell artifact poses";
        result.cornellArtifacts = await this.runCornellArtifactAudit();
      }
      results.push(result);
      $("audit-report").textContent = results.map((r) =>
        `${String(r.scene+1).padStart(2,"0")} ${r.name.padEnd(28)} ${r.fps.toFixed(0).padStart(3)} FPS  ${r.gpuMs == null ? "CPU timing" : `${r.gpuMs.toFixed(2)} ms GPU`}  ${r.triangles.toLocaleString()} tris  world jitter p95 ${(r.motionStability.p95RelativeMax*100).toFixed(2)}%  framebuffer repeat p95 ${r.finalFrameRepeatability.p95ByteDeltaMax.toFixed(0)}/255  motion p95 ${r.continuousMotion.p95ByteDeltaMax.toFixed(0)}/255  moving-light motion p95 ${r.movingLightContinuousMotion.p95ByteDeltaMax.toFixed(0)}/255${r.cacheMotionRecovery ? `  cache recovery p95 ${r.cacheMotionRecovery.recoveryDifference.p95ByteDelta.toFixed(0)}/255` : ""}${r.shadowMapCorrectness ? `  shadow mismatch point ${(r.shadowMapCorrectness.point.classificationMismatchRatio*100).toFixed(1)}% sun ${(r.shadowMapCorrectness.sun.classificationMismatchRatio*100).toFixed(1)}%` : ""}${r.pathTracedReference ? `  reference NRMSE ${(r.pathTracedReference.nrmse*100).toFixed(1)}%` : ""}  overflow ${r.overflows}`
      ).join("\n");
    }
    const minFps = Math.min(...results.map((r) => r.fps));
    const maxGpu = Math.max(0, ...results.map((r) => r.gpuMs || 0));
    const failures = results.filter((r) =>
      r.overflows
      || r.gpuError
      || !r.motionStability.passed
      || !r.finalFrameRepeatability.passed
      || !r.continuousMotion.passed
      || !r.movingLightContinuousMotion.passed
      || (r.cacheMotionRecovery && !r.cacheMotionRecovery.passed)
      || (r.movingLightResponse && !r.movingLightResponse.passed)
      || (r.shadowMapCorrectness && !r.shadowMapCorrectness.passed)
      || (r.pathTracedReference && !r.pathTracedReference.passed)
      || (r.enclosureLeak && !r.enclosureLeak.passed)
      || (r.doorZoomContinuity && !r.doorZoomContinuity.passed)
      || (r.cornellArtifacts && !r.cornellArtifacts.passed)
    );
    const report = {
      timestamp: new Date().toISOString(),
      adapter: this.adapterInfo(),
      quality: this.qualityName,
      resolution: [this.width, this.height],
      giResolution: [this.giWidth, this.giHeight],
      minFps,
      maxGpuMs: maxGpu || null,
      passed: failures.length === 0 && minFps >= 30,
      scenes: results,
    };
    $("audit-title").textContent = report.passed ? "Audit passed" : "Audit completed with warnings";
    $("audit-progress").style.width = "100%";
    $("audit-report").textContent += `\n\n${report.passed ? "PASS" : "CHECK"} · minimum ${minFps.toFixed(1)} FPS · ${failures.length} correctness warnings`;
    this.animateCamera = previousCamera;
    this.animateLights = previousLights;
    this.exposeTestReport(report);
    return report;
  }

  async setScene(index) {
    await this.loadScene(index);
    await this.waitFrames(12);
    return this.metricsSnapshot();
  }

  destroy() {
    if (this.destroyed) return;
    this.destroyed = true;
    this.running = false;
    for (const fn of this.cleanup) fn();
    for (const resource of [
      this.vertexBuffer, this.dynamicVertexBuffer, this.bvhNodeBuffer, this.triangleBuffer,
      this.emissiveBvhNodeBuffer, this.emissiveTriangleBuffer, this.frameBuffer,
      this.hashBuffer, this.stateBuffer, this.probeMetaBuffer, this.accumBuffer,
      this.coneBuffer, this.irradianceBuffer, this.queryResolveBuffer,
      this.irradianceAtlasWrite, this.irradianceAtlas, this.sunShadowDataBuffer,
      ...(this.sunShadowBuffers || []), ...(this.pointShadowBuffers || []),
      ...(this.passBuffers || []), ...(this.gbuffer || []), this.shadowTexture,
      this.pointShadowTexture,
      this.materialAtlas,
    ]) resource?.destroy?.();
    this.cleanup.length = 0;
    if (globalThis.__splitRC === this) globalThis.__splitRC = undefined;
  }
}

export async function startRadianceCascades(loaderGeneration = null) {
  if (
    loaderGeneration != null
    && globalThis.__splitRCLoaderGeneration !== loaderGeneration
  ) return null;
  const existing = globalThis.__splitRC;
  if (existing && !existing.destroyed) {
    if (loaderGeneration == null || existing.loaderGeneration === loaderGeneration) {
      return existing;
    }
    existing.destroy();
  }
  const canvas = $("viewport");
  const renderer = new SplitRadianceCascades(canvas);
  renderer.loaderGeneration = loaderGeneration;
  globalThis.__splitRC = renderer;
  try {
    const initialized = await renderer.initialize();
    const stale = loaderGeneration != null
      && globalThis.__splitRCLoaderGeneration !== loaderGeneration;
    if (!initialized || stale || globalThis.__splitRC !== renderer) {
      renderer.destroy();
      return globalThis.__splitRC || null;
    }
  } catch (error) {
    if (renderer.destroyed) return globalThis.__splitRC || null;
    setStatus("WebGPU initialization failed", error.message || String(error), true);
    document.documentElement.dataset.webgpu = "failed";
    console.error("[Split RC]", error);
  }
  return renderer;
}

if (typeof document !== "undefined") {
  globalThis.__startSplitRC = startRadianceCascades;
  if (globalThis.__splitRCAutoStart !== false) startRadianceCascades();
}
