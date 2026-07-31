import {
  adaptiveBudgetScale, add3, clamp, cross3, dot3, mat4LookAt, mat4Multiply, mat4Ortho,
  mat4Perspective, mul3, normalize3, sub3,
} from "./math.js";
import { createScene, SCENE_INFO } from "./scenes.js";
import {
  computeShader, finalShader, presentShader, rasterShader, shaderConstants as K,
} from "./shaders.js";

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
    nrmse: 0.56, trimmedNrmse99: 0.21,
    lowFrequencyScaleInvariantNrmse: 0.36,
    trimmedLowFrequencyScaleInvariantNrmse99: 0.33,
    p95Absolute: 0.04, p99Absolute: 0.14,
  }),
  1: Object.freeze({
    nrmse: 0.40, trimmedNrmse99: 0.33,
    lowFrequencyScaleInvariantNrmse: 0.24,
    trimmedLowFrequencyScaleInvariantNrmse99: 0.22,
    // Tangent-complete c0 support removes camera-dependent coverage holes and
    // improves every aggregate Sponza error metric. Its deterministic 99th
    // percentile is 0.386, so keep the general 0.40 safety ceiling here too.
    p95Absolute: 0.15, p99Absolute: 0.40,
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
  }

  async initialize() {
    if (!navigator.gpu) {
      throw new Error("WebGPU is not available in this browser. Use current Chrome or Edge with hardware acceleration enabled.");
    }
    this.adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!this.adapter) throw new Error("No WebGPU adapter was returned by the browser.");
    const requiredFeatures = [];
    if (this.adapter.features.has("timestamp-query")) requiredFeatures.push("timestamp-query");
    this.device = await this.adapter.requestDevice({ requiredFeatures });
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
    await this.createMaterialAtlas();
    this.createPersistentResources();
    this.installUI();
    await this.loadScene(1);
    this.running = true;
    this.lastTime = performance.now();
    requestAnimationFrame((t) => this.frame(t));
    hideStatus();
    this.updateAdapterLabel();
    console.info("[Split RC] renderer-ready", this.adapterInfo());

    const automaticTest = new URLSearchParams(location.search).get("autotest");
    if (automaticTest === "reference") {
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
          debugMode: Math.max(0, Math.min(6, Math.floor(numericParameter("mode", 1)))),
          preservePose: parameters.get("show") === "1",
        });
        this.exposeTestReport(report);
      }, 200);
    } else if (automaticTest?.startsWith("motion-")) {
      setTimeout(async () => {
        const requested = Number(automaticTest.slice(7));
        const index = clamp(Number.isFinite(requested) ? Math.floor(requested) : 0, 0, SCENE_INFO.length - 1);
        await this.loadScene(index);
        const report = {
          scene: index,
          baseline: await this.runContinuousMotionAudit({ frames: 32, warmup: 64 }),
          movingLights: await this.runContinuousMotionAudit({
            frames: 32,
            warmup: 64,
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
        await this.loadScene(index);
        const report = {
          scene: index,
          reference: await this.runPathTracedReferenceAudit(),
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
          warmup: index === 6 ? 192 : ([5, 11].includes(index) ? 96 : 48),
        });
        result.finalFrameRepeatability = await this.runFinalFrameRepeatabilityAudit({ poses: 5, warmup: 64 });
        result.continuousMotion = await this.runContinuousMotionAudit({
          frames: index === 1 ? 32 : 24,
          warmup: 64,
        });
        result.movingLightContinuousMotion = await this.runContinuousMotionAudit({
          frames: index === 1 ? 32 : 24,
          warmup: 64,
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
        if ([0, 1, 10, 11].includes(index)) {
          result.pathTracedReference = await this.runPathTracedReferenceAudit();
        }
        result.passed = !result.overflows
          && !result.gpuError
          && result.motionStability.passed
          && result.finalFrameRepeatability.passed
          && result.continuousMotion.passed
          && result.movingLightContinuousMotion.passed
          && (!result.movingLightResponse || result.movingLightResponse.passed)
          && (!result.cacheMotionRecovery || result.cacheMotionRecovery.passed)
          && (!result.pathTracedReference || result.pathTracedReference.passed);
        this.exposeTestReport(result);
      }, 200);
    } else if (automaticTest != null) {
      // Sixty scene frames guarantee at least one 45-frame timestamp-query
      // cycle after each scene resets its metric history.
      setTimeout(() => this.runValidation({ framesPerScene: 60 }), 200);
    }
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
          { format: "rgba16float" },
          { format: "rgba16float" },
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
      splitRays: cp("splitRays"),
      merge: cp("mergeCascade"),
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
    const response = await fetch("/models/sponza-atlas.webp");
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
    this.frameBuffer = createBuffer(d, "frame uniforms", 256, GPU.UNIFORM | GPU.COPY_DST);
    this.hashBuffer = createBuffer(d, "double-buffered sparse probe hash", K.totalHashSlots * K.hashFrames * 8, GPU.STORAGE | GPU.COPY_SRC | GPU.COPY_DST);
    this.stateBuffer = createBuffer(d, "probe counters, ray prefixes, and diagnostics", K.stateWords * 4, GPU.STORAGE | GPU.COPY_SRC | GPU.COPY_DST);
    this.probeMetaBuffer = createBuffer(d, "sparse probe metadata", K.totalProbeMeta * 16, GPU.STORAGE | GPU.COPY_DST);
    this.accumBuffer = createBuffer(
      d,
      "double-buffered fixed-point ray intervals",
      K.totalDirectionData * 5 * K.accumFrames * 4,
      GPU.STORAGE | GPU.COPY_DST,
    );
    this.coneBuffer = createBuffer(d, "merged radiance cones", K.totalDirectionData * 16, GPU.STORAGE | GPU.COPY_DST);
    this.irradianceBuffer = createBuffer(d, "double-buffered bordered 6x6 probe irradiance", K.probeCaps[0] * K.irradianceTexels * K.irradianceFrames * 16, GPU.STORAGE | GPU.COPY_SRC | GPU.COPY_DST);
    // WebGPU forbids writable storage and sampled usage for one texture in a
    // single synchronization scope. Keep the compute target separate, then
    // copy the completed frame half into the filterable atlas.
    this.irradianceAtlasWrite = d.createTexture({
      label: "double-buffered 8x8 probe irradiance storage atlas",
      size: [K.irradianceAtlasWidth, K.irradianceAtlasFrameHeight * K.irradianceFrames],
      format: "rgba16float",
      usage: TEX.STORAGE_BINDING | TEX.COPY_SRC,
    });
    this.irradianceAtlas = d.createTexture({
      label: "double-buffered filterable 8x8 probe irradiance atlas",
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
      "G-buffer world position",
      "rgba16float",
      TEX.RENDER_ATTACHMENT | TEX.TEXTURE_BINDING | TEX.COPY_SRC,
    );
    this.emissiveTexture = texture("G-buffer emissive radiance", "rgba16float");
    this.depthTexture = texture("G-buffer depth", "depth24plus", TEX.RENDER_ATTACHMENT);
    const compositeUsage = TEX.RENDER_ATTACHMENT | TEX.TEXTURE_BINDING | TEX.COPY_SRC;
    this.compositeTexture = texture("current Split RC composite", this.format, compositeUsage);
    this.gbuffer = [
      this.albedoTexture,
      this.normalTexture,
      this.worldTexture,
      this.emissiveTexture,
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
        { binding: 9, resource: { buffer: this.accumBuffer } },
        { binding: 10, resource: this.irradianceSampler },
        { binding: 11, resource: this.emissiveTexture.createView() },
        { binding: 12, resource: this.pointShadowArrayView },
        { binding: 13, resource: this.pointShadowSampler },
        { binding: 14, resource: { buffer: this.sunShadowDataBuffer } },
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
    this.bvhNodeBuffer?.destroy();
    this.triangleBuffer?.destroy();
    const geometry = scene.geometry;
    this.vertexBuffer = createBuffer(this.device, `${info.short} raster geometry`, geometry.vertices.byteLength, GPU.VERTEX | GPU.COPY_DST, geometry.vertices);
    this.bvhNodeBuffer = createBuffer(this.device, `${info.short} BVH nodes`, geometry.nodes.byteLength, GPU.STORAGE | GPU.COPY_DST, geometry.nodes);
    this.triangleBuffer = createBuffer(this.device, `${info.short} BVH triangles`, geometry.triangles.byteLength, GPU.STORAGE | GPU.COPY_DST, geometry.triangles);
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
    console.info("[Split RC] scene-loaded", { index, name: info.name, triangles: geometry.triangleCount, bvhNodes: geometry.nodeCount });
  }

  setCameraFromScene(scene) {
    const offset = sub3(scene.camera, scene.target);
    const horizontal = Math.hypot(offset[0], offset[2]);
    this.camera = {
      target: [...scene.target],
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
    const sunHorizontal = this.scene.sunHorizontal ?? (this.sceneIndex === 1 ? 0.28 : 0.7);
    const sunHeight = this.scene.sunHeight ?? (this.sceneIndex === 1 ? -0.96 : -0.74);
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
    const hue = this.sceneIndex % 3;
    const pointColor = this.scene.pointColor ?? (this.sceneIndex === 1
      ? [1.0, 1.0, 1.0]
      : hue === 0 ? [1.0, 0.18, 0.045] : hue === 1 ? [0.08, 0.5, 1.0] : [0.18, 1.0, 0.35]);
    const sunColor = this.scene.sunColor ?? (this.sceneIndex === 1 ? [1.0, 0.98, 0.92] : [1.0, 0.84, 0.63]);
    const pointIntensity = this.scene.pointIntensity ?? (this.sceneIndex === 1 ? 0.0 : 10.0 + this.sceneIndex * 0.7);
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

    const u = new Float32Array(64);
    u.set(viewProjection, 0);
    u.set(sunVP, 16);
    const featureFlags = (this.temporalStability ? 8 : 0)
      | (this.sceneIndex === 1 ? 16 : 0)
      | (this.animateLights ? 32 : 0);
    u.set([...cameraPosition, featureFlags], 32);
    u.set([...sunDirection, seconds], 36);
    u.set([...sunColor, this.scene.sun], 40);
    u.set([...pointPosition, pointRange], 44);
    u.set([...pointColor, pointIntensity], 48);
    u.set([...this.scene.env, this.scene.baseSpacing], 52);
    u.set([this.width, this.height, this.giWidth, this.giHeight], 56);
    const frameParity = this.frameIndex & 1;
    // A nonzero fixed-light value enables exact sample-count accumulation in
    // the shader; its magnitude is used only by animated-light EMA. Exact-key
    // rejection still makes disocclusions immediate.
    const fixedLightHistory = this.sampleFrameIndex < 24
      ? 0.92
      : 0.98;
    const historyBlend = this.temporalStability && this.historyValid
      ? (this.animateLights
        ? 0.965
        : fixedLightHistory)
      : 0;
    const exposure = this.scene.exposure ?? (this.sceneIndex === 1 ? 1.55 : 1.0);
    u.set([this.indirectStrength, exposure, this.debugMode, frameParity + historyBlend], 60);
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
        pointShadow.end();
      }
    }

    const gbuffer = encoder.beginRenderPass({
      label: "G-buffer",
      colorAttachments: [
        { view: this.albedoTexture.createView(), clearValue: [0,0,0,0], loadOp: "clear", storeOp: "store" },
        { view: this.normalTexture.createView(), clearValue: [0,0,0,0], loadOp: "clear", storeOp: "store" },
        { view: this.worldTexture.createView(), clearValue: [0,0,0,0], loadOp: "clear", storeOp: "store" },
        { view: this.emissiveTexture.createView(), clearValue: [0,0,0,0], loadOp: "clear", storeOp: "store" },
      ],
      depthStencilAttachment: { view: this.depthTexture.createView(), depthClearValue: 1, depthLoadOp: "clear", depthStoreOp: "store" },
      timestampWrites: profile ? { querySet: this.querySet, beginningOfPassWriteIndex: 2, endOfPassWriteIndex: 3 } : undefined,
    });
    gbuffer.setPipeline(this.gbufferPipeline);
    gbuffer.setBindGroup(0, this.rasterBindGroup);
    gbuffer.setVertexBuffer(0, this.vertexBuffer);
    gbuffer.draw(this.scene.geometry.vertexCount);
    gbuffer.end();

    encoder.clearBuffer(this.stateBuffer);
    const accumFrameBytes = K.totalDirectionData * 5 * 4;
    encoder.clearBuffer(this.accumBuffer, (this.frameIndex & 1) * accumFrameBytes, accumFrameBytes);
    let pass = encoder.beginComputePass({
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
    pass = encoder.beginComputePass({
      label: "prefilter probe irradiance",
      timestampWrites: profile ? { querySet: this.querySet, endOfPassWriteIndex: 5 } : undefined,
    });
    pass.setPipeline(this.computePipelines.prefilter);
    pass.setBindGroup(0, this.computeBindGroups[0]);
    pass.dispatchWorkgroups(Math.ceil(K.probeCaps[0] * K.irradianceTexels / 64));
    pass.end();
    const atlasFrameY = (this.frameIndex & 1) * K.irradianceAtlasFrameHeight;
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
      const worldBytesPerRow = Math.ceil(this.width * 8 / 256) * 256;
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
        worldBytesPerRow * this.height,
        GPU.COPY_DST | GPU.MAP_READ,
      );
      const diagnosticBuffer = createBuffer(
        d,
        "per-capture sparse diagnostics",
        32,
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
        { buffer: normalBuffer, bytesPerRow: worldBytesPerRow, rowsPerImage: this.height },
        [this.width, this.height],
      );
      encoder.copyBufferToBuffer(this.stateBuffer, 0, diagnosticBuffer, 0, 32);
      captureResources = {
        ...captureJob,
        buffer,
        worldBuffer,
        normalBuffer,
        diagnosticBuffer,
        bytesPerRow,
        worldBytesPerRow,
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
      stateRead = createBuffer(d, "diagnostic readback", 32, GPU.COPY_DST | GPU.MAP_READ);
      encoder.copyBufferToBuffer(this.stateBuffer, 0, stateRead, 0, 32);
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
        worldPixels: new Uint16Array(job.worldBuffer.getMappedRange().slice(0)),
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
    $("scene-description").textContent = `${info.description} ${this.scene.geometry.triangleCount.toLocaleString()} ray-traced triangles.`;
    $("scene-index").textContent = `${String(this.sceneIndex+1).padStart(2,"0")} / ${SCENE_INFO.length}`;
    $("scene-select").value = String(this.sceneIndex);
    document.querySelectorAll(".scene-strip button").forEach((button, i) => button.classList.toggle("active", i === this.sceneIndex));
  }

  installUI() {
    const select = $("scene-select");
    const strip = $("scene-strip");
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
      // Capture both buffers and the frame parity in one submission. Separate
      // asynchronous copies allowed animation frames to advance between copies,
      // occasionally pairing a hash frame with the wrong irradiance frame.
      current = (this.frameIndex - 1) & 1;
      previous = 1 - current;
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
    const byteOffset = y * frame.worldBytesPerRow + x * 8;
    const index = byteOffset >> 1;
    return [
      halfToFloat(frame.worldPixels[index]),
      halfToFloat(frame.worldPixels[index + 1]),
      halfToFloat(frame.worldPixels[index + 2]),
      halfToFloat(frame.worldPixels[index + 3]),
    ];
  }

  normalAt(frame, x, y) {
    const byteOffset = y * frame.worldBytesPerRow + x * 8;
    const index = byteOffset >> 1;
    const normal = [
      halfToFloat(frame.normalPixels[index]),
      halfToFloat(frame.normalPixels[index + 1]),
      halfToFloat(frame.normalPixels[index + 2]),
    ];
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
        if (!(world[3] > 0.5) || !world.every(Number.isFinite)) continue;
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
        if (!(world[3] > 0.5) || !world.every(Number.isFinite)) continue;
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
      const pointPassed = !this.pointShadowsEnabled
        || (lightPassed(point, 32) && pointFaceCoveragePassed);
      const sunPassed = lightPassed(sun, 32);
      return {
        resolution: [width, height],
        scene: this.sceneIndex,
        time,
        point,
        sun,
        pointFaceCoverageRequired,
        pointFaceCoveragePassed,
        pointPassed,
        sunPassed,
        passed: pointPassed && sunPassed,
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
        severeUnderlitRatio: severeUnderlitPixels / Math.max(1, activePixels),
        severeOverlitRatio: severeOverlitPixels / Math.max(1, activePixels),
      };
      const frozenBaseline = REFERENCE_BASELINES[this.sceneIndex];
      report.frozenBaselineCeilings = frozenBaseline || null;
      report.frozenBaselinePassed = !frozenBaseline || Object.entries(frozenBaseline)
        .every(([metric, ceiling]) => report[metric] <= ceiling);
      report.paperSceneStrictPassed = this.sceneIndex !== 1
        || (
          report.nrmse <= 0.40
          && report.trimmedNrmse99 <= 0.33
          && report.lowFrequencyScaleInvariantNrmse <= 0.24
          && report.p95Absolute <= 0.15
          && report.severeOverlitRatio <= 0.04
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
      triangles: this.scene?.geometry.triangleCount || 0,
      probes: [...this.probeCounts],
      rays: this.rayCount,
      hitRate: this.rayCount ? this.hitCount/this.rayCount : 0,
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
    $("audit-report").textContent = "Beginning deterministic 12-scene audit…";
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
        // The heightmap's wider visible-probe footprint takes longer to finish
        // the exact-count bootstrap. Keep the production tolerance unchanged
        // and sample its steady state, as the reference audit already does.
        warmup: i === 6 ? 192 : ([5, 11].includes(i) ? 96 : 48),
      });
      result.finalFrameRepeatability = await this.runFinalFrameRepeatabilityAudit({
        poses: 4,
        warmup: 48,
        holdFrames: 2,
      });
      result.continuousMotion = await this.runContinuousMotionAudit({
        frames: i === 1 ? 32 : 12,
        warmup: 48,
      });
      result.movingLightContinuousMotion = await this.runContinuousMotionAudit({
        frames: i === 1 ? 32 : 12,
        warmup: 48,
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
      if ([0, 1, 10, 11].includes(i)) {
        $("audit-title").textContent = `Comparing ${SCENE_INFO[i].short} with a path-traced reference`;
        result.pathTracedReference = await this.runPathTracedReferenceAudit({
          width: 64,
          height: 36,
          samples: 512,
          warmup: 96,
        });
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
    this.destroyed = true;
    this.running = false;
    for (const fn of this.cleanup) fn();
    for (const resource of [
      this.vertexBuffer, this.bvhNodeBuffer, this.triangleBuffer, this.frameBuffer,
      this.hashBuffer, this.stateBuffer, this.probeMetaBuffer, this.accumBuffer,
      this.coneBuffer, this.irradianceBuffer, this.queryResolveBuffer,
      this.irradianceAtlasWrite, this.irradianceAtlas, this.sunShadowDataBuffer,
      ...(this.sunShadowBuffers || []), ...(this.pointShadowBuffers || []),
      ...(this.passBuffers || []), ...(this.gbuffer || []), this.shadowTexture,
      this.pointShadowTexture,
      this.materialAtlas,
    ]) resource?.destroy?.();
  }
}

let started = false;
export async function startRadianceCascades() {
  if (started) return globalThis.__splitRC;
  started = true;
  const canvas = $("viewport");
  const renderer = new SplitRadianceCascades(canvas);
  globalThis.__splitRC = renderer;
  try {
    await renderer.initialize();
  } catch (error) {
    setStatus("WebGPU initialization failed", error.message || String(error), true);
    document.documentElement.dataset.webgpu = "failed";
    console.error("[Split RC]", error);
  }
  return renderer;
}

if (typeof document !== "undefined") {
  startRadianceCascades();
}
