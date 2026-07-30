import {
  add3, clamp, mat4LookAt, mat4Multiply, mat4Ortho,
  mat4Perspective, mul3, normalize3, sub3,
} from "./math.js";
import { createScene, SCENE_INFO } from "./scenes.js";
import { computeShader, finalShader, rasterShader, shaderConstants as K } from "./shaders.js";

const GPU = globalThis.GPUBufferUsage;
const TEX = globalThis.GPUTextureUsage;
const MAP = globalThis.GPUMapMode;
const SHADER = globalThis.GPUShaderStage;

const QUALITY = {
  performance: { giDivisor: 8, pixelRatio: 0.72, shadow: 1024, raysPerSample: 2 },
  balanced: { giDivisor: 6, pixelRatio: 0.9, shadow: 1536, raysPerSample: 12 },
  quality: { giDivisor: 4, pixelRatio: 1, shadow: 2048, raysPerSample: 14 },
  ultra: { giDivisor: 3, pixelRatio: 1, shadow: 2048, raysPerSample: 16 },
};

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

function makePassBuffer(device, cascade) {
  return createBuffer(device, `cascade-${cascade}-params`, 16, GPU.UNIFORM | GPU.COPY_DST, new Uint32Array([cascade, 0, 0, 0]));
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
    this.qualityName = "balanced";
    this.indirectStrength = 1;
    this.sunSpeed = 1;
    this.debugMode = 0;
    this.temporalStability = true;
    // The paper's primary evaluation path is single-bounce Split RC. Its
    // multibounce, rough-specular, and C(-1) experiments remain available as
    // explicit extensions, but are not mixed into the baseline by default.
    this.multibounce = false;
    this.roughSpecular = false;
    this.cMinusOne = false;
    this.historyValid = false;
    this.animateCamera = true;
    this.animateLights = true;
    this.running = false;
    this.destroyed = false;
    this.frameIndex = 0;
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
    this.loadingScene = null;
    this.loadToken = 0;
    this.mouse = { down: false, x: 0, y: 0 };
    this.camera = { azimuth: 0.75, elevation: 0.32, distance: 20, target: [0, 0, 0] };
    this.keys = new Set();
    this.cleanup = [];
    this.testTimeOverride = null;
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
      console.error("[Split RC] WebGPU validation error", event.error);
      this.lastGpuError = String(event.error?.message || event.error);
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
    await this.loadScene(0);
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
          repeatability: await this.runFinalFrameRepeatabilityAudit(),
          metrics: this.metricsSnapshot(),
        };
        report.passed = report.reference.passed && report.repeatability.passed;
        this.exposeTestReport(report);
      }, 200);
    } else if (automaticTest != null) {
      setTimeout(() => this.runValidation({ framesPerScene: 36 }), 200);
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

    this.frameLayout = device.createBindGroupLayout({
      label: "frame-uniform-layout",
      entries: [
        { binding: 0, visibility: SHADER.VERTEX | SHADER.FRAGMENT, buffer: { type: "uniform" } },
        { binding: 1, visibility: SHADER.FRAGMENT, texture: { sampleType: "float" } },
        { binding: 2, visibility: SHADER.FRAGMENT, sampler: { type: "filtering" } },
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
        targets: [{ format: "rgba8unorm" }, { format: "rgba16float" }, { format: "rgba16float" }],
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
        { binding: 12, visibility: SHADER.COMPUTE, texture: { sampleType: "float" } },
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
      initSecondary: cp("initSecondary"),
      initHigher: cp("initHigher"),
      canonicalize: cp("canonicalizeProbes"),
      countBase: cp("countBaseRays"),
      countSecondary: cp("countSecondaryRays"),
      countHigher: cp("countHigherRays"),
      assignOffsets: cp("assignRayOffsets"),
      splitRays: cp("splitRays"),
      splitSecondary: cp("splitSecondaryRays"),
      merge: cp("mergeCascade"),
      prefilter: cp("prefilterIrradiance"),
      validateReference: cp("validateReference"),
    };

    this.finalLayout = device.createBindGroupLayout({
      label: "Split RC composite layout",
      entries: [
        { binding: 0, visibility: SHADER.FRAGMENT, buffer: { type: "uniform" } },
        { binding: 1, visibility: SHADER.FRAGMENT, texture: { sampleType: "float" } },
        { binding: 2, visibility: SHADER.FRAGMENT, texture: { sampleType: "unfilterable-float" } },
        { binding: 3, visibility: SHADER.FRAGMENT, texture: { sampleType: "unfilterable-float" } },
        { binding: 4, visibility: SHADER.FRAGMENT, texture: { sampleType: "depth" } },
        { binding: 5, visibility: SHADER.FRAGMENT, sampler: { type: "comparison" } },
        { binding: 6, visibility: SHADER.FRAGMENT, buffer: { type: "storage" } },
        { binding: 7, visibility: SHADER.FRAGMENT, buffer: { type: "read-only-storage" } },
        { binding: 8, visibility: SHADER.FRAGMENT, buffer: { type: "read-only-storage" } },
        { binding: 9, visibility: SHADER.FRAGMENT, buffer: { type: "storage" } },
      ],
    });
    this.finalPipeline = device.createRenderPipeline({
      label: "Split RC final composite",
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.finalLayout] }),
      vertex: { module: finalModule, entryPoint: "fullscreenVS" },
      fragment: { module: finalModule, entryPoint: "finalFS", targets: [{ format: this.format }] },
      primitive: { topology: "triangle-list" },
    });
    const error = await device.popErrorScope();
    if (error) throw new Error(`WebGPU pipeline validation failed: ${error.message}`);
  }

  async createMaterialAtlas() {
    const response = await fetch("/models/sponza-atlas.webp");
    if (!response.ok) throw new Error(`Sponza material atlas request failed (${response.status}).`);
    const bitmap = await createImageBitmap(await response.blob(), { colorSpaceConversion: "default" });
    this.materialAtlas = this.device.createTexture({
      label: "official Sponza base-color atlas",
      size: [bitmap.width, bitmap.height],
      format: "rgba8unorm-srgb",
      usage: TEX.TEXTURE_BINDING | TEX.COPY_DST | TEX.RENDER_ATTACHMENT,
    });
    this.device.queue.copyExternalImageToTexture(
      { source: bitmap },
      { texture: this.materialAtlas },
      [bitmap.width, bitmap.height],
    );
    bitmap.close();
    this.materialSampler = this.device.createSampler({
      minFilter: "linear",
      magFilter: "linear",
      mipmapFilter: "linear",
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
    this.accumBuffer = createBuffer(d, "fixed-point ray intervals", K.totalDirectionData * 5 * 4, GPU.STORAGE | GPU.COPY_DST);
    this.coneBuffer = createBuffer(d, "merged radiance cones", K.totalDirectionData * 16, GPU.STORAGE | GPU.COPY_DST);
    this.irradianceBuffer = createBuffer(d, "double-buffered 6x6 probe irradiance", K.probeCaps[0] * K.irradianceTexels * K.irradianceFrames * 16, GPU.STORAGE | GPU.COPY_SRC | GPU.COPY_DST);
    this.passBuffers = [0, 1, 2, 3].map((i) => makePassBuffer(d, i));
    this.shadowSampler = d.createSampler({ compare: "less-equal", minFilter: "linear", magFilter: "linear" });
    this.rasterBindGroup = d.createBindGroup({
      layout: this.frameLayout,
      entries: [
        { binding: 0, resource: { buffer: this.frameBuffer } },
        { binding: 1, resource: this.materialAtlas.createView() },
        { binding: 2, resource: this.materialSampler },
      ],
    });
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
    this.historyValid = false;
  }

  createSizedResources() {
    const quality = QUALITY[this.qualityName];
    const rect = this.canvas.getBoundingClientRect();
    const width = Math.max(2, Math.floor(rect.width * Math.min(devicePixelRatio, 2) * quality.pixelRatio));
    const height = Math.max(2, Math.floor(rect.height * Math.min(devicePixelRatio, 2) * quality.pixelRatio));
    if (this.width === width && this.height === height && this.gbuffer) return;
    this.width = width;
    this.height = height;
    this.giWidth = Math.max(1, Math.ceil(width / quality.giDivisor));
    this.giHeight = Math.max(1, Math.ceil(height / quality.giDivisor));
    this.raysPerSample = quality.raysPerSample;
    this.device.queue.writeBuffer(this.passBuffers[0], 0, new Uint32Array([0, this.raysPerSample, 0, 0]));
    this.probeMetaBuffer?.destroy();
    const hitRecordVec4s = this.giWidth * this.giHeight * this.raysPerSample * 4;
    this.probeMetaBuffer = createBuffer(
      this.device,
      "sparse probe metadata and double-buffered secondary hit records",
      (K.totalProbeMeta + hitRecordVec4s) * 16,
      GPU.STORAGE | GPU.COPY_DST,
    );
    this.historyValid = false;
    this.canvas.width = width;
    this.canvas.height = height;
    for (const texture of this.gbuffer || []) texture.destroy();
    this.shadowTexture?.destroy();
    const d = this.device;
    const texture = (label, format, usage = TEX.RENDER_ATTACHMENT | TEX.TEXTURE_BINDING) =>
      d.createTexture({ label, size: [width, height], format, usage });
    this.albedoTexture = texture("G-buffer albedo", "rgba8unorm");
    this.normalTexture = texture("G-buffer normal", "rgba16float");
    this.worldTexture = texture("G-buffer world position", "rgba16float");
    this.depthTexture = texture("G-buffer depth", "depth24plus", TEX.RENDER_ATTACHMENT);
    this.gbuffer = [this.albedoTexture, this.normalTexture, this.worldTexture, this.depthTexture];
    this.shadowTexture = d.createTexture({
      label: "sun shadow map",
      size: [quality.shadow, quality.shadow],
      format: "depth32float",
      usage: TEX.RENDER_ATTACHMENT | TEX.TEXTURE_BINDING,
    });
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
      { binding: 12, resource: this.materialAtlas.createView() },
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
        { binding: 4, resource: this.shadowTexture.createView() },
        { binding: 5, resource: this.shadowSampler },
        { binding: 6, resource: { buffer: this.hashBuffer } },
        { binding: 7, resource: { buffer: this.irradianceBuffer } },
        { binding: 8, resource: { buffer: this.coneBuffer } },
        { binding: 9, resource: { buffer: this.accumBuffer } },
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

  cameraPosition(time) {
    let azimuth = this.camera.azimuth;
    const pathAmplitude = this.sceneIndex === 1 ? 0.045 : 0.28;
    if (this.animateCamera) azimuth += Math.sin(time * 0.16 + this.sceneIndex * 0.37) * pathAmplitude;
    const distance = this.camera.distance;
    const ce = Math.cos(this.camera.elevation);
    return add3(this.camera.target, [
      Math.cos(azimuth) * ce * distance,
      Math.sin(this.camera.elevation) * distance,
      Math.sin(azimuth) * ce * distance,
    ]);
  }

  updateUniforms(now) {
    const seconds = this.testTimeOverride ?? (now - this.startTime) / 1000;
    const cameraPosition = this.cameraPosition(seconds);
    const view = mat4LookAt(cameraPosition, this.camera.target);
    const projection = mat4Perspective(Math.PI / 3, this.width / this.height, Math.max(0.03, this.scene.radius * 0.001), this.scene.radius * 5 + 100);
    const viewProjection = mat4Multiply(projection, view);
    const sunTime = this.animateLights ? seconds * this.sunSpeed : 0.7;
    const sunAngle = sunTime * 0.12 + this.sceneIndex * 0.61;
    const sunHorizontal = this.sceneIndex === 1 ? 0.28 : 0.7;
    const sunHeight = this.sceneIndex === 1 ? -0.96 : -0.74;
    const sunDirection = normalize3([
      Math.cos(sunAngle) * sunHorizontal,
      sunHeight,
      Math.sin(sunAngle) * sunHorizontal,
    ]);
    const sunDistance = this.scene.radius * 1.8 + 30;
    const lightPosition = sub3(this.camera.target, mul3(sunDirection, sunDistance));
    const sunView = mat4LookAt(lightPosition, this.camera.target, Math.abs(sunDirection[1]) > 0.95 ? [0, 0, 1] : [0, 1, 0]);
    // Cover the full scene bounding sphere. A tighter crop produced bright
    // unshadowed wedges whenever a camera-visible corner left the sun frustum.
    const extent = Math.max(10, this.scene.radius * 1.18);
    const sunProjection = mat4Ortho(-extent, extent, -extent, extent, 0.1, sunDistance * 2.3);
    const sunVP = mat4Multiply(sunProjection, sunView);
    const pointAngle = sunTime * 0.67 + this.sceneIndex;
    const pointPosition = add3(this.camera.target, [Math.cos(pointAngle) * this.scene.radius * 0.28, 2.5 + Math.sin(pointAngle * 1.7) * 1.8, Math.sin(pointAngle) * this.scene.radius * 0.28]);
    const hue = this.sceneIndex % 3;
    const pointColor = this.sceneIndex === 1
      ? [1.0, 1.0, 1.0]
      : hue === 0 ? [1.0, 0.18, 0.045] : hue === 1 ? [0.08, 0.5, 1.0] : [0.18, 1.0, 0.35];
    const sunColor = this.sceneIndex === 1 ? [1.0, 0.98, 0.92] : [1.0, 0.84, 0.63];
    const pointIntensity = this.sceneIndex === 1 ? 0.0 : 10.0 + this.sceneIndex * 0.7;

    const u = new Float32Array(64);
    u.set(viewProjection, 0);
    u.set(sunVP, 16);
    const featureFlags = (this.multibounce ? 1 : 0)
      | (this.roughSpecular ? 2 : 0)
      | (this.cMinusOne ? 4 : 0)
      | (this.temporalStability ? 8 : 0);
    u.set([...cameraPosition, featureFlags], 32);
    u.set([...sunDirection, seconds], 36);
    u.set([...sunColor, this.scene.sun], 40);
    u.set([...pointPosition, this.scene.radius * 0.72], 44);
    u.set([...pointColor, pointIntensity], 48);
    u.set([...this.scene.env, this.scene.baseSpacing], 52);
    u.set([this.width, this.height, this.giWidth, this.giHeight], 56);
    const frameParity = this.frameIndex & 1;
    // A long world-space half-life removes the last allocator/visibility
    // shimmer under camera motion. Exact-key rejection still makes
    // disocclusions immediate, while animated lighting remains smooth.
    const historyBlend = this.temporalStability && this.historyValid ? 0.98 : 0;
    const exposure = this.sceneIndex === 1 ? 1.55 : 1.0;
    u.set([this.indirectStrength, exposure, this.debugMode, frameParity + historyBlend], 60);
    this.device.queue.writeBuffer(this.frameBuffer, 0, u);
  }

  render(now) {
    this.createSizedResources();
    this.updateUniforms(now);
    const d = this.device;
    const encoder = d.createCommandEncoder({ label: `Split RC frame ${this.frameIndex}` });
    const profile = this.timestampSupported && !this.profilePending && this.frameIndex % 45 === 0;
    const captureJob = this.captureRequest;
    this.captureRequest = null;

    const shadow = encoder.beginRenderPass({
      label: "sun shadow map",
      colorAttachments: [],
      depthStencilAttachment: { view: this.shadowTexture.createView(), depthClearValue: 1, depthLoadOp: "clear", depthStoreOp: "store" },
      timestampWrites: profile ? { querySet: this.querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } : undefined,
    });
    shadow.setPipeline(this.shadowPipeline);
    shadow.setBindGroup(0, this.rasterBindGroup);
    shadow.setVertexBuffer(0, this.vertexBuffer);
    shadow.draw(this.scene.geometry.vertexCount);
    shadow.end();

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
    gbuffer.end();

    encoder.clearBuffer(this.stateBuffer);
    encoder.clearBuffer(this.accumBuffer);
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
    pass.dispatchWorkgroups(Math.ceil(this.giWidth / 8), Math.ceil(this.giHeight / 8));
    pass.end();
    pass = encoder.beginComputePass({ label: "initialize secondary multibounce probes" });
    pass.setPipeline(this.computePipelines.initSecondary);
    pass.setBindGroup(0, this.computeBindGroups[0]);
    pass.dispatchWorkgroups(Math.ceil(this.giWidth / 8), Math.ceil(this.giHeight / 8), this.raysPerSample);
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
    pass = encoder.beginComputePass({ label: "count deterministic secondary rays" });
    pass.setPipeline(this.computePipelines.countSecondary);
    pass.setBindGroup(0, this.computeBindGroups[0]);
    pass.dispatchWorkgroups(Math.ceil(this.giWidth / 8), Math.ceil(this.giHeight / 8), this.raysPerSample);
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

    pass = encoder.beginComputePass({ label: "trace and split surface rays" });
    pass.setPipeline(this.computePipelines.splitRays);
    pass.setBindGroup(0, this.computeBindGroups[0]);
    pass.dispatchWorkgroups(Math.ceil(this.giWidth / 8), Math.ceil(this.giHeight / 8), this.raysPerSample);
    pass.end();
    pass = encoder.beginComputePass({ label: "trace secondary multibounce rays" });
    pass.setPipeline(this.computePipelines.splitSecondary);
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
    pass.dispatchWorkgroups(Math.ceil(K.probeCaps[0] * 36 / 64));
    pass.end();

    const finalPass = encoder.beginRenderPass({
      label: "tone-mapped final composite",
      colorAttachments: [{ view: this.context.getCurrentTexture().createView(), clearValue: [0,0,0,1], loadOp: "clear", storeOp: "store" }],
      timestampWrites: profile ? { querySet: this.querySet, beginningOfPassWriteIndex: 6, endOfPassWriteIndex: 7 } : undefined,
    });
    finalPass.setPipeline(this.finalPipeline);
    finalPass.setBindGroup(0, this.finalBindGroup);
    finalPass.draw(3);
    finalPass.end();

    let captureResources;
    if (captureJob) {
      const bytesPerRow = Math.ceil(this.width * 4 / 256) * 256;
      const texture = d.createTexture({
        label: "final-frame audit target",
        size: [this.width, this.height],
        format: this.format,
        usage: TEX.RENDER_ATTACHMENT | TEX.COPY_SRC,
      });
      const buffer = createBuffer(
        d,
        "final-frame audit readback",
        bytesPerRow * this.height,
        GPU.COPY_DST | GPU.MAP_READ,
      );
      const auditPass = encoder.beginRenderPass({
        label: "deterministic final-frame audit composite",
        colorAttachments: [{
          view: texture.createView(),
          clearValue: [0, 0, 0, 1],
          loadOp: "clear",
          storeOp: "store",
        }],
      });
      auditPass.setPipeline(this.finalPipeline);
      auditPass.setBindGroup(0, this.finalBindGroup);
      auditPass.draw(3);
      auditPass.end();
      encoder.copyTextureToBuffer(
        { texture },
        { buffer, bytesPerRow, rowsPerImage: this.height },
        [this.width, this.height],
      );
      captureResources = { ...captureJob, texture, buffer, bytesPerRow, width: this.width, height: this.height };
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
      await job.buffer.mapAsync(MAP.READ);
      job.resolve({
        width: job.width,
        height: job.height,
        bytesPerRow: job.bytesPerRow,
        pixels: new Uint8Array(job.buffer.getMappedRange().slice(0)),
      });
    } catch (error) {
      job.reject(error);
    } finally {
      job.buffer.destroy();
      job.texture.destroy();
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
      this.overflowCount = s[6];
    } catch (error) {
      console.warn("[Split RC] diagnostic readback failed", error);
    } finally {
      buffer.destroy();
      this.statusPending = false;
    }
  }

  frame(now) {
    if (!this.running || this.destroyed) return;
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
    if ($("gi-resolution")) $("gi-resolution").textContent = `${this.giWidth}×${this.giHeight} · ${this.raysPerSample} spp`;
    const pass = this.passTimes;
    for (const name of ["frame","geometry","gi","composite"]) {
      const value = pass[name] || 0;
      if ($(`pass-${name}`)) $(`pass-${name}`).textContent = value ? `${value.toFixed(2)} ms` : "sampling";
      if ($(`bar-${name}`)) $(`bar-${name}`).style.width = `${clamp(value / 22 * 100, 2, 100)}%`;
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
    on($("animate-lights"), "change", (e) => { this.animateLights = e.target.checked; });
    on($("temporal-stability"), "change", (e) => {
      this.temporalStability = e.target.checked;
      this.resetProbeHistory();
    });
    on($("multibounce"), "change", (e) => {
      this.multibounce = e.target.checked;
      this.resetProbeHistory();
    });
    on($("rough-specular"), "change", (e) => { this.roughSpecular = e.target.checked; });
    on($("c-minus-one"), "change", (e) => {
      this.cMinusOne = e.target.checked;
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
    await this.device.queue.onSubmittedWorkDone();
    const hashBytes = K.totalHashSlots * K.hashFrames * 8;
    const irradianceBytes = K.probeCaps[0] * K.irradianceTexels * K.irradianceFrames * 16;
    const [hashData, irradianceData] = await Promise.all([
      this.readGpuBuffer(this.hashBuffer, hashBytes, "stability hash readback"),
      this.readGpuBuffer(this.irradianceBuffer, irradianceBytes, "stability irradiance readback"),
    ]);
    const hash = new Uint32Array(hashData);
    const fields = new Float32Array(irradianceData);
    const current = (this.frameIndex - 1) & 1;
    const previous = 1 - current;
    const frameMap = (frame) => {
      const map = new Map();
      const base = frame * K.totalHashSlots * 2;
      for (let slot = 0; slot < K.hashSizes[0]; slot++) {
        const key = hash[base + slot * 2];
        const index = hash[base + slot * 2 + 1];
        if (key !== 0xffffffff && (key & 0x40000000) === 0 && index < K.probeCaps[0]) map.set(key, index);
      }
      return map;
    };
    const currentMap = frameMap(current);
    const previousMap = frameMap(previous);
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
      matchedProbes: relative.length,
      medianRelative: percentile(relative, 0.5),
      p95Relative: percentile(relative, 0.95),
      p99Relative: percentile(relative, 0.99),
      p95Absolute: percentile(absolute, 0.95),
      maxAbsolute: absolute.at(-1) ?? Infinity,
    };
  }

  async runMotionStabilityAudit({ samples = 5, interval = 4, warmup = 32 } = {}) {
    const previousCamera = this.animateCamera;
    const previousLights = this.animateLights;
    this.animateCamera = true;
    this.animateLights = false;
    await this.waitFrames(warmup);
    const measurements = [];
    for (let i = 0; i < samples; i++) {
      await this.waitFrames(interval);
      measurements.push(await this.measureWorldProbeStability());
    }
    this.animateCamera = previousCamera;
    this.animateLights = previousLights;
    const valid = measurements.filter((m) => Number.isFinite(m.p95Relative) && m.matchedProbes >= 16);
    const maximum = (name) => valid.length ? Math.max(...valid.map((m) => m[name])) : Infinity;
    const minimum = (name) => valid.length ? Math.min(...valid.map((m) => m[name])) : 0;
    return {
      samples: valid.length,
      matchedProbesMin: minimum("matchedProbes"),
      p95RelativeMax: maximum("p95Relative"),
      p99RelativeMax: maximum("p99Relative"),
      p95AbsoluteMax: maximum("p95Absolute"),
      passed: valid.length === samples && maximum("p95Relative") <= 0.015 && maximum("p95Absolute") <= 0.012,
      measurements,
    };
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
    return {
      p95ByteDelta: percentile(0.95),
      p99ByteDelta: percentile(0.99),
      maxByteDelta: differences.at(-1) || 0,
      rmseByteDelta: rmse,
      changedChannelRatio: changed / Math.max(1, differences.length),
      passed: percentile(0.95) <= 1
        && percentile(0.99) <= 2
        && rmse <= 0.75
        && (differences.at(-1) || 0) <= 48,
    };
  }

  async runFinalFrameRepeatabilityAudit({ poses = 6, warmup = 64, holdFrames = 2 } = {}) {
    const saved = {
      animateCamera: this.animateCamera,
      animateLights: this.animateLights,
      temporalStability: this.temporalStability,
      multibounce: this.multibounce,
      roughSpecular: this.roughSpecular,
      cMinusOne: this.cMinusOne,
      testTimeOverride: this.testTimeOverride,
    };
    try {
      this.animateCamera = true;
      this.animateLights = false;
      this.temporalStability = true;
      this.multibounce = false;
      this.roughSpecular = false;
      this.cMinusOne = false;
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

  async runPathTracedReferenceAudit({ width = 48, height = 27, samples = 128, warmup = 64 } = {}) {
    const saved = {
      animateCamera: this.animateCamera,
      animateLights: this.animateLights,
      temporalStability: this.temporalStability,
      multibounce: this.multibounce,
      roughSpecular: this.roughSpecular,
      cMinusOne: this.cMinusOne,
      testTimeOverride: this.testTimeOverride,
    };
    let auditBuffer;
    let auditPassBuffer;
    try {
      this.animateCamera = false;
      this.animateLights = false;
      this.temporalStability = true;
      this.multibounce = false;
      this.roughSpecular = false;
      this.cMinusOne = false;
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
        16,
        GPU.UNIFORM | GPU.COPY_DST,
        new Uint32Array([0, samples, width, height]),
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
          { binding: 12, resource: this.materialAtlas.createView() },
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
      const pairs = [];
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
        pairs.push({ reference, current });
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
      const report = {
        resolution: [width, height],
        samples,
        activePixels,
        nrmse,
        optimalEnergyScale: optimalScale,
        scaleInvariantNrmse,
        meanReferenceLuminance: referenceLuminance / Math.max(1, activePixels),
        meanSplitRCLuminance: currentLuminance / Math.max(1, activePixels),
        meanSignedLuminanceBias: signedBias / Math.max(1, activePixels),
        medianRelative: percentile(relative, 0.5),
        p95Relative: percentile(relative, 0.95),
        p95Absolute: percentile(absolute, 0.95),
      };
      report.passed = activePixels >= 32
        && report.nrmse <= 0.51
        && report.scaleInvariantNrmse <= 0.4
        && report.p95Absolute <= 0.2
        && Math.abs(report.meanSignedLuminanceBias) <= 0.1;
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
    console.info("[Split RC] validation-complete", report);
  }

  async runValidation({ framesPerScene = 72 } = {}) {
    const card = $("audit-card");
    card.hidden = false;
    $("audit-report").textContent = "Beginning deterministic 10-scene audit…";
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
      result.motionStability = await this.runMotionStabilityAudit({ samples: 4, interval: 3, warmup: 32 });
      result.finalFrameRepeatability = await this.runFinalFrameRepeatabilityAudit({
        poses: 4,
        warmup: 48,
        holdFrames: 2,
      });
      if (i === 1) {
        $("audit-title").textContent = "Comparing Sponza with a path-traced reference";
        result.pathTracedReference = await this.runPathTracedReferenceAudit({
          width: 48,
          height: 27,
          samples: 128,
          warmup: 64,
        });
      }
      results.push(result);
      $("audit-report").textContent = results.map((r) =>
        `${String(r.scene+1).padStart(2,"0")} ${r.name.padEnd(28)} ${r.fps.toFixed(0).padStart(3)} FPS  ${r.gpuMs == null ? "CPU timing" : `${r.gpuMs.toFixed(2)} ms GPU`}  ${r.triangles.toLocaleString()} tris  world jitter p95 ${(r.motionStability.p95RelativeMax*100).toFixed(2)}%  framebuffer p95 ${r.finalFrameRepeatability.p95ByteDeltaMax.toFixed(0)}/255${r.pathTracedReference ? `  reference NRMSE ${(r.pathTracedReference.nrmse*100).toFixed(1)}%` : ""}  overflow ${r.overflows}`
      ).join("\n");
    }
    const minFps = Math.min(...results.map((r) => r.fps));
    const maxGpu = Math.max(0, ...results.map((r) => r.gpuMs || 0));
    const failures = results.filter((r) =>
      r.overflows
      || r.gpuError
      || !r.motionStability.passed
      || !r.finalFrameRepeatability.passed
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
      ...(this.passBuffers || []), ...(this.gbuffer || []), this.shadowTexture,
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
