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
  performance: { giDivisor: 8, pixelRatio: 0.72, shadow: 1024, raysPerSample: 1 },
  balanced: { giDivisor: 6, pixelRatio: 0.9, shadow: 1536, raysPerSample: 2 },
  quality: { giDivisor: 4, pixelRatio: 1, shadow: 2048, raysPerSample: 2 },
  ultra: { giDivisor: 3, pixelRatio: 1, shadow: 2048, raysPerSample: 3 },
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
    this.createPersistentResources();
    this.installUI();
    await this.loadScene(0);
    this.running = true;
    this.lastTime = performance.now();
    requestAnimationFrame((t) => this.frame(t));
    hideStatus();
    this.updateAdapterLabel();
    console.info("[Split RC] renderer-ready", this.adapterInfo());

    if (new URLSearchParams(location.search).has("autotest")) {
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
      entries: [{ binding: 0, visibility: SHADER.VERTEX | SHADER.FRAGMENT, buffer: { type: "uniform" } }],
    });
    this.rasterLayout = device.createPipelineLayout({ bindGroupLayouts: [this.frameLayout] });
    const vertexBuffers = [{
      arrayStride: 48,
      attributes: [
        { shaderLocation: 0, offset: 0, format: "float32x3" },
        { shaderLocation: 1, offset: 12, format: "float32x3" },
        { shaderLocation: 2, offset: 24, format: "float32x3" },
        { shaderLocation: 3, offset: 36, format: "float32x3" },
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
      splitRays: cp("splitRays"),
      merge: cp("mergeCascade"),
      prefilter: cp("prefilterIrradiance"),
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

  createPersistentResources() {
    const d = this.device;
    this.frameBuffer = createBuffer(d, "frame uniforms", 256, GPU.UNIFORM | GPU.COPY_DST);
    this.hashBuffer = createBuffer(d, "double-buffered sparse probe hash", K.totalHashSlots * K.hashFrames * 8, GPU.STORAGE | GPU.COPY_DST);
    this.stateBuffer = createBuffer(d, "probe counters and diagnostics", 32, GPU.STORAGE | GPU.COPY_SRC | GPU.COPY_DST);
    this.probeMetaBuffer = createBuffer(d, "sparse probe metadata", K.totalProbeMeta * 16, GPU.STORAGE | GPU.COPY_DST);
    this.accumBuffer = createBuffer(d, "fixed-point ray intervals", K.totalDirectionData * 5 * 4, GPU.STORAGE | GPU.COPY_DST);
    this.coneBuffer = createBuffer(d, "merged radiance cones", K.totalDirectionData * 16, GPU.STORAGE | GPU.COPY_DST);
    this.irradianceBuffer = createBuffer(d, "double-buffered 6x6 probe irradiance", K.probeCaps[0] * K.irradianceTexels * K.irradianceFrames * 16, GPU.STORAGE | GPU.COPY_DST);
    this.passBuffers = [0, 1, 2, 3].map((i) => makePassBuffer(d, i));
    this.shadowSampler = d.createSampler({ compare: "less-equal", minFilter: "linear", magFilter: "linear" });
    this.rasterBindGroup = d.createBindGroup({
      layout: this.frameLayout,
      entries: [{ binding: 0, resource: { buffer: this.frameBuffer } }],
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
    const seconds = (now - this.startTime) / 1000;
    const cameraPosition = this.cameraPosition(seconds);
    const view = mat4LookAt(cameraPosition, this.camera.target);
    const projection = mat4Perspective(Math.PI / 3, this.width / this.height, Math.max(0.03, this.scene.radius * 0.001), this.scene.radius * 5 + 100);
    const viewProjection = mat4Multiply(projection, view);
    const sunTime = this.animateLights ? seconds * this.sunSpeed : 0.7;
    const sunAngle = sunTime * 0.12 + this.sceneIndex * 0.61;
    const sunDirection = normalize3([Math.cos(sunAngle) * 0.7, -0.74, Math.sin(sunAngle) * 0.7]);
    const sunDistance = this.scene.radius * 1.8 + 30;
    const lightPosition = sub3(this.camera.target, mul3(sunDirection, sunDistance));
    const sunView = mat4LookAt(lightPosition, this.camera.target, Math.abs(sunDirection[1]) > 0.95 ? [0, 0, 1] : [0, 1, 0]);
    const extent = Math.max(8, this.scene.radius * 0.78);
    const sunProjection = mat4Ortho(-extent, extent, -extent, extent, 0.1, sunDistance * 2.3);
    const sunVP = mat4Multiply(sunProjection, sunView);
    const pointAngle = sunTime * 0.67 + this.sceneIndex;
    const pointPosition = add3(this.camera.target, [Math.cos(pointAngle) * this.scene.radius * 0.28, 2.5 + Math.sin(pointAngle * 1.7) * 1.8, Math.sin(pointAngle) * this.scene.radius * 0.28]);
    const hue = this.sceneIndex % 3;
    const pointColor = this.sceneIndex === 1
      ? [1.0, 0.45, 0.12]
      : hue === 0 ? [1.0, 0.18, 0.045] : hue === 1 ? [0.08, 0.5, 1.0] : [0.18, 1.0, 0.35];

    const u = new Float32Array(64);
    u.set(viewProjection, 0);
    u.set(sunVP, 16);
    u.set([...cameraPosition, 1], 32);
    u.set([...sunDirection, seconds], 36);
    u.set([1.0, 0.84, 0.63, this.scene.sun], 40);
    u.set([...pointPosition, this.scene.radius * 0.72], 44);
    u.set([...pointColor, 10.0 + this.sceneIndex * 0.7], 48);
    u.set([...this.scene.env, this.scene.baseSpacing], 52);
    u.set([this.width, this.height, this.giWidth, this.giHeight], 56);
    const frameParity = this.frameIndex & 1;
    const historyBlend = this.temporalStability && this.historyValid ? 0.88 : 0;
    u.set([this.indirectStrength, 1.0, this.debugMode, frameParity + historyBlend], 60);
    this.device.queue.writeBuffer(this.frameBuffer, 0, u);
  }

  render(now) {
    this.createSizedResources();
    this.updateUniforms(now);
    const d = this.device;
    const encoder = d.createCommandEncoder({ label: `Split RC frame ${this.frameIndex}` });
    const profile = this.timestampSupported && !this.profilePending && this.frameIndex % 45 === 0;

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
    for (let cascade = 1; cascade < 4; cascade++) {
      pass = encoder.beginComputePass({ label: `initialize cascade ${cascade}` });
      pass.setPipeline(this.computePipelines.initHigher);
      pass.setBindGroup(0, this.computeBindGroups[cascade]);
      pass.dispatchWorkgroups(Math.ceil(K.probeCaps[cascade - 1] / 64));
      pass.end();
    }

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
      results.push(result);
      $("audit-report").textContent = results.map((r) =>
        `${String(r.scene+1).padStart(2,"0")} ${r.name.padEnd(28)} ${r.fps.toFixed(0).padStart(3)} FPS  ${r.gpuMs == null ? "CPU timing" : `${r.gpuMs.toFixed(2)} ms GPU`}  ${r.triangles.toLocaleString()} tris  ${r.probes.reduce((a,b)=>a+b,0)} probes  overflow ${r.overflows}`
      ).join("\n");
    }
    const minFps = Math.min(...results.map((r) => r.fps));
    const maxGpu = Math.max(0, ...results.map((r) => r.gpuMs || 0));
    const failures = results.filter((r) => r.overflows || r.gpuError);
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
