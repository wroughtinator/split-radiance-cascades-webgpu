import { spawn } from "node:child_process";
import { mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";

const options = new Map();
for (let index = 2; index < process.argv.length; index += 2) {
  options.set(process.argv[index], process.argv[index + 1]);
}

const url = options.get("--url");
if (!url) throw new Error("Usage: node scripts/chrome-webgpu-audit.mjs --url <url> [--output <png>] [--timeout <ms>]");
const output = resolve(options.get("--output") || "tmp/webgpu-audit.png");
const timeout = Number(options.get("--timeout") || 120_000);
const chromePath = options.get("--chrome") || "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe";
const port = Number(options.get("--port") || (9400 + Math.floor(Math.random() * 400)));
const width = Math.max(640, Number(options.get("--width") || 1280));
const height = Math.max(480, Number(options.get("--height") || 720));
const debugView = options.has("--debug-view") ? Number(options.get("--debug-view")) : null;
const frameStatsRequested = options.get("--frame-stats") === "1";
const parseVector = (name) => {
  const text = options.get(name);
  if (!text) return null;
  const values = text.split(",").map(Number);
  if (values.length !== 3 || values.some((value) => !Number.isFinite(value))) {
    throw new Error(`${name} must be three comma-separated finite numbers`);
  }
  return values;
};
const cameraPosition = parseVector("--camera");
const cameraTarget = parseVector("--target");
if ((cameraPosition === null) !== (cameraTarget === null)) {
  throw new Error("--camera and --target must be provided together");
}
const profile = resolve(options.get("--profile") || `tmp/chrome-webgpu-${port}`);
await mkdir(profile, { recursive: true });
await mkdir(dirname(output), { recursive: true });

const chrome = spawn(chromePath, [
  "--headless=new",
  "--enable-unsafe-webgpu",
  "--disable-gpu-sandbox",
  "--disable-software-rasterizer",
  "--use-angle=d3d11",
  `--remote-debugging-port=${port}`,
  `--user-data-dir=${profile}`,
  `--window-size=${width},${height}`,
  "--hide-scrollbars",
  "about:blank",
], { windowsHide: true, stdio: "ignore" });

const started = Date.now();
const wait = (milliseconds) => new Promise((resolveWait) => setTimeout(resolveWait, milliseconds));
const endpoint = `http://127.0.0.1:${port}`;
let browserVersion;
while (Date.now() - started < 15_000) {
  try {
    const response = await fetch(`${endpoint}/json/version`);
    if (response.ok) {
      browserVersion = await response.json();
      break;
    }
  } catch {}
  await wait(100);
}
if (!browserVersion) {
  chrome.kill();
  throw new Error("Chrome DevTools did not become ready");
}

const targetResponse = await fetch(`${endpoint}/json/new?${encodeURIComponent(url)}`, { method: "PUT" });
if (!targetResponse.ok) throw new Error(`Unable to open audit target (${targetResponse.status})`);
const target = await targetResponse.json();
const socket = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((resolveOpen, rejectOpen) => {
  socket.addEventListener("open", resolveOpen, { once: true });
  socket.addEventListener("error", rejectOpen, { once: true });
});

let nextId = 1;
const pending = new Map();
const consoleErrors = [];
socket.addEventListener("message", (event) => {
  const message = JSON.parse(event.data);
  if (message.id) {
    const handler = pending.get(message.id);
    if (handler) {
      pending.delete(message.id);
      if (message.error) handler.reject(new Error(message.error.message));
      else handler.resolve(message.result);
    }
    return;
  }
  if (message.method === "Runtime.consoleAPICalled" && ["error", "warning"].includes(message.params.type)) {
    consoleErrors.push(message.params.args.map((argument) => argument.value ?? argument.description ?? "").join(" "));
  }
});
const command = (method, params = {}) => new Promise((resolveCommand, rejectCommand) => {
  const id = nextId++;
  pending.set(id, { resolve: resolveCommand, reject: rejectCommand });
  socket.send(JSON.stringify({ id, method, params }));
});
const evaluate = async (expression) => {
  const result = await command("Runtime.evaluate", {
    expression,
    awaitPromise: true,
    returnByValue: true,
    userGesture: false,
  });
  if (result.exceptionDetails) {
    const detail = result.exceptionDetails.exception?.description
      || result.exceptionDetails.exception?.value
      || result.exceptionDetails.text
      || "Page evaluation failed";
    throw new Error(detail);
  }
  return result.result.value;
};

await command("Runtime.enable");
await command("Page.enable");
await command("Page.setLifecycleEventsEnabled", { enabled: true });

const wantsReport = new URL(url).searchParams.has("autotest");
let state;
let debugViewApplied = debugView === null;
let debugViewAppliedFrame = 0;
let cameraPoseApplied = cameraPosition === null;
while (Date.now() - started < timeout) {
  state = await evaluate(`(() => ({
    webgpu: document.documentElement?.dataset.webgpu || null,
    audit: document.documentElement?.dataset.audit || null,
    frameIndex: globalThis.__splitRC?.frameIndex || 0,
    running: !!globalThis.__splitRC?.running,
    reportReady: !!globalThis.__RC_TEST_REPORT__,
    status: document.getElementById('status-detail')?.textContent || null
  }))()`);
  if (state.webgpu === "failed") throw new Error(state.status || "WebGPU initialization failed");
  if (!cameraPoseApplied && state.running && state.frameIndex >= 90) {
    await evaluate(`(() => {
      globalThis.__splitRC.setCameraPose(
        ${JSON.stringify(cameraPosition)},${JSON.stringify(cameraTarget)}
      );
      globalThis.__splitRC.animateCamera = false;
    })()`);
    cameraPoseApplied = true;
  }
  // Pose/test query handlers run on a short startup timer and may select their
  // own diagnostic mode. Apply an explicit harness override only after that
  // initialization window, then render enough frames before capture.
  if (!debugViewApplied && state.running && state.frameIndex >= 120) {
    await evaluate(`(() => {
      globalThis.__splitRC.debugMode = ${JSON.stringify(debugView)};
      const select = document.getElementById('debug-view');
      if (select) select.value = ${JSON.stringify(String(debugView))};
    })()`);
    debugViewApplied = true;
    debugViewAppliedFrame = state.frameIndex;
  }
  const staticFrameReady = state.frameIndex >= Math.max(240, debugViewAppliedFrame + 60);
  if ((wantsReport && state.reportReady) || (!wantsReport && state.running && debugViewApplied && staticFrameReady)) break;
  await wait(250);
}
if (!state || (wantsReport ? !state.reportReady : state.frameIndex < 240)) {
  throw new Error(`Timed out waiting for the WebGPU audit: ${JSON.stringify(state)}`);
}

const audit = await evaluate(`(() => ({
  url: location.href,
  adapter: globalThis.__splitRC?.adapterInfo?.() || null,
  camera: globalThis.__splitRC?.camera ? {
    ...globalThis.__splitRC.camera,
    target: [...globalThis.__splitRC.camera.target]
  } : null,
  pose: globalThis.__splitRC?.cameraPose?.(globalThis.__splitRC?.testTimeOverride || 0) || null,
  metrics: globalThis.__splitRC?.metricsSnapshot?.() || null,
  passTimes: globalThis.__splitRC?.passTimes
    ? { ...globalThis.__splitRC.passTimes }
    : null,
  debugMode: globalThis.__splitRC?.debugMode ?? null,
  emissiveTriangles: globalThis.__splitRC?.scene?.geometry?.emissiveGeometry?.emissiveTriangleCount ?? null,
  report: globalThis.__RC_TEST_REPORT__ || null,
  webgpu: document.documentElement.dataset.webgpu || null,
  auditState: document.documentElement.dataset.audit || null
}))()`);
const frameStats = frameStatsRequested ? await evaluate(`(async () => {
  const frame = await globalThis.__splitRC.captureFinalFrame();
  const values = [];
  let maximum = 0;
  let maximumPixel = [0, 0];
  let maximumRgb = [0, 0, 0];
  let maximumWorld = [0, 0, 0];
  for (let y = 0; y < frame.height; y++) {
    const pixelRow = y * frame.bytesPerRow;
    const worldRow = y * (frame.worldBytesPerRow / 4);
    for (let x = 0; x < frame.width; x++) {
      if (frame.worldPixels[worldRow + x * 4 + 3] < 0.5) continue;
      const offset = pixelRow + x * 4;
      const luminance = frame.pixels[offset] * 0.2126
        + frame.pixels[offset + 1] * 0.7152
        + frame.pixels[offset + 2] * 0.0722;
      values.push(luminance);
      if (luminance > maximum) {
        maximum = luminance;
        maximumPixel = [x, y];
        maximumRgb = [
          frame.pixels[offset], frame.pixels[offset + 1], frame.pixels[offset + 2]
        ];
        maximumWorld = [
          frame.worldPixels[worldRow + x * 4],
          frame.worldPixels[worldRow + x * 4 + 1],
          frame.worldPixels[worldRow + x * 4 + 2],
        ];
      }
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
    p999LuminanceByte: percentile(0.999),
    maximumLuminanceByte: maximum,
    maximumPixel,
    maximumRgb,
    maximumWorld,
  };
})()`) : null;
const screenshot = await command("Page.captureScreenshot", { format: "png", fromSurface: true });
await writeFile(output, Buffer.from(screenshot.data, "base64"));

console.log(JSON.stringify({
  ...audit,
  frameStats,
  screenshot: output,
  consoleErrors,
}, null, 2));

try { await command("Browser.close"); } catch {}
socket.close();
chrome.kill();
