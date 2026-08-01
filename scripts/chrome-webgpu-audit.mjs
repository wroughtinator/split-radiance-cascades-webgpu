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
  "--window-size=1280,720",
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
  if (result.exceptionDetails) throw new Error(result.exceptionDetails.text || "Page evaluation failed");
  return result.result.value;
};

await command("Runtime.enable");
await command("Page.enable");
await command("Page.setLifecycleEventsEnabled", { enabled: true });

const wantsReport = new URL(url).searchParams.has("autotest");
let state;
while (Date.now() - started < timeout) {
  state = await evaluate(`(() => ({
    webgpu: document.documentElement.dataset.webgpu || null,
    audit: document.documentElement.dataset.audit || null,
    frameIndex: globalThis.__splitRC?.frameIndex || 0,
    running: !!globalThis.__splitRC?.running,
    reportReady: !!globalThis.__RC_TEST_REPORT__,
    status: document.getElementById('status-detail')?.textContent || null
  }))()`);
  if (state.webgpu === "failed") throw new Error(state.status || "WebGPU initialization failed");
  if ((wantsReport && state.reportReady) || (!wantsReport && state.running && state.frameIndex >= 240)) break;
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
  report: globalThis.__RC_TEST_REPORT__ || null,
  webgpu: document.documentElement.dataset.webgpu || null,
  auditState: document.documentElement.dataset.audit || null
}))()`);
const screenshot = await command("Page.captureScreenshot", { format: "png", fromSurface: true });
await writeFile(output, Buffer.from(screenshot.data, "base64"));

console.log(JSON.stringify({
  ...audit,
  screenshot: output,
  consoleErrors,
}, null, 2));

try { await command("Browser.close"); } catch {}
socket.close();
chrome.kill();
