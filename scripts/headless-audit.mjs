// Headless Chrome driver for the in-app Split RC autotest gates.
// Usage:
//   node scripts/headless-audit.mjs --url "http://localhost:8791/?autotest=dynamic-sponza" \
//     [--timeout 900000] [--out report.json] [--screenshot shot.png] [--settle 0] [--eval "expr"]
// Polls globalThis.__RC_TEST_REPORT__ and exits 0 iff report.passed.
import { spawn } from "node:child_process";
import { mkdtemp, readFile, writeFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";

const args = process.argv.slice(2);
const opt = (name, dflt) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? args[i + 1] : dflt;
};
const url = opt("url");
const timeoutMs = Number(opt("timeout", "900000"));
const outPath = opt("out", "");
const shotPath = opt("screenshot", "");
const settleMs = Number(opt("settle", "0"));
const evalExpr = opt("eval", "");
if (!url) {
  console.error("--url required");
  process.exit(2);
}

const chrome = "C:/Program Files/Google/Chrome/Application/chrome.exe";
const profile = await mkdtemp(path.join(tmpdir(), "rc-headless-"));
const proc = spawn(chrome, [
  "--headless=new",
  `--remote-debugging-port=0`,
  `--user-data-dir=${profile}`,
  "--no-first-run",
  "--no-default-browser-check",
  "--disable-extensions",
  "--disable-background-timer-throttling",
  "--disable-renderer-backgrounding",
  "--window-size=1280,800",
  "--enable-unsafe-webgpu",
  // Timestamp queries are behind Dawn's unsafe-apis toggle; several gates
  // (cornell-artifacts, dynamic-performance) require a real gpuMs reading.
  "--enable-dawn-features=allow_unsafe_apis",
  "about:blank",
], { stdio: ["ignore", "ignore", "pipe"] });
let stderrBuf = "";
proc.stderr.on("data", (d) => { stderrBuf += d; });

const cleanup = async (code) => {
  try { proc.kill(); } catch {}
  await new Promise((r) => setTimeout(r, 500));
  try { await rm(profile, { recursive: true, force: true }); } catch {}
  process.exit(code);
};

// Wait for the DevToolsActivePort file.
let port = 0;
for (let i = 0; i < 100; i++) {
  try {
    const txt = await readFile(path.join(profile, "DevToolsActivePort"), "utf8");
    port = Number(txt.split("\n")[0]);
    if (port) break;
  } catch {}
  await new Promise((r) => setTimeout(r, 200));
}
if (!port) {
  console.error("chrome did not expose a devtools port; stderr:\n" + stderrBuf.slice(-2000));
  await cleanup(3);
}

const versionInfo = await (await fetch(`http://127.0.0.1:${port}/json/version`)).json();
const ws = new WebSocket(versionInfo.webSocketDebuggerUrl);
await new Promise((res, rej) => { ws.onopen = res; ws.onerror = rej; });

let msgId = 0;
const pending = new Map();
const consoleLines = [];
let sessionId = null;
ws.onmessage = (ev) => {
  const msg = JSON.parse(ev.data);
  if (msg.id !== undefined && pending.has(msg.id)) {
    const { res, rej } = pending.get(msg.id);
    pending.delete(msg.id);
    if (msg.error) rej(new Error(JSON.stringify(msg.error)));
    else res(msg.result);
  } else if (msg.method === "Runtime.consoleAPICalled" && msg.sessionId === sessionId) {
    const text = (msg.params.args || [])
      .map((a) => a.value !== undefined ? String(a.value) : (a.description || a.type))
      .join(" ");
    consoleLines.push(`[${msg.params.type}] ${text}`);
  } else if (msg.method === "Runtime.exceptionThrown" && msg.sessionId === sessionId) {
    consoleLines.push(`[exception] ${msg.params.exceptionDetails?.text || ""} ${msg.params.exceptionDetails?.exception?.description || ""}`);
  }
};
const send = (method, params = {}, useSession = true) => new Promise((res, rej) => {
  const id = ++msgId;
  pending.set(id, { res, rej });
  const payload = { id, method, params };
  if (useSession && sessionId) payload.sessionId = sessionId;
  ws.send(JSON.stringify(payload));
});

const { targetId } = await send("Target.createTarget", { url: "about:blank" }, false);
const attach = await send("Target.attachToTarget", { targetId, flatten: true }, false);
sessionId = attach.sessionId;
await send("Runtime.enable");
await send("Page.enable");
await send("Emulation.setFocusEmulationEnabled", { enabled: true });
await send("Page.navigate", { url });

const evalJson = async (expr) => {
  const r = await send("Runtime.evaluate", {
    expression: expr,
    returnByValue: true,
    awaitPromise: true,
  });
  if (r.exceptionDetails) return { __error: r.exceptionDetails.text || "eval error" };
  return r.result?.value;
};

const start = Date.now();
let report = null;
let lastStatus = "";
while (Date.now() - start < timeoutMs) {
  await new Promise((r) => setTimeout(r, 2000));
  const v = await evalJson(
    "globalThis.__RC_TEST_REPORT__ ? JSON.stringify(globalThis.__RC_TEST_REPORT__) : (document.getElementById('audit-title')?.textContent || document.getElementById('status-line')?.textContent || 'pending')"
  );
  if (typeof v === "string" && v.startsWith("{")) {
    report = JSON.parse(v);
    break;
  }
  if (v !== lastStatus) {
    lastStatus = v;
    console.error(`[status +${((Date.now() - start) / 1000).toFixed(0)}s] ${typeof v === "string" ? v : JSON.stringify(v)}`);
  }
}

if (settleMs) await new Promise((r) => setTimeout(r, settleMs));
if (evalExpr) {
  const v = await evalJson(evalExpr);
  console.log("EVAL_RESULT " + JSON.stringify(v));
}
if (shotPath) {
  const shot = await send("Page.captureScreenshot", { format: "png" });
  await writeFile(shotPath, Buffer.from(shot.data, "base64"));
  console.error(`screenshot -> ${shotPath}`);
}

if (!report) {
  console.error("TIMEOUT waiting for __RC_TEST_REPORT__");
  console.error("last console lines:\n" + consoleLines.slice(-30).join("\n"));
  await cleanup(4);
}
const json = JSON.stringify(report, null, 2);
if (outPath) await writeFile(outPath, json);
else console.log(json);
console.error(`passed=${report.passed}`);
console.error("console tail:\n" + consoleLines.slice(-12).join("\n"));
await cleanup(report.passed ? 0 : 1);
