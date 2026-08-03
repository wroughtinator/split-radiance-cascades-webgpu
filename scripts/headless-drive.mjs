// Drive the renderer headlessly with an arbitrary async expression evaluated
// in page context, with access to globalThis.__splitRC. Prints EVAL_RESULT.
// Usage: node scripts/headless-drive.mjs --script path.js [--timeout 300000]
import { spawn } from "node:child_process";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";

const args = process.argv.slice(2);
const opt = (name, dflt) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? args[i + 1] : dflt;
};
const scriptPath = opt("script");
const timeoutMs = Number(opt("timeout", "300000"));
const url = opt("url", "http://localhost:8791/");
const body = await readFile(scriptPath, "utf8");

const chrome = "C:/Program Files/Google/Chrome/Application/chrome.exe";
const profile = await mkdtemp(path.join(tmpdir(), "rc-drive-"));
const proc = spawn(chrome, [
  "--headless=new", "--remote-debugging-port=0", `--user-data-dir=${profile}`,
  "--no-first-run", "--no-default-browser-check", "--disable-extensions",
  "--disable-background-timer-throttling", "--disable-renderer-backgrounding",
  "--window-size=1280,800", "--enable-unsafe-webgpu",
  "--enable-dawn-features=allow_unsafe_apis",
  "about:blank",
], { stdio: ["ignore", "ignore", "pipe"] });
const cleanup = async (code) => {
  try { proc.kill(); } catch {}
  await new Promise((r) => setTimeout(r, 400));
  try { await rm(profile, { recursive: true, force: true }); } catch {}
  process.exit(code);
};
let port = 0;
for (let i = 0; i < 100 && !port; i++) {
  try { port = Number((await readFile(path.join(profile, "DevToolsActivePort"), "utf8")).split("\n")[0]); } catch {}
  if (!port) await new Promise((r) => setTimeout(r, 200));
}
if (!port) await cleanup(3);
const versionInfo = await (await fetch(`http://127.0.0.1:${port}/json/version`)).json();
const ws = new WebSocket(versionInfo.webSocketDebuggerUrl);
await new Promise((res, rej) => { ws.onopen = res; ws.onerror = rej; });
let msgId = 0; const pending = new Map(); let sessionId = null;
ws.onmessage = (ev) => {
  const msg = JSON.parse(ev.data);
  if (msg.id !== undefined && pending.has(msg.id)) {
    const { res, rej } = pending.get(msg.id); pending.delete(msg.id);
    if (msg.error) rej(new Error(JSON.stringify(msg.error))); else res(msg.result);
  }
};
const send = (method, params = {}, useSession = true) => new Promise((res, rej) => {
  const id = ++msgId; pending.set(id, { res, rej });
  const payload = { id, method, params };
  if (useSession && sessionId) payload.sessionId = sessionId;
  ws.send(JSON.stringify(payload));
});
const { targetId } = await send("Target.createTarget", { url: "about:blank" }, false);
sessionId = (await send("Target.attachToTarget", { targetId, flatten: true }, false)).sessionId;
await send("Runtime.enable");
await send("Page.enable");
await send("Page.navigate", { url });
// Wait for renderer
for (let i = 0; i < 150; i++) {
  const r = await send("Runtime.evaluate", { expression: "!!globalThis.__splitRC && !!globalThis.__splitRC.device", returnByValue: true });
  if (r.result?.value) break;
  await new Promise((rr) => setTimeout(rr, 400));
}
const wrapped = `(async () => { const R = globalThis.__splitRC; try { ${body} } catch (e) { return { error: String(e && e.stack || e) }; } })()`;
const result = await Promise.race([
  send("Runtime.evaluate", { expression: wrapped, returnByValue: true, awaitPromise: true, timeout: timeoutMs }),
  new Promise((_, rej) => setTimeout(() => rej(new Error("drive timeout")), timeoutMs)),
]).catch((e) => ({ result: { value: { error: String(e) } } }));
console.log("EVAL_RESULT " + JSON.stringify(result.result?.value ?? result.exceptionDetails?.text ?? null));
const shotPath = opt("screenshot", "");
if (shotPath) {
  const { writeFile } = await import("node:fs/promises");
  const shot = await send("Page.captureScreenshot", { format: "png" });
  await writeFile(shotPath, Buffer.from(shot.data, "base64"));
  console.error(`screenshot -> ${shotPath}`);
}
await cleanup(0);
