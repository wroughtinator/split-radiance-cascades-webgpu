// Serve the production netlify-dist bundle as-is (with its _headers security
// policy applied) for local preview and tunnel publishing.
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)), "..", "netlify-dist",
);
const port = Number(process.env.PORT || 8899);

const types = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json",
  ".webp": "image/webp",
  ".png": "image/png",
  ".svg": "image/svg+xml",
  ".rcb": "application/octet-stream",
  ".md": "text/markdown; charset=utf-8",
};

let headerRules = [];
try {
  const raw = await readFile(path.join(root, "_headers"), "utf8");
  let current = null;
  for (const line of raw.split("\n")) {
    if (!line.trim()) continue;
    if (!line.startsWith(" ") && !line.startsWith("\t")) {
      current = { pattern: line.trim(), headers: {} };
      headerRules.push(current);
    } else if (current) {
      const i = line.indexOf(":");
      if (i > 0) current.headers[line.slice(0, i).trim()] = line.slice(i + 1).trim();
    }
  }
} catch {}

const matchRule = (p) => {
  const out = {};
  for (const rule of headerRules) {
    const pattern = rule.pattern;
    const ok = pattern === "/*"
      || (pattern.endsWith("/*") && p.startsWith(pattern.slice(0, -1)))
      || pattern === p;
    if (ok) Object.assign(out, rule.headers);
  }
  return out;
};

createServer(async (req, res) => {
  try {
    const url = new URL(req.url, `http://localhost:${port}`);
    let rel = decodeURIComponent(url.pathname).replace(/^\/+/, "");
    if (rel === "") rel = "index.html";
    let filePath = path.join(root, rel);
    if (!filePath.startsWith(root)) throw new Error("path escape");
    let data;
    try {
      data = await readFile(filePath);
    } catch {
      filePath = path.join(root, "404.html");
      data = await readFile(filePath);
    }
    res.writeHead(200, {
      "content-type": types[path.extname(filePath)] || "application/octet-stream",
      ...matchRule("/" + rel),
    });
    res.end(data);
  } catch {
    res.writeHead(500, { "content-type": "text/plain" });
    res.end("error");
  }
}).listen(port, () => console.log(`production bundle on http://localhost:${port}`));
