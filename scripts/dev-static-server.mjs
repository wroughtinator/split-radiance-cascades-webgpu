// Minimal static dev server: serves standalone/index.html at "/" and maps
// /rc/* -> public/rc/*, /models/* -> public/models/* so renderer edits are
// live on refresh without a build step.
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const port = Number(process.env.PORT || 8791);

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

createServer(async (req, res) => {
  try {
    const url = new URL(req.url, `http://localhost:${port}`);
    let filePath;
    if (url.pathname === "/" || url.pathname === "/index.html") {
      filePath = path.join(root, "standalone", "index.html");
    } else {
      const rel = decodeURIComponent(url.pathname).replace(/^\/+/, "");
      filePath = path.join(root, "public", rel);
    }
    if (!filePath.startsWith(root)) throw new Error("path escape");
    const data = await readFile(filePath);
    res.writeHead(200, {
      "content-type": types[path.extname(filePath)] || "application/octet-stream",
      "cache-control": "no-store",
    });
    res.end(data);
  } catch {
    res.writeHead(404, { "content-type": "text/plain" });
    res.end("not found");
  }
}, ).listen(port, () => console.log(`dev static server on http://localhost:${port}`));
