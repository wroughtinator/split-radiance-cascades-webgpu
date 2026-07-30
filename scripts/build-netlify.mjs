import { cp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const output = path.join(root, "netlify-dist");
if (path.basename(output) !== "netlify-dist" || path.dirname(output) !== root) {
  throw new Error(`Refusing to replace unexpected output path: ${output}`);
}

await rm(output, { recursive: true, force: true });
await mkdir(output, { recursive: true });
await cp(path.join(root, "public", "rc"), path.join(output, "rc"), { recursive: true });
await mkdir(path.join(output, "models"), { recursive: true });
await cp(path.join(root, "public", "models", "sponza.rcb"), path.join(output, "models", "sponza.rcb"));
await cp(path.join(root, "public", "models", "sponza-atlas.webp"), path.join(output, "models", "sponza-atlas.webp"));
await cp(path.join(root, "public", "models", "SPONZA-LICENSE.md"), path.join(output, "models", "SPONZA-LICENSE.md"));
await cp(path.join(root, "public", "favicon.svg"), path.join(output, "favicon.svg"));
await cp(path.join(root, "public", "og.png"), path.join(output, "og.png"));
await cp(path.join(root, "standalone", "index.html"), path.join(output, "index.html"));
await cp(path.join(root, "standalone", "index.html"), path.join(output, "404.html"));

const headers = `/*
  Cross-Origin-Opener-Policy: same-origin
  Cross-Origin-Embedder-Policy: require-corp
  Cross-Origin-Resource-Policy: same-origin
  X-Content-Type-Options: nosniff
  Referrer-Policy: strict-origin-when-cross-origin
  Permissions-Policy: camera=(), microphone=(), geolocation=()
  Content-Security-Policy: default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; worker-src 'self' blob:; object-src 'none'; base-uri 'self'; frame-ancestors 'none'

/rc/*
  Cache-Control: public, max-age=0, must-revalidate

/models/*
  Cache-Control: public, max-age=31536000, immutable

/og.png
  Cache-Control: public, max-age=3600
`;
await writeFile(path.join(output, "_headers"), headers, "utf8");
await writeFile(path.join(output, "_redirects"), "/* /index.html 200\n", "utf8");

const manifest = {
  name: "Split Radiance Cascades — WebGPU GI Lab",
  generatedAt: new Date().toISOString(),
  entrypoint: "index.html",
  deployment: "Upload this entire folder to Netlify Drop.",
  source: "https://arxiv.org/abs/2607.20384",
};
await writeFile(path.join(output, "build-manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`, "utf8");

const html = await readFile(path.join(output, "index.html"), "utf8");
for (const required of ["/rc/engine.js", "/rc/app.css", "/og.png", "id=\"viewport\"", "id=\"run-validation\""]) {
  if (!html.includes(required)) throw new Error(`Static build is missing ${required}`);
}
await readFile(path.join(output, "models", "sponza-atlas.webp"));
console.log(`Netlify bundle ready: ${output}`);
