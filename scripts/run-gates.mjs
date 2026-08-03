// Sequentially run a list of autotest gates headlessly and save JSON reports.
// Usage: node scripts/run-gates.mjs <outDir> <gate1> <gate2> ...
import { spawn } from "node:child_process";
import path from "node:path";
import { mkdir } from "node:fs/promises";

const [outDir, ...gates] = process.argv.slice(2);
await mkdir(outDir, { recursive: true });
const results = [];
for (const gate of gates) {
  const out = path.join(outDir, `${gate}.json`);
  const started = Date.now();
  const code = await new Promise((resolve) => {
    const child = spawn(process.execPath, [
      "scripts/headless-audit.mjs",
      "--url", `http://localhost:8791/?autotest=${gate}`,
      "--timeout", "1500000",
      "--out", out,
    ], { stdio: ["ignore", "ignore", "inherit"] });
    child.on("exit", resolve);
  });
  const seconds = ((Date.now() - started) / 1000).toFixed(0);
  results.push({ gate, code, seconds });
  console.log(`GATE ${gate}: exit=${code} (${seconds}s)`);
}
console.log("ALL DONE");
console.log(JSON.stringify(results));
