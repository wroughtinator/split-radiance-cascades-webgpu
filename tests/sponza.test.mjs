import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { gunzipSync } from "node:zlib";
import test from "node:test";

test("the packed paper-scene Sponza payload has a valid production BVH", async () => {
  const [compressed, atlas] = await Promise.all([
    readFile(new URL("../public/models/sponza.rcb", import.meta.url)),
    readFile(new URL("../public/models/sponza-atlas.webp", import.meta.url)),
  ]);
  const packed = gunzipSync(compressed);
  const header = new Uint32Array(packed.buffer, packed.byteOffset, 8);
  const bounds = new Float32Array(packed.buffer, packed.byteOffset + 32, 6);

  assert.equal(header[0], 0x31424352);
  assert.equal(header[1], 2);
  assert.equal(header[5], 262267 * 3);
  assert.equal(header[7], 262267);
  assert.ok(header[6] > 100_000, "expected a non-trivial packed BVH");
  assert.ok(compressed.byteLength < 15_000_000, "scene payload should remain web-deliverable");
  assert.ok(atlas.byteLength > 1_000_000, "expected the official base-color material atlas");
  assert.deepEqual([...atlas.subarray(0, 4)], [0x52, 0x49, 0x46, 0x46], "atlas should be a WebP/RIFF payload");
  assert.ok([...bounds].every(Number.isFinite));
  assert.ok(bounds[0] < bounds[3] && bounds[1] < bounds[4] && bounds[2] < bounds[5]);
});
