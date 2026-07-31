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
  assert.equal(header[1], 3);
  assert.equal(header[5], 262269 * 3);
  assert.equal(header[7], 262269);
  assert.ok(header[6] > 100_000, "expected a non-trivial packed BVH");
  assert.ok(compressed.byteLength < 18_000_000, "scene payload should remain web-deliverable");
  assert.ok(atlas.byteLength > 1_000_000, "expected the official base-color material atlas");
  assert.deepEqual([...atlas.subarray(0, 4)], [0x52, 0x49, 0x46, 0x46], "atlas should be a WebP/RIFF payload");
  assert.ok([...bounds].every(Number.isFinite));
  assert.ok(bounds[0] < bounds[3] && bounds[1] < bounds[4] && bounds[2] < bounds[5]);

  const triangleOffset = 64 + header[2] * 4 + header[3] * 4;
  const triangles = new Float32Array(
    packed.buffer,
    packed.byteOffset + triangleOffset,
    header[4],
  );
  let emissiveTriangles = 0;
  for (let triangle = 0; triangle < header[7]; triangle++) {
    const base = triangle * 32;
    if (triangles[base + 16] > 2) emissiveTriangles++;
  }
  assert.equal(emissiveTriangles, 2, "paper Sponza should contain the red area-emitter quad");
});
