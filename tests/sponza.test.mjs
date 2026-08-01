import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { gunzipSync } from "node:zlib";
import test from "node:test";

test("the packed official Sponza payload has a valid production BVH", async () => {
  const [compressed, atlas] = await Promise.all([
    readFile(new URL("../public/models/sponza.rcb", import.meta.url)),
    readFile(new URL("../public/models/sponza-atlas.webp", import.meta.url)),
  ]);
  const packed = gunzipSync(compressed);
  const header = new Uint32Array(packed.buffer, packed.byteOffset, 8);
  const bounds = new Float32Array(packed.buffer, packed.byteOffset + 32, 6);

  assert.equal(header[0], 0x31424352);
  assert.equal(header[1], 3);
  assert.equal(header[5], 262267 * 3);
  assert.equal(header[7], 262267);
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
  const triangleWords = new Uint32Array(
    packed.buffer,
    packed.byteOffset + triangleOffset,
    header[4],
  );
  const decodeOct = (word) => {
    const x0 = (word & 0xffff) / 65535 * 2 - 1;
    const y0 = (word >>> 16) / 65535 * 2 - 1;
    let x = x0;
    let y = y0;
    let z = 1 - Math.abs(x) - Math.abs(y);
    if (z < 0) {
      x = (1 - Math.abs(y0)) * Math.sign(x0 || 1);
      y = (1 - Math.abs(x0)) * Math.sign(y0 || 1);
    }
    const length = Math.hypot(x, y, z);
    return [x / length, y / length, z / length];
  };
  let emissiveTriangles = 0;
  for (let triangle = 0; triangle < header[7]; triangle++) {
    const base = triangle * 32;
    if (triangles[base + 16] <= 2) continue;
    emissiveTriangles++;
    const a = triangles.slice(base, base + 3);
    const b = triangles.slice(base + 4, base + 7);
    const c = triangles.slice(base + 8, base + 11);
    const ab = b.map((value, axis) => value - a[axis]);
    const ac = c.map((value, axis) => value - a[axis]);
    const face = [
      ab[1] * ac[2] - ab[2] * ac[1],
      ab[2] * ac[0] - ab[0] * ac[2],
      ab[0] * ac[1] - ab[1] * ac[0],
    ];
    const authored = decodeOct(triangleWords[base + 28]);
    const agreement = face.reduce((sum, value, axis) => sum + value * authored[axis], 0);
    assert.ok(agreement > 0, "emissive BVH winding must agree with its authored radiometric normal");
  }
  assert.equal(emissiveTriangles, 0, "official Sponza must not contain synthetic baked emitters");
});
