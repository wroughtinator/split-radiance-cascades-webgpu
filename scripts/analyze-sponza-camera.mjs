import { readFile } from "node:fs/promises";
import { gunzipSync } from "node:zlib";

const packed = gunzipSync(await readFile(new URL("../public/models/sponza.rcb", import.meta.url)));
const header = new Uint32Array(packed.buffer, packed.byteOffset, 8);
const vertexBytes = header[2] * 4;
const nodeBytes = header[3] * 4;
const nodes = new Float32Array(packed.buffer, packed.byteOffset + 64 + vertexBytes, header[3]);
const nodeWords = new Uint32Array(nodes.buffer, nodes.byteOffset, nodes.length);
const triangles = new Float32Array(
  packed.buffer,
  packed.byteOffset + 64 + vertexBytes + nodeBytes,
  header[4],
);

function rayBox(origin, inverse, offset, limit) {
  let near = 0;
  let far = limit;
  for (let axis = 0; axis < 3; axis++) {
    const a = (nodes[offset + axis] - origin[axis]) * inverse[axis];
    const b = (nodes[offset + 4 + axis] - origin[axis]) * inverse[axis];
    near = Math.max(near, Math.min(a, b));
    far = Math.min(far, Math.max(a, b));
  }
  return far >= near;
}

function rayTriangle(origin, direction, offset, limit) {
  const ax = triangles[offset], ay = triangles[offset + 1], az = triangles[offset + 2];
  const e1 = [triangles[offset + 4] - ax, triangles[offset + 5] - ay, triangles[offset + 6] - az];
  const e2 = [triangles[offset + 8] - ax, triangles[offset + 9] - ay, triangles[offset + 10] - az];
  const p = [
    direction[1] * e2[2] - direction[2] * e2[1],
    direction[2] * e2[0] - direction[0] * e2[2],
    direction[0] * e2[1] - direction[1] * e2[0],
  ];
  const det = e1[0] * p[0] + e1[1] * p[1] + e1[2] * p[2];
  if (Math.abs(det) < 1e-8) return limit;
  const inv = 1 / det;
  const tv = [origin[0] - ax, origin[1] - ay, origin[2] - az];
  const u = (tv[0] * p[0] + tv[1] * p[1] + tv[2] * p[2]) * inv;
  if (u < 0 || u > 1) return limit;
  const q = [
    tv[1] * e1[2] - tv[2] * e1[1],
    tv[2] * e1[0] - tv[0] * e1[2],
    tv[0] * e1[1] - tv[1] * e1[0],
  ];
  const v = (direction[0] * q[0] + direction[1] * q[1] + direction[2] * q[2]) * inv;
  if (v < 0 || u + v > 1) return limit;
  const t = (e2[0] * q[0] + e2[1] * q[1] + e2[2] * q[2]) * inv;
  return t > 0.001 && t < limit ? t : limit;
}

function trace(origin, direction, maxDistance = 100) {
  const inverse = direction.map((value) => Math.sign(value || 1) / Math.max(1e-8, Math.abs(value)));
  const stack = [0];
  let result = maxDistance;
  while (stack.length) {
    const node = stack.pop();
    const offset = node * 8;
    if (!rayBox(origin, inverse, offset, result)) continue;
    const left = nodeWords[offset + 3];
    const right = nodeWords[offset + 7];
    if (left & 0x80000000) {
      const first = left & 0x7fffffff;
      for (let i = 0; i < right; i++) result = rayTriangle(origin, direction, (first + i) * 28, result);
    } else {
      stack.push(left, right);
    }
  }
  return result;
}

const directions = [
  [1, 0, 0], [-1, 0, 0], [0, 1, 0],
  [0, -1, 0], [0, 0, 1], [0, 0, -1],
];
const candidates = [];
for (let y = 2; y <= 9; y += 0.5) {
  for (let x = -14; x <= 13; x += 0.5) {
    for (let z = -8; z <= 8; z += 0.5) {
      const distances = directions.map((direction) => trace([x, y, z], direction, 60));
      candidates.push({ position: [x, y, z], clearance: Math.min(...distances), distances });
    }
  }
}
candidates.sort((a, b) => b.clearance - a.clearance);
console.log(JSON.stringify(candidates.slice(0, 30), null, 2));
