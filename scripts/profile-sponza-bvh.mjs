import { readFile } from "node:fs/promises";
import { gunzipSync } from "node:zlib";
import { decodeEqualArea, r2 } from "../public/rc/math.js";

const packed = gunzipSync(await readFile(new URL("../public/models/sponza.rcb", import.meta.url)));
const header = new Uint32Array(packed.buffer, packed.byteOffset, 8);
const vertexBytes = header[2] * 4;
const nodes = new Float32Array(packed.buffer, packed.byteOffset + 64 + vertexBytes, header[3]);
const nodeWords = new Uint32Array(nodes.buffer, nodes.byteOffset, nodes.length);
const triangleOffset = packed.byteOffset + 64 + vertexBytes + header[3] * 4;
const triangles = new Float32Array(packed.buffer, triangleOffset, header[4]);
const triangleStride = header[4] / header[7];

function rayBox(origin, inverse, node, limit) {
  const offset = node * 8;
  let near = 0;
  let far = limit;
  for (let axis = 0; axis < 3; axis++) {
    const a = (nodes[offset + axis] - origin[axis]) * inverse[axis];
    const b = (nodes[offset + 4 + axis] - origin[axis]) * inverse[axis];
    near = Math.max(near, Math.min(a, b));
    far = Math.min(far, Math.max(a, b));
  }
  return far >= near ? near : limit;
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
  const determinant = e1[0] * p[0] + e1[1] * p[1] + e1[2] * p[2];
  if (Math.abs(determinant) < 1e-8) return limit;
  const inverseDeterminant = 1 / determinant;
  const translated = [origin[0] - ax, origin[1] - ay, origin[2] - az];
  const u = (translated[0] * p[0] + translated[1] * p[1] + translated[2] * p[2]) * inverseDeterminant;
  if (u < 0 || u > 1) return limit;
  const q = [
    translated[1] * e1[2] - translated[2] * e1[1],
    translated[2] * e1[0] - translated[0] * e1[2],
    translated[0] * e1[1] - translated[1] * e1[0],
  ];
  const v = (direction[0] * q[0] + direction[1] * q[1] + direction[2] * q[2]) * inverseDeterminant;
  if (v < 0 || u + v > 1) return limit;
  const distance = (e2[0] * q[0] + e2[1] * q[1] + e2[2] * q[2]) * inverseDeterminant;
  return distance > 0.001 && distance < limit ? distance : limit;
}

function trace(origin, direction, maxDistance = 100) {
  const inverse = direction.map((value) => Math.sign(value || 1) / Math.max(1e-20, Math.abs(value)));
  const stack = [0];
  let distance = maxDistance;
  let nodeVisits = 0;
  let triangleTests = 0;
  while (stack.length) {
    const node = stack.pop();
    nodeVisits++;
    if (rayBox(origin, inverse, node, distance) >= distance) continue;
    const offset = node * 8;
    const left = nodeWords[offset + 3];
    const right = nodeWords[offset + 7];
    if (left & 0x80000000) {
      const first = left & 0x7fffffff;
      triangleTests += right;
      for (let triangle = 0; triangle < right; triangle++) {
        distance = rayTriangle(origin, direction, (first + triangle) * triangleStride, distance);
      }
    } else {
      const leftNear = rayBox(origin, inverse, left, distance);
      const rightNear = rayBox(origin, inverse, right, distance);
      if (leftNear < rightNear) {
        if (rightNear < distance) stack.push(right);
        if (leftNear < distance) stack.push(left);
      } else {
        if (leftNear < distance) stack.push(left);
        if (rightNear < distance) stack.push(right);
      }
    }
  }
  return { distance, nodeVisits, triangleTests };
}

const origins = [
  [-8, 8.5, -0.5],
  [0, 2.2, 0],
  [8, 3.5, 0],
  [-6, 2.2, 4],
];
const results = [];
for (let index = 0; index < 1024; index++) {
  results.push(trace(origins[index % origins.length], decodeEqualArea(r2(index)), 100));
}
const percentile = (values, p) => {
  values.sort((a, b) => a - b);
  return values[Math.min(values.length - 1, Math.floor((values.length - 1) * p))];
};
const visits = results.map((result) => result.nodeVisits);
const tests = results.map((result) => result.triangleTests);
console.log(JSON.stringify({
  rays: results.length,
  nodeVisits: { median: percentile(visits, 0.5), p95: percentile(visits, 0.95), max: Math.max(...visits) },
  triangleTests: { median: percentile(tests, 0.5), p95: percentile(tests, 0.95), max: Math.max(...tests) },
  hitRate: results.filter((result) => result.distance < 100).length / results.length,
}, null, 2));
