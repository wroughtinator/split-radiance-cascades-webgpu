import assert from "node:assert/strict";
import test from "node:test";
import {
  CASCADE_DIRECTIONS, buildBVH, decodeEqualArea, encodeEqualArea,
  intersectTriangle, nearestProbe, normalize3, packProbeKey, r2,
} from "../public/rc/math.js";

test("paper cascade direction counts branch by four", () => {
  assert.deepEqual(CASCADE_DIRECTIONS, [32, 128, 512, 2048]);
  for (let i = 1; i < CASCADE_DIRECTIONS.length; i++) {
    assert.equal(CASCADE_DIRECTIONS[i] / CASCADE_DIRECTIONS[i - 1], 4);
  }
});

test("equal-area mapping round-trips normalized directions", () => {
  for (let i = 0; i < 512; i++) {
    const direction = normalize3([
      Math.sin(i * 1.713),
      Math.cos(i * 0.927),
      Math.sin(i * 0.337 + 1.2),
    ]);
    const decoded = decodeEqualArea(encodeEqualArea(direction));
    const dot = direction[0] * decoded[0] + direction[1] * decoded[1] + direction[2] * decoded[2];
    assert.ok(dot > 0.999999, `direction ${i} round-trip dot ${dot}`);
  }
});

test("R2 sequence remains in the unit square and avoids repeats", () => {
  const points = new Set();
  for (let i = 0; i < 4096; i++) {
    const [u, v] = r2(i);
    assert.ok(u >= 0 && u < 1 && v >= 0 && v < 1);
    points.add(`${u.toFixed(9)},${v.toFixed(9)}`);
  }
  assert.equal(points.size, 4096);
});

test("probe snapping uses paper half-cell offsets and stable keys", () => {
  assert.deepEqual(nearestProbe([0.1, -0.1, 1.9], 1), [0.5, -0.5, 1.5]);
  assert.equal(packProbeKey([0.1, -0.1, 1.9], 1, 0), packProbeKey([0.4, -0.4, 1.6], 1, 0));
  assert.notEqual(packProbeKey([0.1, -0.1, 1.9], 1, 0), packProbeKey([0.1, -0.1, 1.9], 1, 1));
});

test("BVH construction preserves ray/triangle intersections", () => {
  const triangle = { a: [-1, 0, -1], b: [1, 0, -1], c: [0, 0, 1], albedo: [1,1,1], emissive: [0,0,0] };
  const bvh = buildBVH([triangle]);
  assert.equal(bvh.triangleCount, 1);
  assert.equal(bvh.nodeCount, 1);
  assert.ok(Math.abs(intersectTriangle([0, 2, 0], [0, -1, 0], triangle) - 2) < 1e-6);
  assert.equal(intersectTriangle([3, 2, 0], [0, -1, 0], triangle), Infinity);
});
