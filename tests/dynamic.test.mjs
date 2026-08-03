import assert from "node:assert/strict";
import test from "node:test";
import { conservativeRigidSweep, DynamicScene } from "../public/rc/dynamic.js";
import { Geometry } from "../public/rc/scenes.js";

function staticFixture() {
  const geometry = new Geometry();
  geometry.box([0, 0, 0], [2, 2, 2]);
  return geometry.finish();
}

test("dynamic Sponza stress set uses reusable BLASes and bounded TLAS uploads", () => {
  const dynamic = new DynamicScene(1, staticFixture());
  assert.equal(dynamic.instanceCount, 48);
  assert.ok(dynamic.emissiveInstanceCount >= 6);
  assert.ok(dynamic.dynamicBlasTriangleCount > 100);
  assert.equal(dynamic.frameInfo()[0], dynamic.tlasNodeOffset);
  const immutableRasterBytes = dynamic.rasterVertices.byteLength;
  dynamic.rasterDirty = false;
  dynamic.update(1 / 60);
  assert.equal(dynamic.rasterDirty, false, "rigid animation must not stream raster vertices");
  assert.equal(dynamic.rasterVertices.byteLength, immutableRasterBytes);
  assert.equal(dynamic.frameInfo()[3], dynamic.sweptTlasNodeOffset);
  const perFrameUpload = dynamic.tlasData.byteLength
    + dynamic.sweptTlasData.byteLength + dynamic.emissiveTlasData.byteLength
    + dynamic.instanceData.byteLength;
  assert.ok(perFrameUpload < 64 * 1024, `${perFrameUpload} dynamic bytes/frame exceeds budget`);
  const recordWords = new Uint32Array(dynamic.instanceData.buffer);
  assert.ok((recordWords[31] & 2) !== 0, "moving instance must carry a swept-change flag");
  dynamic.emissionScale = 0;
  dynamic.update(2);
  assert.equal(dynamic.emissiveInstanceCount, 0, "mesh-light response gate needs a true lights-off state");
  assert.equal(dynamic.frameInfo()[2], 0xffffffff);
});

test("dynamic raster primitive order matches BLAS triangle order", () => {
  const geometry = new Geometry();
  geometry.box([0, 0, 0], [2, 2, 2]);
  const finished = geometry.finish();
  assert.equal(finished.orderedSourceIndices.length, finished.triangleCount);
  const source = finished.orderedSourceIndices[0];
  const rasterA = Array.from(finished.vertices.slice(source * 48, source * 48 + 3));
  const blasA = Array.from(finished.triangles.slice(0, 3));
  assert.deepEqual(rasterA, blasA);
});

test("the daylight door is an ordinary dynamic instance with a swept hierarchy", () => {
  const dynamic = new DynamicScene(8, staticFixture());
  assert.equal(dynamic.instanceCount, 1);
  assert.equal(dynamic.frameInfo()[3], 0xffffffff);
  const closedCenter = [...dynamic.instances[0].center];
  dynamic.rasterDirty = false;
  dynamic.update(4);
  assert.equal(dynamic.instanceCount, 1);
  assert.notDeepEqual(dynamic.instances[0].center, closedCenter);
  assert.equal(dynamic.rasterDirty, false);
  assert.equal(dynamic.frameInfo()[3], dynamic.sweptTlasNodeOffset);
});

test("rigid sweep is the tight union of the two discrete frame poses", () => {
  const asset = {
    boundsMin: [-2, -0.1, -0.1],
    boundsMax: [2, 0.1, 0.1],
  };
  const previous = {
    center: [0, 0, 0],
    rotation: [0, 0, 0, 1],
    scale: [1, 1, 1],
  };
  // History only ever sampled the object at discrete frame poses, so the
  // swept volume is the union of the two exact endpoint AABBs. A 180-degree
  // in-plane flip still reports motion (its displacement is the full
  // diameter) but must NOT dilate to the rotation sphere: intermediate arc
  // positions never contributed to any cached estimate. Per-frame induction
  // covers multi-frame paths because every earlier pose pair was already
  // invalidated when it happened.
  const flipped = conservativeRigidSweep(asset, previous, {
    center: [0, 0, 0],
    rotation: [0, 0, 1, 0],
    scale: [1, 1, 1],
  });
  const radius = Math.hypot(2, 0.1, 0.1);
  assert.equal(flipped.moved, true);
  assert.ok(flipped.maximumPointDisplacement >= radius * 2 - 1e-9);
  assert.deepEqual(flipped.minimum, [-2, -0.1, -0.1]);
  assert.deepEqual(flipped.maximum, [2, 0.1, 0.1]);

  // A quarter turn about Z occupies genuinely different endpoint boxes; the
  // sweep must cover exactly their union (the door-leaf case: no room-scale
  // center-sphere cube).
  const quarter = conservativeRigidSweep(asset, previous, {
    center: [0, 0, 0],
    rotation: quaternionFromAxisAngle([0, 0, 1], Math.PI / 2),
    scale: [1, 1, 1],
  });
  assert.equal(quarter.moved, true);
  for (let axis = 0; axis < 2; axis++) {
    assert.ok(Math.abs(quarter.minimum[axis] - -2) < 1e-6);
    assert.ok(Math.abs(quarter.maximum[axis] - 2) < 1e-6);
  }
  assert.ok(Math.abs(quarter.minimum[2] - -0.1) < 1e-6);
  assert.ok(Math.abs(quarter.maximum[2] - 0.1) < 1e-6);

  const quaternionSignOnly = conservativeRigidSweep(asset, previous, {
    ...previous,
    rotation: [0, 0, 0, -1],
  });
  assert.equal(quaternionSignOnly.moved, false);
});

function quaternionFromAxisAngle(axis, angle) {
  const s = Math.sin(angle / 2);
  return [axis[0] * s, axis[1] * s, axis[2] * s, Math.cos(angle / 2)];
}
