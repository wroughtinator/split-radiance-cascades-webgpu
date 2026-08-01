import assert from "node:assert/strict";
import test from "node:test";
import { DynamicScene } from "../public/rc/dynamic.js";
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
