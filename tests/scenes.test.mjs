import assert from "node:assert/strict";
import test from "node:test";
import { createScene, SCENE_INFO } from "../public/rc/scenes.js";

test("ten production stress scenes build valid geometry and BVHs", () => {
  assert.equal(SCENE_INFO.length, 10);
  let totalTriangles = 0;
  for (let i = 0; i < SCENE_INFO.length; i++) {
    const scene = createScene(i);
    const g = scene.geometry;
    assert.ok(g.vertexCount > 0, `${scene.name} has no raster vertices`);
    assert.equal(g.vertexCount, g.triangleCount * 3);
    assert.equal(g.triangles.length, g.triangleCount * 28);
    assert.equal(g.nodes.length, g.nodeCount * 8);
    assert.ok(g.triangleCount >= 800, `${scene.name} is not a meaningful stress case`);
    assert.ok(Number.isFinite(scene.radius) && scene.radius > 0);
    assert.ok(scene.baseSpacing > 0);
    for (const value of [...g.boundsMin, ...g.boundsMax, ...g.vertices.subarray(0, Math.min(2048, g.vertices.length))]) {
      assert.ok(Number.isFinite(value), `${scene.name} contains non-finite geometry`);
    }
    totalTriangles += g.triangleCount;
  }
  assert.ok(totalTriangles > 50_000, `expected broad geometry coverage, received ${totalTriangles} triangles`);
});

test("scene suite contains requested terrain and complexity stressors", () => {
  const names = SCENE_INFO.map((scene) => `${scene.name} ${scene.description}`).join(" ").toLowerCase();
  for (const term of ["heightfield", "concave", "forest", "moving", "emissive", "terrain", "complex"]) {
    assert.match(names, new RegExp(term));
  }
});
