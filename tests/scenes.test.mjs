import assert from "node:assert/strict";
import test from "node:test";
import { automaticBaseSpacing, createScene, Geometry, SCENE_INFO } from "../public/rc/scenes.js";

test("production validation scenes build valid geometry and BVHs", () => {
  assert.equal(SCENE_INFO.length, 13);
  let totalTriangles = 0;
  for (let i = 0; i < SCENE_INFO.length; i++) {
    const scene = createScene(i);
    const g = scene.geometry;
    assert.ok(g.vertexCount > 0, `${scene.name} has no raster vertices`);
    assert.equal(g.vertexCount, g.triangleCount * 3);
    assert.equal(g.triangles.length, g.triangleCount * 32);
    assert.equal(g.nodes.length, g.nodeCount * 8);
    const minimumTriangles = ["Cornell", "Visibility"].includes(scene.short) ? 30 : 700;
    assert.ok(g.triangleCount >= minimumTriangles, `${scene.name} is not a meaningful validation case`);
    assert.ok(Number.isFinite(scene.radius) && scene.radius > 0);
    assert.equal(
      scene.baseSpacing,
      automaticBaseSpacing(scene.radius, g.triangleCount),
      `${scene.name} bypasses universal automatic probe spacing`,
    );
    for (const value of [...g.boundsMin, ...g.boundsMax, ...g.vertices.subarray(0, Math.min(2048, g.vertices.length))]) {
      assert.ok(Number.isFinite(value), `${scene.name} contains non-finite geometry`);
    }
    totalTriangles += g.triangleCount;
  }
  assert.ok(totalTriangles > 50_000, `expected broad geometry coverage, received ${totalTriangles} triangles`);
});

test("automatic GI scale is scene-identity independent", () => {
  assert.equal(automaticBaseSpacing(20, 10_000), automaticBaseSpacing(20, 10_000));
  assert.ok(automaticBaseSpacing(40, 10_000) > automaticBaseSpacing(20, 10_000));
  assert.ok(automaticBaseSpacing(20, 100_000) < automaticBaseSpacing(20, 1_000));
});

test("scene suite contains requested terrain and complexity stressors", () => {
  const names = SCENE_INFO.map((scene) => `${scene.name} ${scene.description}`).join(" ").toLowerCase();
  for (const term of ["heightfield", "heightmap", "cornell", "concave", "forest", "moving", "emissive", "terrain", "complex"]) {
    assert.match(names, new RegExp(term));
  }
});

test("sphere raster normals and geometric winding both face outward", () => {
  const geometry = new Geometry();
  geometry.sphere([0, 0, 0], 2, [1, 1, 1], [0, 0, 0], 8, 16);
  const built = geometry.finish();

  for (let vertex = 0; vertex < built.vertexCount; vertex++) {
    const offset = vertex * 16;
    const dot = built.vertices[offset] * built.vertices[offset + 3]
      + built.vertices[offset + 1] * built.vertices[offset + 4]
      + built.vertices[offset + 2] * built.vertices[offset + 5];
    assert.ok(dot > 1.99, `sphere vertex ${vertex} has a flipped smooth normal`);
  }

  for (let triangle = 0; triangle < built.triangleCount; triangle++) {
    const offset = triangle * 32;
    const a = built.triangles.subarray(offset, offset + 3);
    const b = built.triangles.subarray(offset + 4, offset + 7);
    const c = built.triangles.subarray(offset + 8, offset + 11);
    const ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    const ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    const face = [
      ab[1] * ac[2] - ab[2] * ac[1],
      ab[2] * ac[0] - ab[0] * ac[2],
      ab[0] * ac[1] - ab[1] * ac[0],
    ];
    const centroid = [
      (a[0] + b[0] + c[0]) / 3,
      (a[1] + b[1] + c[1]) / 3,
      (a[2] + b[2] + c[2]) / 3,
    ];
    const orientation = face[0] * centroid[0]
      + face[1] * centroid[1]
      + face[2] * centroid[2];
    assert.ok(orientation > 0, `sphere triangle ${triangle} is wound inward`);
  }
});

test("closed convex primitive faces are consistently wound outward", () => {
  const cases = [
    ["box", (geometry) => geometry.box([0, 0, 0], [2, 3, 4])],
    ["sphere", (geometry) => geometry.sphere([0, 0, 0], 2, undefined, undefined, 6, 12)],
    ["cylinder", (geometry) => geometry.cylinder([0, 0, 0], 2, 4, undefined, 12)],
    ["cone", (geometry) => geometry.cone([0, 0, 0], 2, 4, undefined, 12)],
  ];

  for (const [name, build] of cases) {
    const geometry = new Geometry();
    build(geometry);
    for (let index = 0; index < geometry.triangles.length; index++) {
      const { a, b, c } = geometry.triangles[index];
      const ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
      const ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
      const face = [
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
      ];
      const centroid = [
        (a[0] + b[0] + c[0]) / 3,
        (a[1] + b[1] + c[1]) / 3,
        (a[2] + b[2] + c[2]) / 3,
      ];
      const orientation = face[0] * centroid[0]
        + face[1] * centroid[1]
        + face[2] * centroid[2];
      assert.ok(orientation > 0, `${name} triangle ${index} is wound inward`);
    }
  }
});
