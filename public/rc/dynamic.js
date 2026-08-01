import { Geometry, daylightDoorOpenAmount } from "./scenes.js?v=2026-08-01-dynamic-tlas10";

const INSTANCE_WORDS = 32;
const NODE_WORDS = 8;
const MAX_INSTANCES = 64;
const MOVED_EPSILON = 1e-5;

const COLORS = [
  [0.92, 0.16, 0.07], [0.04, 0.72, 0.82], [0.95, 0.58, 0.08],
  [0.46, 0.12, 0.78], [0.10, 0.58, 0.22], [0.72, 0.76, 0.82],
];

function quaternionFromEuler(x, y, z) {
  const sx = Math.sin(x * 0.5), cx = Math.cos(x * 0.5);
  const sy = Math.sin(y * 0.5), cy = Math.cos(y * 0.5);
  const sz = Math.sin(z * 0.5), cz = Math.cos(z * 0.5);
  return [
    sx * cy * cz - cx * sy * sz,
    cx * sy * cz + sx * cy * sz,
    cx * cy * sz - sx * sy * cz,
    cx * cy * cz + sx * sy * sz,
  ];
}

function rotateByQuaternion(vector, quaternion) {
  const [x, y, z] = vector;
  const [qx, qy, qz, qw] = quaternion;
  const tx = 2 * (qy * z - qz * y);
  const ty = 2 * (qz * x - qx * z);
  const tz = 2 * (qx * y - qy * x);
  return [
    x + qw * tx + qy * tz - qz * ty,
    y + qw * ty + qz * tx - qx * tz,
    z + qw * tz + qx * ty - qy * tx,
  ];
}

function normalize(vector) {
  const inverse = 1 / Math.max(1e-12, Math.hypot(...vector));
  return vector.map((value) => value * inverse);
}

function transformedBounds(asset, instance) {
  const minimum = [Infinity, Infinity, Infinity];
  const maximum = [-Infinity, -Infinity, -Infinity];
  for (let mask = 0; mask < 8; mask++) {
    const local = [0, 1, 2].map((axis) => (
      (mask & (1 << axis) ? asset.boundsMax[axis] : asset.boundsMin[axis])
      * instance.scale[axis]
    ));
    const rotated = rotateByQuaternion(local, instance.rotation);
    for (let axis = 0; axis < 3; axis++) {
      const world = rotated[axis] + instance.center[axis];
      minimum[axis] = Math.min(minimum[axis], world);
      maximum[axis] = Math.max(maximum[axis], world);
    }
  }
  return { minimum, maximum };
}

function makeAsset(name) {
  const geometry = new Geometry();
  if (name === "box") geometry.box([0, 0, 0], [2, 2, 2], [1, 1, 1]);
  if (name === "panel") {
    // A two-sided radiometric panel has four source triangles instead of the
    // twelve outward faces of a closed box. Besides matching how a practical
    // moving area light is built, this keeps exact polygon-form-factor
    // integration proportional to luminous area rather than enclosure detail.
    // The finite separation avoids zero-thickness AABBs on all backends.
    geometry.quad(
      [-1, -0.62, 0.025], [1, -0.62, 0.025],
      [1, 0.62, 0.025], [-1, 0.62, 0.025], [1, 1, 1], [0, 0, 0],
    );
    geometry.quad(
      [1, -0.62, -0.025], [-1, -0.62, -0.025],
      [-1, 0.62, -0.025], [1, 0.62, -0.025], [1, 1, 1], [0, 0, 0],
    );
  }
  if (name === "sphere") geometry.sphere([0, 0, 0], 1, [1, 1, 1], [0, 0, 0], 7, 12);
  if (name === "torus") geometry.torus([0, 0, 0], 0.78, 0.27, [1, 1, 1], 12, 7);
  if (name === "cylinder") geometry.cylinder([0, 0, 0], 0.72, 2, [1, 1, 1], 12);
  const finished = geometry.finish();
  return {
    name,
    vertices: finished.vertices,
    nodes: finished.nodes,
    triangles: finished.triangles,
    nodeCount: finished.nodeCount,
    triangleCount: finished.triangles.length / INSTANCE_WORDS,
    vertexCount: finished.vertexCount,
    boundsMin: finished.boundsMin,
    boundsMax: finished.boundsMax,
  };
}

function sponzaInstances(seconds) {
  const result = [];
  for (let index = 0; index < 48; index++) {
    const column = index % 12;
    const row = Math.floor(index / 12);
    const phase = seconds * (0.38 + (index % 5) * 0.035) + index * 1.731;
    const emissive = index % 7 === 0;
    // Moving mesh lights are compact two-sided radiometric panels. Curved
    // non-emissive movers still stress BLAS transforms and normal handling,
    // while source integration stays proportional to luminous area.
    const asset = emissive ? "panel" : ["box", "sphere", "torus", "cylinder"][index % 4];
    const scaleBase = 0.18 + (index % 3) * 0.04;
    const center = [
      -0.5 + column * 0.9,
      0.28 + scaleBase + row * 0.46 + Math.sin(phase * 1.37) * (0.12 + row * 0.045),
      -2.4 + row * 1.6 + Math.sin(phase * 0.73) * 0.18,
    ];
    const scale = asset === "panel"
      ? [scaleBase * 0.24, scaleBase * 0.24, scaleBase * 0.24]
      : asset === "box"
      ? [scaleBase * 0.30, scaleBase * (0.5 + (index % 4) * 0.10), scaleBase * 0.30]
      : asset === "cylinder"
        ? [scaleBase * 0.72, scaleBase * 1.22, scaleBase * 0.72]
        : [scaleBase, scaleBase, scaleBase];
    const tint = COLORS[index % COLORS.length];
    result.push({
      asset,
      center,
      scale,
      rotation: quaternionFromEuler(
        Math.sin(phase * 0.41) * 0.42,
        phase * (0.55 + (index % 4) * 0.11),
        Math.cos(phase * 0.37) * 0.38,
      ),
      albedo: emissive ? [0.92, 0.92, 0.88] : tint,
      // Match the sky/sun scene-linear exposure range. The former 3.2x red
      // boxes could legitimately saturate an entire nearby Sponza wall and
      // obscured the geometry this validation scene is meant to reveal.
      emission: emissive ? tint.map((value) => 0.35 + value * 0.55) : [0, 0, 0],
      closed: asset !== "panel",
    });
  }
  return result;
}

function doorInstances(seconds) {
  const openness = daylightDoorOpenAmount(seconds);
  const angle = openness * Math.PI * 0.49;
  const doorHalfWidth = 1.35;
  const doorHeight = 4.35;
  const front = 7;
  const overlap = 0.18;
  const hinge = [-doorHalfWidth - overlap, doorHeight * 0.5 - 0.02, front - 0.27];
  const width = doorHalfWidth * 2 + overlap * 2;
  return [{
    asset: "box",
    center: [
      hinge[0] + Math.cos(angle) * width * 0.5,
      hinge[1],
      hinge[2] + Math.sin(angle) * width * 0.5,
    ],
    scale: [width * 0.5, (doorHeight + 0.32) * 0.5, 0.14],
    rotation: quaternionFromEuler(0, angle, 0),
    albedo: [0.18, 0.21, 0.22],
    emission: [0, 0, 0],
    closed: true,
  }];
}

function sceneInstances(sceneIndex, seconds) {
  if (sceneIndex === 1) return sponzaInstances(seconds);
  if (sceneIndex === 8) return doorInstances(seconds);
  return [];
}

function packTlas(bounds, globalNodeOffset, capacity) {
  const nodes = [];
  const indices = bounds.map((_, index) => index);
  const build = (members) => {
    const index = nodes.length;
    const node = {
      minimum: [Infinity, Infinity, Infinity],
      maximum: [-Infinity, -Infinity, -Infinity],
      left: 0,
      right: 0,
      leaf: members.length === 1,
    };
    nodes.push(node);
    for (const member of members) {
      for (let axis = 0; axis < 3; axis++) {
        node.minimum[axis] = Math.min(node.minimum[axis], bounds[member].minimum[axis]);
        node.maximum[axis] = Math.max(node.maximum[axis], bounds[member].maximum[axis]);
      }
    }
    if (node.leaf) {
      node.left = bounds[members[0]].instanceIndex ?? members[0];
      node.right = 1;
      return index;
    }
    const extent = node.maximum.map((value, axis) => value - node.minimum[axis]);
    const axis = extent.indexOf(Math.max(...extent));
    members.sort((a, b) => (
      bounds[a].minimum[axis] + bounds[a].maximum[axis]
      - bounds[b].minimum[axis] - bounds[b].maximum[axis]
    ));
    const middle = Math.floor(members.length / 2);
    node.left = build(members.slice(0, middle));
    node.right = build(members.slice(middle));
    return index;
  };
  if (indices.length) build(indices);
  const packed = new Float32Array(capacity * NODE_WORDS);
  const words = new Uint32Array(packed.buffer);
  for (let index = 0; index < nodes.length; index++) {
    const node = nodes[index];
    const offset = index * NODE_WORDS;
    packed.set(node.minimum, offset);
    packed.set(node.maximum, offset + 4);
    words[offset + 3] = node.leaf
      ? (0x80000000 | node.left) >>> 0
      : globalNodeOffset + node.left;
    words[offset + 7] = node.leaf ? node.right : globalNodeOffset + node.right;
  }
  return { packed, nodeCount: nodes.length };
}

function appendBlasNodes(target, targetNodeOffset, source, sourceNodeOffset, triangleOffset) {
  const destinationWords = new Uint32Array(target.buffer);
  const sourceWords = new Uint32Array(source.buffer, source.byteOffset, source.length);
  for (let node = 0; node < source.length / NODE_WORDS; node++) {
    const sourceBase = node * NODE_WORDS;
    const destinationBase = (targetNodeOffset + node) * NODE_WORDS;
    target.set(source.subarray(sourceBase, sourceBase + NODE_WORDS), destinationBase);
    const left = sourceWords[sourceBase + 3];
    const right = sourceWords[sourceBase + 7];
    if ((left & 0x80000000) !== 0) {
      destinationWords[destinationBase + 3] = (0x80000000 | ((left & 0x7fffffff) + triangleOffset)) >>> 0;
      destinationWords[destinationBase + 7] = right;
    } else {
      destinationWords[destinationBase + 3] = targetNodeOffset + left;
      destinationWords[destinationBase + 7] = targetNodeOffset + right;
    }
  }
}

export class DynamicScene {
  constructor(sceneIndex, staticGeometry) {
    this.sceneIndex = sceneIndex;
    this.assets = ["box", "sphere", "torus", "cylinder", "panel"].map(makeAsset);
    this.assetByName = new Map(this.assets.map((asset) => [asset.name, asset]));
    this.maximumInstances = MAX_INSTANCES;
    this.tlasCapacity = MAX_INSTANCES * 2 - 1;
    this.staticNodeCount = staticGeometry.nodes.length / NODE_WORDS;
    this.staticTriangleCount = staticGeometry.triangles.length / INSTANCE_WORDS;
    this.tlasNodeOffset = this.staticNodeCount;
    this.sweptTlasNodeOffset = this.staticNodeCount + this.tlasCapacity;
    this.emissiveTlasNodeOffset = this.staticNodeCount + this.tlasCapacity * 2;
    let dynamicNodeCount = 0;
    let dynamicTriangleCount = 0;
    for (const asset of this.assets) {
      asset.globalNodeOffset = this.staticNodeCount + this.tlasCapacity * 3 + dynamicNodeCount;
      asset.globalTriangleOffset = this.staticTriangleCount + dynamicTriangleCount;
      dynamicNodeCount += asset.nodeCount;
      dynamicTriangleCount += asset.triangleCount;
    }
    this.dynamicBlasTriangleCount = dynamicTriangleCount;
    this.instanceRecordOffset = this.staticTriangleCount + dynamicTriangleCount;
    this.combinedNodes = new Float32Array(
      staticGeometry.nodes.length + (this.tlasCapacity * 3 + dynamicNodeCount) * NODE_WORDS,
    );
    this.combinedNodes.set(staticGeometry.nodes);
    this.combinedTriangles = new Float32Array(
      staticGeometry.triangles.length
      + dynamicTriangleCount * INSTANCE_WORDS
      + MAX_INSTANCES * INSTANCE_WORDS,
    );
    this.combinedTriangles.set(staticGeometry.triangles);
    for (const asset of this.assets) {
      appendBlasNodes(
        this.combinedNodes,
        asset.globalNodeOffset,
        asset.nodes,
        asset.globalNodeOffset,
        asset.globalTriangleOffset,
      );
      this.combinedTriangles.set(asset.triangles, asset.globalTriangleOffset * INSTANCE_WORDS);
    }
    this.previousBounds = [];
    this.instances = [];
    this.emissionScale = 1;
    this.rasterVertices = new Float32Array(16);
    this.tlasData = new Float32Array(this.tlasCapacity * NODE_WORDS);
    this.sweptTlasData = new Float32Array(this.tlasCapacity * NODE_WORDS);
    this.emissiveTlasData = new Float32Array(this.tlasCapacity * NODE_WORDS);
    this.instanceData = new Float32Array(MAX_INSTANCES * INSTANCE_WORDS);
    this.update(0, true);
    this.combinedNodes.set(this.tlasData, this.tlasNodeOffset * NODE_WORDS);
    this.combinedNodes.set(this.sweptTlasData, this.sweptTlasNodeOffset * NODE_WORDS);
    this.combinedNodes.set(this.emissiveTlasData, this.emissiveTlasNodeOffset * NODE_WORDS);
    this.combinedTriangles.set(this.instanceData, this.instanceRecordOffset * INSTANCE_WORDS);
  }

  update(seconds, initial = false) {
    const instances = sceneInstances(this.sceneIndex, seconds);
    if (this.emissionScale !== 1) {
      for (const instance of instances) {
        instance.emission = instance.emission.map((value) => value * this.emissionScale);
      }
    }
    if (instances.length > MAX_INSTANCES) throw new Error("Dynamic instance capacity exceeded.");
    const rasterSignature = instances.map((instance) => instance.asset).join("|");
    this.rasterDirty = rasterSignature !== this.rasterSignature;
    if (this.rasterDirty) {
      const rasterFloatCount = instances.reduce((total, instance) => (
        total + this.assetByName.get(instance.asset).vertices.length
      ), 0);
      this.rasterVertices = new Float32Array(Math.max(16, rasterFloatCount));
      let output = 0;
      for (let instanceIndex = 0; instanceIndex < instances.length; instanceIndex++) {
        const source = this.assetByName.get(instances[instanceIndex].asset).vertices;
        for (let vertex = 0; vertex < source.length; vertex += 16) {
          this.rasterVertices.set(source.subarray(vertex, vertex + 16), output);
          // Procedural dynamic meshes never sample the material atlas. This
          // otherwise-unused scalar carries the stable instance record index
          // into the dynamic vertex entry points, which restore -1 for the
          // fragment material selector.
          this.rasterVertices[output + 14] = instanceIndex;
          output += 16;
        }
      }
      this.rasterSignature = rasterSignature;
      this.vertexCount = output / 16;
    }
    const bounds = instances.map((instance, instanceIndex) => ({
      ...transformedBounds(this.assetByName.get(instance.asset), instance),
      instanceIndex,
    }));
    this.instanceData.fill(0);
    const words = new Uint32Array(this.instanceData.buffer);
    let dynamicTriangles = 0;
    for (let index = 0; index < instances.length; index++) {
      const instance = instances[index];
      const asset = this.assetByName.get(instance.asset);
      const previous = this.previousBounds[index];
      const displacement = previous
        ? Math.max(...bounds[index].minimum.map((value, axis) => Math.abs(value - previous.minimum[axis])),
          ...bounds[index].maximum.map((value, axis) => Math.abs(value - previous.maximum[axis])))
        : 0;
      const moved = !initial && displacement > MOVED_EPSILON;
      const sweptMinimum = moved
        ? bounds[index].minimum.map((value, axis) => Math.min(value, previous.minimum[axis]))
        : [1, 1, 1];
      const sweptMaximum = moved
        ? bounds[index].maximum.map((value, axis) => Math.max(value, previous.maximum[axis]))
        : [-1, -1, -1];
      const base = index * INSTANCE_WORDS;
      this.instanceData.set([...instance.center, 1], base);
      this.instanceData.set(instance.rotation, base + 4);
      this.instanceData.set([...instance.scale, 0], base + 8);
      this.instanceData.set([...instance.albedo, 0], base + 12);
      this.instanceData.set([...instance.emission, 0], base + 16);
      this.instanceData.set([...sweptMinimum, 0], base + 20);
      this.instanceData.set([...sweptMaximum, 0], base + 24);
      words[base + 28] = asset.globalNodeOffset;
      words[base + 29] = asset.globalTriangleOffset;
      words[base + 30] = asset.triangleCount;
      words[base + 31] = (instance.emission.some((value) => value > 0) ? 1 : 0)
        | (moved ? 2 : 0) | (instance.closed ? 4 : 0);

      dynamicTriangles += asset.triangleCount;
    }
    const tlas = packTlas(bounds, this.tlasNodeOffset, this.tlasCapacity);
    const sweptBounds = bounds.flatMap((current, instanceIndex) => {
      const previous = this.previousBounds[instanceIndex];
      if (initial || !previous) return [];
      const displacement = Math.max(
        ...current.minimum.map((value, axis) => Math.abs(value - previous.minimum[axis])),
        ...current.maximum.map((value, axis) => Math.abs(value - previous.maximum[axis])),
      );
      if (displacement <= MOVED_EPSILON) return [];
      return [{
        instanceIndex,
        minimum: current.minimum.map((value, axis) => Math.min(value, previous.minimum[axis])),
        maximum: current.maximum.map((value, axis) => Math.max(value, previous.maximum[axis])),
      }];
    });
    const sweptTlas = packTlas(
      sweptBounds,
      this.sweptTlasNodeOffset,
      this.tlasCapacity,
    );
    const emissiveBounds = bounds.filter((_, instanceIndex) => (
      instances[instanceIndex].emission.some((value) => value > 0)
    ));
    const emissiveTlas = packTlas(
      emissiveBounds,
      this.emissiveTlasNodeOffset,
      this.tlasCapacity,
    );
    this.tlasData = tlas.packed;
    this.tlasNodeCount = tlas.nodeCount;
    this.sweptTlasData = sweptTlas.packed;
    this.sweptTlasNodeCount = sweptTlas.nodeCount;
    this.emissiveTlasData = emissiveTlas.packed;
    this.emissiveTlasNodeCount = emissiveTlas.nodeCount;
    this.previousBounds = bounds;
    this.instances = instances;
    this.instanceCount = instances.length;
    this.emissiveInstanceCount = instances.filter((instance) => (
      instance.emission.some((value) => value > 0)
    )).length;
    this.triangleCount = dynamicTriangles;
    return this;
  }

  frameInfo() {
    return [
      this.instanceCount ? this.tlasNodeOffset : 0xffffffff,
      this.instanceRecordOffset,
      this.emissiveTlasNodeCount ? this.emissiveTlasNodeOffset : 0xffffffff,
      this.sweptTlasNodeCount ? this.sweptTlasNodeOffset : 0xffffffff,
    ];
  }
}

export function createDynamicScene(sceneIndex, staticGeometry) {
  if (sceneIndex !== 1 && sceneIndex !== 8) return null;
  return new DynamicScene(sceneIndex, staticGeometry);
}
