import { readFile, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { gzipSync } from "node:zlib";
import sharp from "sharp";
import { buildBVH, normalize3 } from "../public/rc/math.js";

const SOURCE_ROOT = new URL("../vendor/sponza/", import.meta.url);
const OUTPUT_ROOT = new URL("../public/models/", import.meta.url);
const SOURCE_GLTF = new URL("Sponza.gltf", SOURCE_ROOT);
const SOURCE_BIN = new URL("Sponza.bin", SOURCE_ROOT);
// The custom extension is intentional: static servers must not auto-apply a
// Content-Encoding header because the browser explicitly decompresses this blob.
const OUTPUT = new URL("sponza.rcb", OUTPUT_ROOT);
const ATLAS_OUTPUT = new URL("sponza-atlas.webp", OUTPUT_ROOT);
const TEXTURE_ROOT = new URL("textures/", SOURCE_ROOT);
const ATLAS_SIZE = 4096;
const ATLAS_GRID = 5;
const ATLAS_CELL = 819;
const ATLAS_PADDING = 4;
const ATLAS_CONTENT = ATLAS_CELL - ATLAS_PADDING * 2;

// Linear-space averages of the 25 base-color textures. These are only fallbacks
// for primitives without texture coordinates; normal rendering uses the atlas.
const MATERIAL_COLORS = [
  [0.0281, 0.0262, 0.0123], [0.0783, 0.0580, 0.0609], [0.3333, 0.3333, 0.3333],
  [0.1079, 0.1020, 0.0024], [0.0511, 0.0410, 0.0262], [0.1884, 0.1673, 0.1269],
  [0.1039, 0.0870, 0.0654], [0.2441, 0.1911, 0.1182], [0.1911, 0.1647, 0.1247],
  [0.2733, 0.2227, 0.1523], [0.2167, 0.1911, 0.1475], [0.0685, 0.0624, 0.0538],
  [0.1572, 0.1313, 0.0962], [0.1269, 0.1039, 0.0943], [0.0053, 0.0817, 0.0061],
  [0.0084, 0.0434, 0.1204], [0.1020, 0.0105, 0.0049], [0.0136, 0.0566, 0.1698],
  [0.1381, 0.0049, 0.0024], [0.0084, 0.0624, 0.0012], [0.0800, 0.0321, 0.0163],
  [0.0074, 0.0089, 0.0089], [0.2197, 0.1750, 0.1182], [0.0654, 0.0511, 0.0291],
  [0.0888, 0.0800, 0.0835],
];

const COMPONENT_BYTES = new Map([[5121, 1], [5123, 2], [5125, 4], [5126, 4]]);
const COMPONENTS = new Map([["SCALAR", 1], ["VEC2", 2], ["VEC3", 3], ["VEC4", 4]]);

function accessorReader(gltf, binary, accessorIndex) {
  const accessor = gltf.accessors[accessorIndex];
  const view = gltf.bufferViews[accessor.bufferView];
  const components = COMPONENTS.get(accessor.type);
  const componentBytes = COMPONENT_BYTES.get(accessor.componentType);
  const stride = view.byteStride || components * componentBytes;
  const base = (view.byteOffset || 0) + (accessor.byteOffset || 0);
  const data = new DataView(binary.buffer, binary.byteOffset, binary.byteLength);
  const scalar = accessor.componentType === 5121
    ? (offset) => data.getUint8(offset)
    : accessor.componentType === 5123
      ? (offset) => data.getUint16(offset, true)
      : accessor.componentType === 5125
        ? (offset) => data.getUint32(offset, true)
        : (offset) => data.getFloat32(offset, true);
  return {
    count: accessor.count,
    read(index) {
      const offset = base + index * stride;
      if (components === 1) return scalar(offset);
      return Array.from({ length: components }, (_, component) => scalar(offset + component * componentBytes));
    },
  };
}

function nodeScale(gltf) {
  const node = gltf.nodes[gltf.scenes[gltf.scene || 0].nodes[0]];
  return node.scale || [1, 1, 1];
}

const gltf = JSON.parse(await readFile(SOURCE_GLTF, "utf8"));
const binary = await readFile(SOURCE_BIN);
const scale = nodeScale(gltf);
const primitiveData = [];
let triangleCount = 0;

const atlasLayers = [];
for (let materialIndex = 0; materialIndex < gltf.materials.length; materialIndex++) {
  const material = gltf.materials[materialIndex];
  const textureIndex = material.pbrMetallicRoughness?.baseColorTexture?.index;
  const sourceIndex = textureIndex == null ? null : gltf.textures[textureIndex]?.source;
  const uri = sourceIndex == null ? null : gltf.images[sourceIndex]?.uri;
  if (!uri) continue;
  const tile = await sharp(await readFile(new URL(uri, TEXTURE_ROOT)))
    .resize(ATLAS_CONTENT, ATLAS_CONTENT, { fit: "fill" })
    .extend({
      top: ATLAS_PADDING,
      bottom: ATLAS_PADDING,
      left: ATLAS_PADDING,
      right: ATLAS_PADDING,
      extendWith: "copy",
    })
    .png()
    .toBuffer();
  atlasLayers.push({
    input: tile,
    left: (materialIndex % ATLAS_GRID) * ATLAS_CELL,
    top: Math.floor(materialIndex / ATLAS_GRID) * ATLAS_CELL,
  });
}
await sharp({
  create: {
    width: ATLAS_SIZE,
    height: ATLAS_SIZE,
    channels: 4,
    background: { r: 255, g: 255, b: 255, alpha: 1 },
  },
}).composite(atlasLayers).webp({ quality: 92, alphaQuality: 100, smartSubsample: true })
  .toFile(fileURLToPath(ATLAS_OUTPUT));

for (const mesh of gltf.meshes) {
  for (const primitive of mesh.primitives) {
    if ((primitive.mode ?? 4) !== 4) continue;
    const positions = accessorReader(gltf, binary, primitive.attributes.POSITION);
    const normals = accessorReader(gltf, binary, primitive.attributes.NORMAL);
    const texcoords = primitive.attributes.TEXCOORD_0 == null
      ? null
      : accessorReader(gltf, binary, primitive.attributes.TEXCOORD_0);
    const indices = primitive.indices == null ? null : accessorReader(gltf, binary, primitive.indices);
    const indexCount = indices?.count ?? positions.count;
    const materialIndex = primitive.material || 0;
    const material = gltf.materials[materialIndex] || {};
    const factor = material.pbrMetallicRoughness?.baseColorFactor?.slice(0, 3) || [1, 1, 1];
    const alphaCutoff = material.alphaMode === "MASK" ? (material.alphaCutoff ?? 0.5) : 0;
    triangleCount += Math.floor(indexCount / 3);
    primitiveData.push({
      positions,
      normals,
      texcoords,
      indices,
      indexCount,
      material: materialIndex,
      factor,
      alphaCutoff,
    });
  }
}

const vertices = new Float32Array(triangleCount * 3 * 16);
const triangles = new Array(triangleCount);
const boundsMin = [Infinity, Infinity, Infinity];
const boundsMax = [-Infinity, -Infinity, -Infinity];
let vertexOffset = 0;
let triangleOffset = 0;

for (const primitive of primitiveData) {
  const fallback = MATERIAL_COLORS[primitive.material] || [0.18, 0.18, 0.18];
  const albedo = primitive.texcoords ? primitive.factor : fallback;
  for (let index = 0; index + 2 < primitive.indexCount; index += 3) {
    const points = new Array(3);
    const uvs = new Array(3);
    for (let corner = 0; corner < 3; corner++) {
      const sourceIndex = primitive.indices ? primitive.indices.read(index + corner) : index + corner;
      const sourcePosition = primitive.positions.read(sourceIndex);
      const position = sourcePosition.map((value, axis) => value * scale[axis]);
      const normal = normalize3(primitive.normals.read(sourceIndex).map((value, axis) => value / scale[axis]));
      const uv = primitive.texcoords?.read(sourceIndex) || [0, 0];
      points[corner] = position;
      uvs[corner] = uv;
      vertices.set([
        ...position,
        ...normal,
        ...albedo,
        0, 0, 0,
        ...uv,
        primitive.texcoords ? primitive.material : -1,
        primitive.alphaCutoff,
      ], vertexOffset);
      vertexOffset += 16;
      for (let axis = 0; axis < 3; axis++) {
        boundsMin[axis] = Math.min(boundsMin[axis], position[axis]);
        boundsMax[axis] = Math.max(boundsMax[axis], position[axis]);
      }
    }
    triangles[triangleOffset++] = {
      a: points[0],
      b: points[1],
      c: points[2],
      albedo,
      emissive: [0, 0, 0],
      uvs,
      material: primitive.texcoords ? primitive.material : -1,
      alphaCutoff: primitive.alphaCutoff,
    };
  }
}

const bvh = buildBVH(triangles, 4);
const headerBytes = 64;
const packed = new ArrayBuffer(
  headerBytes + vertices.byteLength + bvh.nodes.byteLength + bvh.triangles.byteLength,
);
const headerU32 = new Uint32Array(packed, 0, 8);
const headerF32 = new Float32Array(packed, 32, 6);
headerU32.set([
  0x31424352,
  2,
  vertices.length,
  bvh.nodes.length,
  bvh.triangles.length,
  vertices.length / 16,
  bvh.nodeCount,
  bvh.triangleCount,
]);
headerF32.set([...boundsMin, ...boundsMax]);
let byteOffset = headerBytes;
new Float32Array(packed, byteOffset, vertices.length).set(vertices);
byteOffset += vertices.byteLength;
new Float32Array(packed, byteOffset, bvh.nodes.length).set(bvh.nodes);
byteOffset += bvh.nodes.byteLength;
new Float32Array(packed, byteOffset, bvh.triangles.length).set(bvh.triangles);

const compressed = gzipSync(new Uint8Array(packed), { level: 9 });
await writeFile(OUTPUT, compressed);
console.log(JSON.stringify({
  output: OUTPUT.pathname,
  sourceTriangles: triangleCount,
  bvhNodes: bvh.nodeCount,
  uncompressedBytes: packed.byteLength,
  compressedBytes: compressed.byteLength,
  atlas: ATLAS_OUTPUT.pathname,
}, null, 2));
