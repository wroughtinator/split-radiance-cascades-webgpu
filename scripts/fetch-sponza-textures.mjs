import { mkdir, readFile, writeFile } from "node:fs/promises";

const sourceRoot = new URL("../vendor/sponza/", import.meta.url);
const textureRoot = new URL("textures/", sourceRoot);
const gltf = JSON.parse(await readFile(new URL("Sponza.gltf", sourceRoot), "utf8"));
const upstream = "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/Sponza/glTF/";

await mkdir(textureRoot, { recursive: true });

const uris = [...new Set(gltf.materials.map((material) => {
  const textureIndex = material.pbrMetallicRoughness?.baseColorTexture?.index;
  const sourceIndex = textureIndex == null ? null : gltf.textures[textureIndex]?.source;
  return sourceIndex == null ? null : gltf.images[sourceIndex]?.uri;
}).filter(Boolean))];

for (const uri of uris) {
  const response = await fetch(upstream + uri);
  if (!response.ok) throw new Error(`Failed to fetch ${uri}: ${response.status}`);
  await writeFile(new URL(uri, textureRoot), new Uint8Array(await response.arrayBuffer()));
  console.log(`Fetched ${uri}`);
}

console.log(`Fetched ${uris.length} official Khronos Sponza base-color textures.`);
