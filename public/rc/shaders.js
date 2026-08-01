export const shaderConstants = {
  hashOffsets: [0, 32768, 40960, 43008],
  hashSizes: [32768, 8192, 2048, 1024],
  probeOffsets: [0, 16384, 20480, 21504],
  probeCaps: [16384, 4096, 1024, 512],
  dataOffsets: [0, 524288, 1048576, 1572864],
  directions: [32, 128, 512, 2048],
  totalHashSlots: 44032,
  hashFrames: 2,
  totalProbeMeta: 22016,
  totalDirectionData: 2621440,
  irradianceTexels: 64,
  irradianceAtlasWidth: 512,
  irradianceAtlasFrameHeight: 2048,
  irradianceFrames: 2,
  accumFrames: 2,
  stateWords: 66064,
};

export const rasterShader = /* wgsl */`
struct FrameUniforms {
  viewProj: mat4x4<f32>,
  sunViewProj: mat4x4<f32>,
  cameraPos: vec4f,
  sunDirTime: vec4f,
  sunColorIntensity: vec4f,
  pointPosRange: vec4f,
  pointColorIntensity: vec4f,
  envBaseSpacing: vec4f,
  resolution: vec4f,
  controls: vec4f,
};
@group(0) @binding(0) var<uniform> frame: FrameUniforms;
@group(0) @binding(1) var albedoAtlas: texture_2d_array<f32>;
@group(0) @binding(2) var atlasSampler: sampler;
@group(0) @binding(3) var<uniform> lightViewProj: mat4x4<f32>;

struct VertexIn {
  @location(0) position: vec3f,
  @location(1) normal: vec3f,
  @location(2) albedo: vec3f,
  @location(3) emissive: vec3f,
  @location(4) uv: vec2f,
  @location(5) materialCutoff: vec2f,
};
struct VertexOut {
  @builtin(position) clip: vec4f,
  @location(0) world: vec3f,
  @location(1) normal: vec3f,
  @location(2) albedo: vec3f,
  @location(3) emissive: vec3f,
  @location(4) uv: vec2f,
  @location(5) @interpolate(flat) materialCutoff: vec2f,
};
@vertex fn gbufferVS(v: VertexIn) -> VertexOut {
  var o: VertexOut;
  o.clip = frame.viewProj * vec4f(v.position, 1.0);
  o.world = v.position;
  o.normal = normalize(v.normal);
  o.albedo = v.albedo;
  o.emissive = v.emissive;
  o.uv = v.uv;
  o.materialCutoff = v.materialCutoff;
  return o;
}
struct GBufferOut {
  @location(0) albedo: vec4f,
  @location(1) normal: vec4f,
  @location(2) world: vec4f,
};
fn materialSample(uv:vec2f,material:f32)->vec4f{
  // Derivatives must be evaluated before the material-dependent early return
  // so every fragment in the quad executes them in uniform control flow.
  let uvDx=dpdx(uv);
  let uvDy=dpdy(uv);
  if(material<0.0){return vec4f(1);}
  let index=i32(u32(material+0.5));
  // Keep the periodic UV derivatives continuous across fract() seams.  Using
  // implicit derivatives of the wrapped coordinate selects a coarse mip on
  // the seam itself and makes thin Sponza details flash during translation.
  let sampled=textureSampleGrad(
    albedoAtlas,atlasSampler,fract(uv),index,uvDx,uvDy
  );
  if((u32(frame.cameraPos.w+0.5)&16u)==0u){return sampled;}
  let luminance=dot(sampled.rgb,vec3f(0.2126,0.7152,0.0722));
  let detail=clamp(0.72+luminance*0.34,0.62,1.04);
  let stone=vec3f(0.93,0.96,0.96)*detail;
  let cyan=vec3f(0.18,0.82,0.86)*mix(0.82,1.08,luminance);
  let paperColor=select(stone,cyan,index>=15&&index<=19);
  return vec4f(paperColor,sampled.a);
}
fn encodeNormalOct(normalIn:vec3f)->vec2f{
  let normal=normalize(normalIn);
  var oct=normal.xy/(abs(normal.x)+abs(normal.y)+abs(normal.z));
  if(normal.z<0.0){
    oct=(vec2f(1.0)-abs(oct.yx))*select(vec2f(-1.0),vec2f(1.0),oct>=vec2f(0));
  }
  return oct;
}
fn encodeSurfaceEmission(emission:vec3f)->vec3f{
  // Component-wise logarithmic HDR coding covers [0,255] without a
  // scene-dependent exposure. Red uses the albedo alpha channel; green/blue
  // use the two channels freed by octahedral normal encoding.
  return clamp(log2(vec3f(1.0)+max(emission,vec3f(0)))/8.0,vec3f(0),vec3f(1));
}
@fragment fn gbufferFS(v: VertexOut, @builtin(front_facing) frontFacing: bool) -> GBufferOut {
  var o: GBufferOut;
  let surface=materialSample(v.uv,v.materialCutoff.x);
  if(v.materialCutoff.y>0.0&&surface.a<v.materialCutoff.y){discard;}
  let emission=encodeSurfaceEmission(v.emissive);
  o.albedo = vec4f(v.albedo*surface.rgb, emission.r);
  o.normal = vec4f(
    encodeNormalOct(select(-v.normal,v.normal,frontFacing)),emission.gb
  );
  // Keep two independent bits in the exact fp32 marker: 0.5 marks a raster
  // back face and 1.0 marks any RGB emissive surface. This lets validation
  // reject red-only mesh lights without sacrificing packed emission color.
  let backFaceMarker=select(0.5,1.0,frontFacing);
  let emissiveMarker=select(0.0,1.0,any(v.emissive>vec3f(0)));
  o.world = vec4f(v.world,backFaceMarker+emissiveMarker);
  return o;
}
struct ShadowOut {
  @builtin(position) clip: vec4f,
  @location(0) uv: vec2f,
  @location(1) @interpolate(flat) materialCutoff: vec2f,
};
@vertex fn shadowVS(v: VertexIn) -> ShadowOut {
  var o:ShadowOut;
  o.clip=lightViewProj*vec4f(v.position,1.0);
  o.uv=v.uv;
  o.materialCutoff=v.materialCutoff;
  return o;
}
@fragment fn shadowFS(v:ShadowOut) {
  let surface=materialSample(v.uv,v.materialCutoff.x);
  if(v.materialCutoff.y>0.0&&surface.a<v.materialCutoff.y){discard;}
}
@vertex fn pointShadowVS(v: VertexIn) -> ShadowOut {
  var o:ShadowOut;
  o.clip=lightViewProj*vec4f(v.position,1.0);
  o.uv=v.uv;
  o.materialCutoff=v.materialCutoff;
  return o;
}
`;

const sharedCompute = /* wgsl */`
struct FrameUniforms {
  viewProj: mat4x4<f32>,
  sunViewProj: mat4x4<f32>,
  cameraPos: vec4f,
  sunDirTime: vec4f,
  sunColorIntensity: vec4f,
  pointPosRange: vec4f,
  pointColorIntensity: vec4f,
  envBaseSpacing: vec4f,
  resolution: vec4f,
  controls: vec4f,
};
struct HashSlot { key: atomic<u32>, index: atomic<u32> };
// Keep the frame and history epoch as independent 32-bit values. Packing both
// into one word made the R2 rotation repeat after 65,536 frames and allowed a
// stale ray-map tag to collide after the same interval.
struct PassParams {
  cascade: u32, value: u32, pad0: u32, pad1: u32,
  sampleFrame: u32, sampleEpoch: u32, pad2: u32, pad3: u32,
};
struct SunShadowUniforms {
  matrices: array<mat4x4<f32>,4>,
  splitDepths: vec4f,
  texelSizes: vec4f,
  cameraForward: vec4f,
  parameters: vec4f,
};
struct BvhNode { minMeta: vec4f, maxMeta: vec4f };
struct Triangle {
  a: vec4f, b: vec4f, c: vec4f, albedo: vec4f, emissive: vec4f,
  uvAB: vec4f, uvCMaterial: vec4f, normalOct: vec4u,
};
struct Hit { t: f32, normal: vec3f, albedo: vec3f, emissive: vec3f };

@group(0) @binding(0) var<uniform> frame: FrameUniforms;
@group(0) @binding(1) var worldTex: texture_2d<f32>;
@group(0) @binding(2) var normalTex: texture_2d<f32>;
@group(0) @binding(3) var<storage, read_write> slots: array<HashSlot>;
@group(0) @binding(4) var<storage, read_write> state: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> probeMeta: array<vec4f>;
@group(0) @binding(6) var<storage, read_write> accum: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> cones: array<vec4f>;
@group(0) @binding(8) var<storage, read_write> irradiance: array<vec4f>;
@group(0) @binding(9) var<storage, read> bvhNodes: array<BvhNode>;
@group(0) @binding(10) var<storage, read> triangles: array<Triangle>;
@group(0) @binding(11) var<uniform> passParams: PassParams;
@group(0) @binding(12) var albedoAtlas: texture_2d_array<f32>;
@group(0) @binding(13) var albedoSampler: sampler;
@group(0) @binding(14) var irradianceAtlasStorage: texture_storage_2d<rgba16float,write>;
@group(0) @binding(15) var irradianceAtlasSampled: texture_2d<f32>;
@group(0) @binding(16) var irradianceAtlasSampler: sampler;
@group(0) @binding(17) var pointShadowAuditTex: texture_depth_2d_array;
@group(0) @binding(18) var pointShadowAuditSampler: sampler_comparison;
@group(0) @binding(19) var sunShadowAuditTex: texture_depth_2d_array;
@group(0) @binding(20) var sunShadowAuditSampler: sampler_comparison;
@group(0) @binding(21) var<uniform> sunShadow: SunShadowUniforms;

fn gbufferNormal(pixel:vec2i)->vec3f{
  let oct=textureLoad(normalTex,pixel,0).xy;
  var normal=vec3f(oct,1.0-abs(oct.x)-abs(oct.y));
  if(normal.z<0.0){
    let old=normal.xy;
    normal.x=(1.0-abs(old.y))*select(-1.0,1.0,old.x>=0.0);
    normal.y=(1.0-abs(old.x))*select(-1.0,1.0,old.y>=0.0);
  }
  return normalize(normal);
}

const EMPTY: u32 = 0xffffffffu;
const PI: f32 = 3.141592653589793;
const TAU: f32 = 6.283185307179586;
const FIXED_SCALE: f32 = 4096.0;
const HASH_OFFSETS = array<u32,4>(0u,32768u,40960u,43008u);
const HASH_SIZES = array<u32,4>(32768u,8192u,2048u,1024u);
const PROBE_OFFSETS = array<u32,4>(0u,16384u,20480u,21504u);
const PROBE_CAPS = array<u32,4>(16384u,4096u,1024u,512u);
const DATA_OFFSETS = array<u32,4>(0u,524288u,1048576u,1572864u);
const DIR_COUNTS = array<u32,4>(32u,128u,512u,2048u);
const HASH_FRAME_STRIDE: u32 = 44032u;
const IRRADIANCE_FRAME_STRIDE: u32 = 1048576u;
const ACCUM_FRAME_STRIDE: u32 = 13107200u;
const RAY_COUNT_OFFSET: u32 = 16u;
const RAY_OFFSET_OFFSET: u32 = 22032u;
const RAY_CURSOR_OFFSET: u32 = 44048u;
const BLOCK_COUNT_OFFSET: u32 = 66064u;
const TOTAL_PROBE_META: u32 = 22016u;

fn hash32(value: u32) -> u32 {
  var x = value;
  x = x ^ (x >> 16u);
  x = x * 0x7feb352du;
  x = x ^ (x >> 15u);
  x = x * 0x846ca68bu;
  return x ^ (x >> 16u);
}

fn keyFromCellSurface(cellIn: vec3i, lod: u32, surfaceClass: u32) -> u32 {
  if(any(cellIn<vec3i(-256))||any(cellIn>vec3i(255))){return EMPTY;}
  let c = cellIn + vec3i(256);
  return u32(c.x) | (u32(c.y) << 9u) | (u32(c.z) << 18u)
    | ((lod & 3u) << 27u) | ((surfaceClass & 7u) << 29u);
}

fn keyFromCell(cellIn: vec3i, lod: u32) -> u32 {
  return keyFromCellSurface(cellIn,lod,0u);
}

// A primary probe belongs to one oriented surface sheet. Without this tag,
// the two sides of a thin wall (or the inside and outside of a closed mesh)
// can hash to the same world-space probe and exchange radiance. The six
// dominant-axis hemispheres match the tangent-plane reconstruction below and
// leave the coarser, long-range cascades spatially shared.
fn surfaceClass(normalIn:vec3f)->u32{
  let normal=normalize(normalIn);
  let a=abs(normal);
  var axis=2u;
  var component=normal.z;
  if(a.x>=a.y&&a.x>=a.z){axis=0u;component=normal.x;}
  else if(a.y>=a.z){axis=1u;component=normal.y;}
  return axis*2u+select(0u,1u,component>=0.0);
}

fn lodDistance(position: vec3f) -> f32 {
  let delta = abs(position - frame.cameraPos.xyz);
  return max(delta.x, max(delta.y, delta.z));
}

fn levelOfDetail(position: vec3f) -> u32 {
  let ratio = max(1.0, lodDistance(position) / max(0.001, frame.envBaseSpacing.w * 18.0));
  return u32(clamp(floor(log2(ratio)), 0.0, 3.0));
}

// The paper starts each coarser LOD at 90% of the nominal boundary, then
// linearly blends across the overlap. x=fine LOD, y=coarse LOD, z=blend.
fn lodSelection(position: vec3f) -> vec3f {
  let fine = levelOfDetail(position);
  if (fine >= 3u) { return vec3f(f32(fine), f32(fine), 0.0); }
  let baseRange = max(0.001, frame.envBaseSpacing.w * 18.0);
  let boundary = baseRange * exp2(f32(fine + 1u));
  let overlapStart = boundary * 0.9;
  let blend = clamp((lodDistance(position) - overlapStart) / max(0.001, boundary - overlapStart), 0.0, 1.0);
  let coarse = select(fine, fine + 1u, blend > 0.0);
  return vec3f(f32(fine), f32(coarse), blend);
}

fn cascadeSpacing(cascade: u32, lod: u32) -> f32 {
  return frame.envBaseSpacing.w * exp2(f32(cascade + (lod & 3u)));
}

fn probeCell(position: vec3f, cascade: u32, lod: u32) -> vec3i {
  return vec3i(floor(position / cascadeSpacing(cascade, lod)));
}

fn probePositionFromCell(cell: vec3i, cascade: u32, lod: u32) -> vec3f {
  return (vec3f(cell) + vec3f(0.5)) * cascadeSpacing(cascade, lod);
}

fn currentFrame() -> u32 {
  return min(1u, u32(floor(frame.controls.w)));
}

fn historyWeight() -> f32 {
  return fract(frame.controls.w);
}

fn featureEnabled(bit:u32)->bool {
  return (u32(frame.cameraPos.w+0.5)&bit)!=0u;
}

fn intervalHistoryWeight()->f32{
  return historyWeight();
}

fn lookupProbeFrame(cascade: u32, key: u32, frameIndex: u32) -> u32 {
  let base = frameIndex * HASH_FRAME_STRIDE + HASH_OFFSETS[cascade];
  let mask = HASH_SIZES[cascade] - 1u;
  let start = hash32(key) & mask;
  for (var step=0u; step<32u; step++) {
    let slot = base + ((start + step) & mask);
    let found = atomicLoad(&slots[slot].key);
    if (found == key) { return atomicLoad(&slots[slot].index); }
    if (found == EMPTY) { return EMPTY; }
  }
  return EMPTY;
}

fn lookupProbe(cascade: u32, key: u32) -> u32 {
  return lookupProbeFrame(cascade, key, currentFrame());
}

fn insertProbeRaw(cascade:u32,key:u32){
  if(key==EMPTY){atomicAdd(&state[6],1u);return;}
  let base = currentFrame() * HASH_FRAME_STRIDE + HASH_OFFSETS[cascade];
  let mask = HASH_SIZES[cascade] - 1u;
  let start = hash32(key) & mask;
  for (var step=0u; step<32u; step++) {
    let slot = base + ((start + step) & mask);
    let result = atomicCompareExchangeWeak(&slots[slot].key, EMPTY, key);
    if (result.exchanged || result.old_value == key) { return; }
  }
  atomicAdd(&state[6], 1u);
}

fn insertProbeKey(cascade:u32,position:vec3f,lod:u32,sheet:u32){
  insertProbeRaw(cascade,keyFromCellSurface(probeCell(position,cascade,lod),lod,sheet));
}

fn dataIndex(cascade: u32, probe: u32, direction: u32) -> u32 {
  return DATA_OFFSETS[cascade] + probe * DIR_COUNTS[cascade] + direction;
}

fn accumIndexFrame(cascade:u32,probe:u32,direction:u32,frameIndex:u32)->u32{
  return frameIndex*ACCUM_FRAME_STRIDE+dataIndex(cascade,probe,direction)*5u;
}

fn accumIndex(cascade:u32,probe:u32,direction:u32)->u32{
  return accumIndexFrame(cascade,probe,direction,currentFrame());
}

fn probeStateIndex(base: u32, cascade: u32, probe: u32) -> u32 {
  return base + PROBE_OFFSETS[cascade] + probe;
}

fn decodeEqualArea(uv: vec2f) -> vec3f {
  let phi = uv.x * TAU;
  let z = uv.y * 2.0 - 1.0;
  let r = sqrt(max(0.0, 1.0-z*z));
  return vec3f(r*cos(phi), r*sin(phi), z);
}

fn mortonDirectionIndex(u:u32,v:u32,cascade:u32)->u32{
  let bits=2u+cascade;
  var result=0u;
  for(var bit=0u;bit<bits;bit++){
    result|=((u>>bit)&1u)<<(bit*2u);
    result|=((v>>bit)&1u)<<(bit*2u+1u);
  }
  return result|(((u>>bits)&1u)<<(bits*2u));
}

fn mortonDirectionCoordinates(index:u32,cascade:u32)->vec2u{
  let bits=2u+cascade;
  var u=0u;
  var v=0u;
  for(var bit=0u;bit<bits;bit++){
    u|=((index>>(bit*2u))&1u)<<bit;
    v|=((index>>(bit*2u+1u))&1u)<<bit;
  }
  u|=((index>>(bits*2u))&1u)<<bits;
  return vec2u(u,v);
}

fn directionIndex(direction: vec3f, cascade: u32) -> u32 {
  let theta = 4u << cascade;
  let width = theta * 2u;
  let uFloat = fract(atan2(direction.y, direction.x) / TAU + 1.0);
  let vFloat = clamp(direction.z * 0.5 + 0.5, 0.0, 0.999999);
  let u=min(width-1u,u32(floor(uFloat*f32(width))));
  let v=min(theta-1u,u32(floor(vFloat*f32(theta))));
  return mortonDirectionIndex(u,v,cascade);
}

fn directionFromIndex(index: u32, cascade: u32) -> vec3f {
  let theta = 4u << cascade;
  let width = theta * 2u;
  let coordinate=mortonDirectionCoordinates(index,cascade);
  let u=coordinate.x;
  let v=coordinate.y;
  return decodeEqualArea(vec2f((f32(u)+0.5)/f32(width),(f32(v)+0.5)/f32(theta)));
}

fn rayBoxNear(origin: vec3f, inverseDirection: vec3f, minB: vec3f, maxB: vec3f, maxDistance: f32) -> f32 {
  let t0 = (minB-origin)*inverseDirection;
  let t1 = (maxB-origin)*inverseDirection;
  let near3 = min(t0,t1);
  let far3 = max(t0,t1);
  let nearT = max(max(near3.x,near3.y),max(near3.z,0.0));
  let farT = min(min(far3.x,far3.y),far3.z);
  return select(maxDistance,nearT,farT>=nearT&&nearT<maxDistance);
}

fn sampleAtlas(uv:vec2f,material:f32)->vec4f{
  if(material<0.0){return vec4f(1);}
  let index=u32(material+0.5);
  // Diffuse ray-hit radiance is deliberately sampled from a low-frequency
  // mip.  Compute shaders have no screen derivatives, and mip zero turns
  // sub-pixel material detail into temporally changing radiance noise.
  let sampled=textureSampleLevel(albedoAtlas,albedoSampler,fract(uv),i32(index),2.0);
  if(!featureEnabled(16u)){return sampled;}
  let luminance=dot(sampled.rgb,vec3f(0.2126,0.7152,0.0722));
  let detail=clamp(0.72+luminance*0.34,0.62,1.04);
  let stone=vec3f(0.93,0.96,0.96)*detail;
  let cyan=vec3f(0.18,0.82,0.86)*mix(0.82,1.08,luminance);
  return vec4f(select(stone,cyan,index>=15u&&index<=19u),sampled.a);
}

fn decodeOctNormal(packed:u32)->vec3f{
  let encoded=vec2f(f32(packed&65535u),f32(packed>>16u))/65535.0*2.0-1.0;
  var normal=vec3f(encoded,1.0-abs(encoded.x)-abs(encoded.y));
  if(normal.z<0.0){
    let old=normal.xy;
    normal.x=(1.0-abs(old.y))*select(-1.0,1.0,old.x>=0.0);
    normal.y=(1.0-abs(old.x))*select(-1.0,1.0,old.y>=0.0);
  }
  return normalize(normal);
}

fn traceTriangle(origin: vec3f, direction: vec3f, tri: Triangle, maxDistance: f32) -> vec3f {
  let edge1=tri.b.xyz-tri.a.xyz;
  let edge2=tri.c.xyz-tri.a.xyz;
  let p=cross(direction,edge2);
  let determinant=dot(edge1,p);
  if(abs(determinant)<1e-7){return vec3f(maxDistance,0,0);}
  let inverse=1.0/determinant;
  let offset=origin-tri.a.xyz;
  let u=dot(offset,p)*inverse;
  let edgeTolerance=2e-6;
  if(u < -edgeTolerance||u>1.0+edgeTolerance){
    return vec3f(maxDistance,0,0);
  }
  let q=cross(offset,edge1);
  let v=dot(direction,q)*inverse;
  if(v < -edgeTolerance||u+v>1.0+edgeTolerance){
    return vec3f(maxDistance,0,0);
  }
  let distance=dot(edge2,q)*inverse;
  if(distance>1e-7&&distance<maxDistance){
    return vec3f(distance,clamp(u,0.0,1.0),clamp(v,0.0,1.0));
  }
  return vec3f(maxDistance,0,0);
}

fn traceScene(origin: vec3f, directionIn: vec3f, maxDistance: f32) -> Hit {
  let direction=normalize(directionIn);
  let inverseDirection=select(vec3f(-1e20),vec3f(1e20),direction>=vec3f(0))
    /max(vec3f(1),abs(direction)*1e20);
  var result=Hit(maxDistance,vec3f(0,1,0),vec3f(0),vec3f(0));
  var stack: array<u32,64>;
  var stackSize=1u;
  stack[0]=0u;
  loop {
    if(stackSize==0u){break;}
    stackSize-=1u;
    let nodeIndex=stack[stackSize];
    let node=bvhNodes[nodeIndex];
    if(rayBoxNear(origin,inverseDirection,node.minMeta.xyz,node.maxMeta.xyz,result.t)>=result.t){continue;}
    let left=bitcast<u32>(node.minMeta.w);
    let right=bitcast<u32>(node.maxMeta.w);
    if((left&0x80000000u)!=0u){
      let first=left&0x7fffffffu;
      for(var j=0u;j<right;j++){
        let tri=triangles[first+j];
        let intersection=traceTriangle(origin,direction,tri,result.t);
        if(intersection.x<result.t){
          let uv=tri.uvAB.xy*(1.0-intersection.y-intersection.z)
            +tri.uvAB.zw*intersection.y+tri.uvCMaterial.xy*intersection.z;
          let surface=sampleAtlas(uv,tri.uvCMaterial.z);
          if(tri.uvCMaterial.w>0.0&&surface.a<tri.uvCMaterial.w){continue;}
          let barycentric=vec3f(1.0-intersection.y-intersection.z,intersection.y,intersection.z);
          var n=normalize(
            decodeOctNormal(tri.normalOct.x)*barycentric.x
            +decodeOctNormal(tri.normalOct.y)*barycentric.y
            +decodeOctNormal(tri.normalOct.z)*barycentric.z
          );
          if(dot(n,direction)>0.0){n=-n;}
          result=Hit(intersection.x,n,tri.albedo.xyz*surface.rgb,tri.emissive.xyz);
        }
      }
    } else {
      let leftNode=bvhNodes[left];
      let rightNode=bvhNodes[right];
      let leftNear=rayBoxNear(origin,inverseDirection,leftNode.minMeta.xyz,leftNode.maxMeta.xyz,result.t);
      let rightNear=rayBoxNear(origin,inverseDirection,rightNode.minMeta.xyz,rightNode.maxMeta.xyz,result.t);
      if(leftNear<result.t&&rightNear<result.t){
        if(stackSize>61u){atomicAdd(&state[7],1u);break;}
        if(leftNear<rightNear){
          stack[stackSize]=right;
          stack[stackSize+1u]=left;
        }else{
          stack[stackSize]=left;
          stack[stackSize+1u]=right;
        }
        stackSize+=2u;
      }else if(leftNear<result.t){
        if(stackSize>62u){atomicAdd(&state[7],1u);break;}
        stack[stackSize]=left;
        stackSize+=1u;
      }else if(rightNear<result.t){
        if(stackSize>62u){atomicAdd(&state[7],1u);break;}
        stack[stackSize]=right;
        stackSize+=1u;
      }
    }
  }
  return result;
}

fn octTexel(normalIn:vec3f)->u32{
  var n=normalize(normalIn);
  n/=max(1e-6,abs(n.x)+abs(n.y)+abs(n.z));
  var f=n.xy;
  if(n.z<0.0){
    f=vec2f(
      (1.0-abs(f.y))*select(-1.0,1.0,f.x>=0.0),
      (1.0-abs(f.x))*select(-1.0,1.0,f.y>=0.0)
    );
  }
  let uv=clamp(f*0.5+0.5,vec2f(0),vec2f(0.99999));
  let xy=vec2u(floor(uv*6.0))+vec2u(1);
  return xy.x+xy.y*8u;
}

fn sampleAtlasIrradiance(probe:u32,normalIn:vec3f,frameIndex:u32)->vec4f{
  var octNormal=normalize(normalIn);
  octNormal/=max(1e-6,abs(octNormal.x)+abs(octNormal.y)+abs(octNormal.z));
  var oct=octNormal.xy;
  if(octNormal.z<0.0){
    oct=vec2f(
      (1.0-abs(oct.y))*select(-1.0,1.0,oct.x>=0.0),
      (1.0-abs(oct.x))*select(-1.0,1.0,oct.y>=0.0)
    );
  }
  let octCoordinate=clamp((oct*0.5+0.5)*6.0+vec2f(0.5),vec2f(0),vec2f(7));
  let tile=vec2f(f32(probe%64u)*8.0,f32(probe/64u)*8.0+f32(frameIndex)*2048.0);
  let atlasUv=(tile+octCoordinate+vec2f(0.5))/vec2f(512.0,4096.0);
  return textureSampleLevel(
    irradianceAtlasSampled,irradianceAtlasSampler,atlasUv,0.0
  );
}

fn samplePrimaryIrradianceLod(position:vec3f,normal:vec3f,lod:u32)->vec4f{
  let spacing=cascadeSpacing(0u,lod);
  let grid=position/spacing-vec3f(0.5);
  let cell=vec3i(floor(grid));
  let fraction=fract(grid);
  let fixedBits=vec3i(floor(position/spacing))-cell;
  let absoluteNormal=abs(normal);
  var normalAxis=2u;
  if(absoluteNormal.x>=absoluteNormal.y&&absoluteNormal.x>=absoluteNormal.z){
    normalAxis=0u;
  }else if(absoluteNormal.y>=absoluteNormal.z){
    normalAxis=1u;
  }
  var normalWeight=0.0;
  if(normalAxis==0u){normalWeight=select(1.0-fraction.x,fraction.x,fixedBits.x==1);}
  if(normalAxis==1u){normalWeight=select(1.0-fraction.y,fraction.y,fixedBits.y==1);}
  if(normalAxis==2u){normalWeight=select(1.0-fraction.z,fraction.z,fixedBits.z==1);}
  var value=vec3f(0);
  var total=0.0;
  for(var corner=0u;corner<4u;corner++){
    var bits=fixedBits;
    if(normalAxis==0u){
      bits.y=i32(corner&1u);bits.z=i32((corner>>1u)&1u);
    }else if(normalAxis==1u){
      bits.x=i32(corner&1u);bits.z=i32((corner>>1u)&1u);
    }else{
      bits.x=i32(corner&1u);bits.y=i32((corner>>1u)&1u);
    }
    let wv=vec3f(
      select(1.0-fraction.x,fraction.x,bits.x==1),
      select(1.0-fraction.y,fraction.y,bits.y==1),
      select(1.0-fraction.z,fraction.z,bits.z==1)
    );
    let spatialWeight=wv.x*wv.y*wv.z;
    let probe=lookupProbe(0u,keyFromCellSurface(cell+bits,lod,surfaceClass(normal)));
    if(probe!=EMPTY&&probe<PROBE_CAPS[0]){
      let irradiance=sampleAtlasIrradiance(probe,normal,currentFrame());
      let activeWeight=spatialWeight*irradiance.a;
      value+=irradiance.xyz*activeWeight;
      total+=activeWeight;
    }
  }
  if(total<1e-5){return vec4f(0,0,0,0);}
  return vec4f(value/total,clamp(total/max(normalWeight,1e-5),0.0,1.0));
}

fn samplePrimaryIrradiance(position:vec3f,normal:vec3f)->vec4f{
  let lods=lodSelection(position);
  // Algorithm 1 shades the world-space surface position itself. Offsetting
  // this gather along the normal hides some dark interpolation cases, but it
  // also moves the query into a different trilinear cell and creates a broad
  // bright-leak bias at walls.
  let fine=samplePrimaryIrradianceLod(position,normal,u32(lods.x));
  if(u32(lods.y)==u32(lods.x)){return fine;}
  let coarse=samplePrimaryIrradianceLod(position,normal,u32(lods.y));
  if(fine.w<0.001){return coarse;}
  if(coarse.w<0.001){return fine;}
  return vec4f(mix(fine.xyz,coarse.xyz,lods.z),mix(fine.w,coarse.w,lods.z));
}

fn samplePrimaryConeDirectionLod(
  position:vec3f,normal:vec3f,lod:u32,direction:u32
)->vec4f{
  let spacing=cascadeSpacing(0u,lod);
  let grid=position/spacing-vec3f(0.5);
  let cell=vec3i(floor(grid));
  let fraction=fract(grid);
  let fixedBits=vec3i(floor(position/spacing))-cell;
  let absoluteNormal=abs(normal);
  var normalAxis=2u;
  if(absoluteNormal.x>=absoluteNormal.y&&absoluteNormal.x>=absoluteNormal.z){
    normalAxis=0u;
  }else if(absoluteNormal.y>=absoluteNormal.z){
    normalAxis=1u;
  }
  var value=vec3f(0);
  var total=0.0;
  for(var corner=0u;corner<4u;corner++){
    var bits=fixedBits;
    if(normalAxis==0u){
      bits.y=i32(corner&1u);bits.z=i32((corner>>1u)&1u);
    }else if(normalAxis==1u){
      bits.x=i32(corner&1u);bits.z=i32((corner>>1u)&1u);
    }else{
      bits.x=i32(corner&1u);bits.y=i32((corner>>1u)&1u);
    }
    let wv=vec3f(
      select(1.0-fraction.x,fraction.x,bits.x==1),
      select(1.0-fraction.y,fraction.y,bits.y==1),
      select(1.0-fraction.z,fraction.z,bits.z==1)
    );
    let weight=wv.x*wv.y*wv.z;
    let probe=lookupProbe(0u,keyFromCellSurface(
      cell+bits,lod,surfaceClass(normal)
    ));
    if(probe!=EMPTY&&probe<PROBE_CAPS[0]){
      let cone=cones[dataIndex(0u,probe,direction)];
      if(cone.w>0.5){value+=cone.xyz*weight;total+=weight;}
    }
  }
  if(total<1e-5){return vec4f(0);}
  return vec4f(value/total,1.0);
}

fn samplePrimaryConeDirection(
  position:vec3f,normal:vec3f,direction:u32
)->vec4f{
  let lods=lodSelection(position);
  let fine=samplePrimaryConeDirectionLod(
    position,normal,u32(lods.x),direction
  );
  if(u32(lods.y)==u32(lods.x)){return fine;}
  let coarse=samplePrimaryConeDirectionLod(
    position,normal,u32(lods.y),direction
  );
  if(fine.w<0.5){return coarse;}
  if(coarse.w<0.5){return fine;}
  return vec4f(mix(fine.xyz,coarse.xyz,lods.z),1.0);
}

fn primaryPointAabbDistanceSquared(
  point:vec3f,minimum:vec3f,maximum:vec3f
)->f32{
  let delta=max(max(minimum-point,vec3f(0)),point-maximum);
  return dot(delta,delta);
}
fn primaryPolygonEdgeIntegral(a:vec3f,b:vec3f)->vec3f{
  let edge=cross(a,b);
  let edgeLength=length(edge);
  if(edgeLength<1e-7){return vec3f(0);}
  return edge*(acos(clamp(dot(a,b),-1.0,1.0))/edgeLength);
}
fn primaryClippedTriangleFormFactor(
  a:vec3f,b:vec3f,c:vec3f,normal:vec3f
)->f32{
  var input=array<vec3f,5>(a,b,c,vec3f(0),vec3f(0));
  var clipped:array<vec3f,5>;
  var clippedCount=0u;
  var previous=input[2];
  var previousDistance=dot(normal,previous);
  var previousInside=previousDistance>0.0;
  for(var index=0u;index<3u;index++){
    let current=input[index];
    let currentDistance=dot(normal,current);
    let currentInside=currentDistance>0.0;
    if(currentInside!=previousInside){
      let t=previousDistance/(previousDistance-currentDistance);
      clipped[clippedCount]=mix(previous,current,clamp(t,0.0,1.0));
      clippedCount++;
    }
    if(currentInside){clipped[clippedCount]=current;clippedCount++;}
    previous=current;
    previousDistance=currentDistance;
    previousInside=currentInside;
  }
  if(clippedCount<3u){return 0.0;}
  var vectorForm=vec3f(0);
  for(var index=0u;index<clippedCount;index++){
    let next=(index+1u)%clippedCount;
    vectorForm+=primaryPolygonEdgeIntegral(
      normalize(clipped[index]),normalize(clipped[next])
    );
  }
  return abs(dot(normal,vectorForm))*0.15915494309189535;
}
fn primaryEmissiveTriangleIrradiance(
  origin:vec3f,normal:vec3f,triangle:Triangle,radius:f32
)->vec3f{
  if(max(triangle.emissive.x,max(triangle.emissive.y,triangle.emissive.z))<=0.0){
    return vec3f(0);
  }
  let minimum=min(triangle.a.xyz,min(triangle.b.xyz,triangle.c.xyz));
  let maximum=max(triangle.a.xyz,max(triangle.b.xyz,triangle.c.xyz));
  let proximityDistance=sqrt(primaryPointAabbDistanceSquared(
    origin,minimum,maximum
  ));
  if(proximityDistance>=radius){return vec3f(0);}
  let aDelta=triangle.a.xyz-origin;
  let bDelta=triangle.b.xyz-origin;
  let cDelta=triangle.c.xyz-origin;
  if(min(length(aDelta),min(length(bDelta),length(cDelta)))<1e-5){
    return vec3f(0);
  }
  let formFactor=primaryClippedTriangleFormFactor(
    aDelta,bDelta,cDelta,normal
  );
  if(formFactor<=1e-7){return vec3f(0);}
  let edgeAB=triangle.b.xyz-triangle.a.xyz;
  let edgeAC=triangle.c.xyz-triangle.a.xyz;
  let fromA=origin-triangle.a.xyz;
  let d00=dot(edgeAB,edgeAB);
  let d01=dot(edgeAB,edgeAC);
  let d11=dot(edgeAC,edgeAC);
  let d20=dot(fromA,edgeAB);
  let d21=dot(fromA,edgeAC);
  let inverseDenominator=1.0/max(1e-12,d00*d11-d01*d01);
  var barycentric=vec3f(
    1.0-(d11*d20-d01*d21)*inverseDenominator
      -(d00*d21-d01*d20)*inverseDenominator,
    (d11*d20-d01*d21)*inverseDenominator,
    (d00*d21-d01*d20)*inverseDenominator
  );
  barycentric=max(barycentric,vec3f(0));
  barycentric/=max(1e-8,barycentric.x+barycentric.y+barycentric.z);
  let closest=triangle.a.xyz*barycentric.x
    +triangle.b.xyz*barycentric.y+triangle.c.xyz*barycentric.z;
  let toLight=closest-origin;
  let lightDistance=length(toLight);
  if(lightDistance>0.012){
    let blocker=traceScene(
      origin,toLight/lightDistance,max(1e-5,lightDistance-0.008)
    );
    if(blocker.t<lightDistance-0.01){return vec3f(0);}
  }
  let proximity=1.0-smoothstep(radius*0.72,radius,proximityDistance);
  return triangle.emissive.xyz*(formFactor*proximity);
}
fn primaryNearEmissiveIrradiance(
  origin:vec3f,normal:vec3f,radius:f32
)->vec3f{
  var result=vec3f(0);
  var stack:array<u32,64>;
  var stackSize=1u;
  stack[0]=0u;
  let radiusSquared=radius*radius;
  loop{
    if(stackSize==0u){break;}
    stackSize-=1u;
    let nodeIndex=stack[stackSize];
    let node=bvhNodes[nodeIndex];
    if(primaryPointAabbDistanceSquared(
      origin,node.minMeta.xyz,node.maxMeta.xyz
    )>radiusSquared){continue;}
    let left=bitcast<u32>(node.minMeta.w);
    let right=bitcast<u32>(node.maxMeta.w);
    if((left&0x80000000u)!=0u){
      let first=left&0x7fffffffu;
      for(var triangleOffset=0u;triangleOffset<right;triangleOffset++){
        result+=primaryEmissiveTriangleIrradiance(
          origin,normal,triangles[first+triangleOffset],radius
        );
      }
    }else{
      if(stackSize>61u){break;}
      stack[stackSize]=left;
      stack[stackSize+1u]=right;
      stackSize+=2u;
    }
  }
  return result;
}

fn directAtHit(position: vec3f, hit: Hit) -> vec3f {
  let lightDirection=normalize(-frame.sunDirTime.xyz);
  let ndl=max(0.0,dot(hit.normal,lightDirection));
  var visibility=1.0;
  if(ndl>0.001){
    let shadowHit=traceScene(position+hit.normal*0.015,lightDirection,10000.0);
    if(shadowHit.t<9999.0){visibility=0.0;}
  }
  let toPoint=frame.pointPosRange.xyz-position;
  let pointDistance=length(toPoint);
  let pointDirection=toPoint/max(pointDistance,1e-4);
  let pointWindow=max(0.0,1.0-pointDistance/frame.pointPosRange.w);
  let pointAttenuation=pointWindow*pointWindow/(1.0+0.06*pointDistance*pointDistance);
  var point=max(0.0,dot(hit.normal,pointDirection))*pointAttenuation*frame.pointColorIntensity.xyz*frame.pointColorIntensity.w;
  if(any(point>vec3f(0.0001))){
    let pointShadow=traceScene(position+hit.normal*0.015,pointDirection,max(0.0,pointDistance-0.03));
    if(pointShadow.t<pointDistance-0.04){point=vec3f(0);}
  }
  let sun=frame.sunColorIntensity.xyz*frame.sunColorIntensity.w*ndl*visibility;
  return min(vec3f(16.0),hit.emissive.xyz+hit.albedo.xyz*(sun+point));
}
`;

export const computeShader = sharedCompute + /* wgsl */`
@compute @workgroup_size(256) fn resetSlots(@builtin(global_invocation_id) gid: vec3u) {
  if(gid.x>=HASH_FRAME_STRIDE){return;}
  let slot=currentFrame()*HASH_FRAME_STRIDE+gid.x;
  atomicStore(&slots[slot].key,EMPTY);
  atomicStore(&slots[slot].index,EMPTY);
}

fn insertTangentSupport(position:vec3f,normal:vec3f,lod:u32){
  let spacing=cascadeSpacing(0u,lod);
  let cell=vec3i(floor(position/spacing-vec3f(0.5)));
  let anchor=probeCell(position,0u,lod);
  let fixedBits=anchor-cell;
  let absoluteNormal=abs(normal);
  var normalAxis=2u;
  if(absoluteNormal.x>=absoluteNormal.y&&absoluteNormal.x>=absoluteNormal.z){
    normalAxis=0u;
  }else if(absoluteNormal.y>=absoluteNormal.z){
    normalAxis=1u;
  }
  for(var corner=0u;corner<4u;corner++){
    var bits=fixedBits;
    if(normalAxis==0u){
      bits.y=i32(corner&1u);
      bits.z=i32((corner>>1u)&1u);
    }else if(normalAxis==1u){
      bits.x=i32(corner&1u);
      bits.z=i32((corner>>1u)&1u);
    }else{
      bits.x=i32(corner&1u);
      bits.y=i32((corner>>1u)&1u);
    }
    insertProbeRaw(0u,keyFromCellSurface(cell+bits,lod,surfaceClass(normal)));
  }
}

@compute @workgroup_size(8,8) fn initBase(@builtin(global_invocation_id) gid: vec3u) {
  let fullSize=vec2u(u32(frame.resolution.x),u32(frame.resolution.y));
  if(any(gid.xy>=fullSize)){return;}
  // The paper notes that full trilinear support roughly doubles probe count
  // and uses nearest-only as a performance tradeoff.  Four tangent-plane
  // neighbors give floors and walls complete deterministic surface support
  // without allocating the four probes across the surface normal. Algorithm 3
  // still assigns its one traced ray to the nearest probe.
  let world=textureLoad(worldTex,vec2i(gid.xy),0);
  if(world.w<0.5){return;}
  let normal=gbufferNormal(vec2i(gid.xy));
  let lods=lodSelection(world.xyz);
  let fine=u32(lods.x);
  insertTangentSupport(world.xyz,normal,fine);
  let coarse=u32(lods.y);
  if(coarse!=fine){insertTangentSupport(world.xyz,normal,coarse);}
}

fn samplesPerFrame()->u32{
  let giSize=vec2u(u32(frame.resolution.z),u32(frame.resolution.w));
  return giSize.x*giSize.y*max(1u,passParams.value);
}

fn sampleIndex(gid:vec3u)->u32{
  let giSize=vec2u(u32(frame.resolution.z),u32(frame.resolution.w));
  return (gid.z*giSize.y+gid.y)*giSize.x+gid.x;
}

fn raySampleProbeBase()->u32{
  return TOTAL_PROBE_META;
}

fn rayBlockSize()->u32{
  return max(1u,passParams.pad0);
}

fn rayBlockCount()->u32{
  return (samplesPerFrame()*2u+rayBlockSize()-1u)/rayBlockSize();
}

fn rayBlockStateIndex(probe:u32,block:u32)->u32{
  return BLOCK_COUNT_OFFSET+probe*rayBlockCount()+block;
}

@compute @workgroup_size(64) fn initHigher(@builtin(global_invocation_id) gid: vec3u) {
  let cascade=passParams.cascade;
  if(cascade==0u||cascade>3u){return;}
  let previous=cascade-1u;
  let activeCount=min(atomicLoad(&state[previous]),PROBE_CAPS[previous]);
  if(gid.x>=activeCount){return;}
  let probeInfo=probeMeta[PROBE_OFFSETS[previous]+gid.x];
  insertProbeKey(cascade,probeInfo.xyz,probeLod(probeInfo),probeSurfaceClass(probeInfo));
}

fn cellFromKey(key:u32)->vec3i{
  return vec3i(
    i32(key&511u)-256,
    i32((key>>9u)&511u)-256,
    i32((key>>18u)&511u)-256
  );
}

fn lodFromKey(key:u32)->u32{
  return (key>>27u)&3u;
}

fn surfaceClassFromKey(key:u32)->u32{
  return (key>>29u)&7u;
}

fn probeMetaBits(info:vec4f)->u32{
  return bitcast<u32>(info.w);
}

fn probeLod(info:vec4f)->u32{
  return probeMetaBits(info)&3u;
}

fn probeSurfaceClass(info:vec4f)->u32{
  return (probeMetaBits(info)>>2u)&7u;
}

// Key insertion and compact-index publication are separate passes. Probe
// indices are storage addresses only: every ordering decision below is made
// from the stable packed key, so the allocation order cannot alter the R2
// sequence or radiance result. Atomic compaction avoids the former O(H^2)
// all-pairs rank pass (over 40 million comparisons per frame on Sponza).
@compute @workgroup_size(64) fn canonicalizeProbes(@builtin(global_invocation_id) gid:vec3u){
  let cascade=passParams.cascade;
  if(gid.x>=HASH_SIZES[cascade]){return;}
  let base=currentFrame()*HASH_FRAME_STRIDE+HASH_OFFSETS[cascade];
  let slot=base+gid.x;
  let key=atomicLoad(&slots[slot].key);
  if(key==EMPTY){return;}
  let compactIndex=atomicAdd(&state[cascade],1u);
  if(compactIndex>=PROBE_CAPS[cascade]){
    atomicStore(&slots[slot].index,EMPTY);
    atomicAdd(&state[6],1u);
    return;
  }
  atomicStore(&slots[slot].index,compactIndex);
  let lod=lodFromKey(key);
  probeMeta[PROBE_OFFSETS[cascade]+compactIndex]=vec4f(
    probePositionFromCell(cellFromKey(key),cascade,lod),
    bitcast<f32>(lod|(surfaceClassFromKey(key)<<2u))
  );
}

@compute @workgroup_size(8,8) fn countBaseRays(@builtin(global_invocation_id) gid:vec3u){
  let giSize=vec2u(u32(frame.resolution.z),u32(frame.resolution.w));
  if(any(gid.xy>=giSize)){return;}
  let fullSize=vec2u(u32(frame.resolution.x),u32(frame.resolution.y));
  let pixel=min(fullSize-vec2u(1),gid.xy*fullSize/giSize);
  let world=textureLoad(worldTex,vec2i(pixel),0);
  if(world.w<0.5){return;}
  let normal=gbufferNormal(vec2i(pixel));
  let sheet=surfaceClass(normal);
  let lods=lodSelection(world.xyz);
  let fine=u32(lods.x);
  let fineProbe=lookupProbe(0u,keyFromCellSurface(probeCell(world.xyz,0u,fine),fine,sheet));
  if(fineProbe!=EMPTY){
    atomicAdd(&state[probeStateIndex(RAY_COUNT_OFFSET,0u,fineProbe)],max(1u,passParams.value));
  }
  let coarse=u32(lods.y);
  if(coarse!=fine){
    let coarseProbe=lookupProbe(0u,keyFromCellSurface(probeCell(world.xyz,0u,coarse),coarse,sheet));
    if(coarseProbe!=EMPTY){
      atomicAdd(&state[probeStateIndex(RAY_COUNT_OFFSET,0u,coarseProbe)],max(1u,passParams.value));
    }
  }
}

@compute @workgroup_size(64) fn countHigherRays(@builtin(global_invocation_id) gid:vec3u){
  let cascade=passParams.cascade;
  if(cascade==0u||cascade>3u){return;}
  let previous=cascade-1u;
  let childCount=min(atomicLoad(&state[previous]),PROBE_CAPS[previous]);
  if(gid.x>=childCount){return;}
  let childInfo=probeMeta[PROBE_OFFSETS[previous]+gid.x];
  let lod=probeLod(childInfo);
  let parent=lookupProbe(cascade,keyFromCellSurface(
    probeCell(childInfo.xyz,cascade,lod),lod,probeSurfaceClass(childInfo)
  ));
  if(parent!=EMPTY){
    let childRays=atomicLoad(&state[probeStateIndex(RAY_COUNT_OFFSET,previous,gid.x)]);
    atomicAdd(&state[probeStateIndex(RAY_COUNT_OFFSET,cascade,parent)],childRays);
  }
}

fn probeKeyFromInfo(info:vec4f,cascade:u32)->u32{
  let lod=probeLod(info);
  return keyFromCellSurface(
    probeCell(info.xyz,cascade,lod),lod,probeSurfaceClass(info)
  );
}

fn parentKeyFromInfo(info:vec4f,cascade:u32)->u32{
  let lod=probeLod(info);
  return keyFromCellSurface(
    probeCell(info.xyz,cascade+1u,lod),lod,probeSurfaceClass(info)
  );
}

// Algorithm 3's reverse hierarchical prefix sum. Probe indices may be
// allocated in any GPU order, so key ordering makes the resulting offsets
// deterministic across frames.
@compute @workgroup_size(64) fn assignRayOffsets(@builtin(global_invocation_id) gid: vec3u) {
  let cascade=passParams.cascade;
  let activeCount=min(atomicLoad(&state[cascade]),PROBE_CAPS[cascade]);
  if(gid.x>=activeCount){return;}
  let info=probeMeta[PROBE_OFFSETS[cascade]+gid.x];
  let key=probeKeyFromInfo(info,cascade);
  var prefix=0u;
  if(cascade==3u){
    for(var other=0u;other<activeCount;other++){
      let otherInfo=probeMeta[PROBE_OFFSETS[cascade]+other];
      if(probeKeyFromInfo(otherInfo,cascade)<key){
        prefix+=atomicLoad(&state[probeStateIndex(RAY_COUNT_OFFSET,cascade,other)]);
      }
    }
  } else {
    let parentKey=parentKeyFromInfo(info,cascade);
    let parent=lookupProbe(cascade+1u,parentKey);
    if(parent==EMPTY){return;}
    prefix=atomicLoad(&state[probeStateIndex(RAY_OFFSET_OFFSET,cascade+1u,parent)]);
    // Adjacent cascade spacings differ by exactly two, so every parent has
    // at most eight spatial children. Enumerating those cells preserves the
    // stable key order required by Algorithm 3 without an O(probeCount^2)
    // scan that dominated some D3D12/Metal WebGPU implementations.
    let lod=probeLod(info);
    let parentCell=probeCell(info.xyz,cascade+1u,lod);
    let sheet=probeSurfaceClass(info);
    for(var child=0u;child<8u;child++){
      let bits=vec3i(i32(child&1u),i32((child>>1u)&1u),i32((child>>2u)&1u));
      let siblingKey=keyFromCellSurface(parentCell*2+bits,lod,sheet);
      if(siblingKey<key){
        let sibling=lookupProbe(cascade,siblingKey);
        if(sibling!=EMPTY){
          prefix+=atomicLoad(&state[probeStateIndex(RAY_COUNT_OFFSET,cascade,sibling)]);
        }
      }
    }
  }
  atomicStore(&state[probeStateIndex(RAY_OFFSET_OFFSET,cascade,gid.x)],prefix);
}

fn deposit(cascade:u32,probe:u32,direction:u32,radiance:vec3f,beta:f32){
  let base=accumIndex(cascade,probe,direction);
  let safe=min(max(radiance,vec3f(0)),vec3f(16));
  atomicAdd(&accum[base],u32(safe.r*FIXED_SCALE+0.5));
  atomicAdd(&accum[base+1u],u32(safe.g*FIXED_SCALE+0.5));
  atomicAdd(&accum[base+2u],u32(safe.b*FIXED_SCALE+0.5));
  atomicAdd(&accum[base+3u],u32(clamp(beta,0.0,1.0)*FIXED_SCALE+0.5));
  atomicAdd(&accum[base+4u],1u);
}

fn mapRaySample(world:vec3f,normal:vec3f,lod:u32,stableSlot:u32){
  let probe=lookupProbe(0u,keyFromCellSurface(
    probeCell(world,0u,lod),lod,surfaceClass(normal)
  ));
  if(probe==EMPTY){return;}
  probeMeta[raySampleProbeBase()+stableSlot]=vec4f(
    bitcast<f32>(probe),
    bitcast<f32>(passParams.sampleEpoch),
    0,
    1
  );
  atomicAdd(&state[rayBlockStateIndex(probe,stableSlot/rayBlockSize())],1u);
}

@compute @workgroup_size(8,8) fn mapPrimaryRaySamples(@builtin(global_invocation_id) gid:vec3u){
  let giSize=vec2u(u32(frame.resolution.z),u32(frame.resolution.w));
  if(any(gid.xy>=giSize)||gid.z>=max(1u,passParams.value)){return;}
  let fullSize=vec2u(u32(frame.resolution.x),u32(frame.resolution.y));
  let pixel=min(fullSize-vec2u(1),gid.xy*fullSize/giSize);
  let world=textureLoad(worldTex,vec2i(pixel),0);
  if(world.w<0.5){return;}
  let normal=gbufferNormal(vec2i(pixel));
  let lods=lodSelection(world.xyz);
  let sample=sampleIndex(gid);
  mapRaySample(world.xyz,normal,u32(lods.x),sample);
  if(u32(lods.y)!=u32(lods.x)){
    mapRaySample(world.xyz,normal,u32(lods.y),samplesPerFrame()+sample);
  }
}

// Convert per-block sample counts to deterministic block prefixes. Combined
// with the bounded in-block scan below, this gives each ray its exact rank in
// stable screen order without atomic arrival-order dependence.
@compute @workgroup_size(64) fn prefixRayBlocks(
  @builtin(workgroup_id) workgroup:vec3u,
  @builtin(local_invocation_id) local:vec3u
){
  let probe=workgroup.x;
  let activeCount=min(atomicLoad(&state[0]),PROBE_CAPS[0]);
  if(probe>=activeCount||local.x!=0u){return;}
  var prefix=0u;
  for(var block=0u;block<rayBlockCount();block++){
    let index=rayBlockStateIndex(probe,block);
    let count=atomicLoad(&state[index]);
    atomicStore(&state[index],prefix);
    prefix+=count;
  }
  if(prefix!=atomicLoad(&state[probeStateIndex(RAY_COUNT_OFFSET,0u,probe)])){
    atomicAdd(&state[7],1u);
  }
}

fn mappedProbe(stableSlot:u32)->u32{
  let entry=probeMeta[raySampleProbeBase()+stableSlot];
  return select(EMPTY,bitcast<u32>(entry.x),bitcast<u32>(entry.y)==passParams.sampleEpoch);
}

fn deterministicLocalRank(stableSlot:u32,probe:u32)->u32{
  let block=stableSlot/rayBlockSize();
  var rank=atomicLoad(&state[rayBlockStateIndex(probe,block)]);
  let first=block*rayBlockSize();
  for(var other=first;other<stableSlot;other++){
    if(mappedProbe(other)==probe){rank+=1u;}
  }
  return rank;
}

fn traceAndSplit(world:vec3f,normal:vec3f,lod:u32,stableSlot:u32){
  let sheet=surfaceClass(normal);
  let key0=keyFromCellSurface(probeCell(world,0u,lod),lod,sheet);
  let baseProbe=lookupProbe(0u,key0);
  if(baseProbe==EMPTY){return;}
  let localRank=deterministicLocalRank(stableSlot,baseProbe);
  let sequenceIndex=atomicLoad(&state[probeStateIndex(RAY_OFFSET_OFFSET,0u,baseProbe)])+localRank;
  let g=1.324717957244746;
  let alpha=vec2f(1.0/g,1.0/(g*g));
  // Algorithm 3 applies a single Cranley-Patterson rotation to the complete
  // R2 sequence each frame.  Stable mode still advances that sequence: its
  // deterministic frame-local rotation is what lets interval accumulation
  // converge instead of tracing the same rays forever.  The non-stable path
  // retains wall-clock scrambling for the paper's single-frame comparison.
  // A continuously advancing deterministic rotation matches Section 5.2:
  // every probe that remains visible keeps receiving new samples instead of becoming
  // dependent on the camera/history state at an arbitrary freeze frame.
  let sampleFrame=passParams.sampleFrame;
  // An odd 32-bit Weyl multiplier is a permutation of all u32 values. Its low
  // and high halves approximate the paper's two irrational temporal rotations
  // (0.75488, 0.56984), while the complete 2D pair cannot repeat before the
  // full 2^32-frame cycle and never loses precision through a large f32 cast.
  let scrambledFrame=sampleFrame*0x91e1c141u;
  var jitter=vec2f(
    f32(scrambledFrame&65535u),
    f32(scrambledFrame>>16u)
  )/65536.0;
  if(!featureEnabled(8u)){
    let temporal=hash32(u32(frame.sunDirTime.w*60.0));
    let temporal2=hash32(temporal^0x9e3779b9u);
    jitter=vec2f(f32(temporal&65535u),f32(temporal2&65535u))/65536.0;
  }
  let uv=fract(vec2f(0.5)+f32(sequenceIndex+1u)*alpha+jitter);
  var direction=decodeEqualArea(uv);
  if(dot(direction,normal)<0.0){direction=-direction;}

  let baseLength=frame.envBaseSpacing.w*exp2(f32(lod&3u))*1.6;
  let maxDistance=baseLength*64.0;
  // Section 7.1 starts c0 after the C(-1) interval.  The previous build
  // traced c0 from the receiving surface and then tried to replace a much
  // larger interval in the composite.  Besides double-counting near hits,
  // that made the result switch discontinuously at an enclosure test.
  // Keep the split point in world space and independent of camera LOD: C(-1)
  // owns [0, delta_s0], while c0 and higher cascades own the remaining ray.
  let cMinusOneEnd=frame.envBaseSpacing.w;
  let surfaceOrigin=world+normal*max(0.008,frame.envBaseSpacing.w*0.012);
  let origin=surfaceOrigin+direction*cMinusOneEnd;
  let remainingDistance=max(0.001,maxDistance-cMinusOneEnd);
  let hit=traceScene(origin,direction,remainingDistance+0.001);
  let didHit=hit.t<remainingDistance;
  var targetCascade=3u;
  if(didHit){
    targetCascade=0u;
    var end=baseLength;
    let intervalDistance=cMinusOneEnd+hit.t;
    loop {
      if(intervalDistance<=end||targetCascade>=3u){break;}
      targetCascade+=1u; end*=4.0;
    }
  }
  let radiance=select(frame.envBaseSpacing.xyz,directAtHit(origin+direction*hit.t,hit),didHit);
  atomicAdd(&state[4],1u);
  if(didHit){atomicAdd(&state[5],1u);}
  for(var cascade=0u;cascade<4u;cascade++){
    if(cascade>targetCascade){break;}
    let key=keyFromCellSurface(
      probeCell(world,cascade,lod),lod,sheet
    );
    let probe=lookupProbe(cascade,key);
    if(probe==EMPTY){continue;}
    let dir=directionIndex(direction,cascade);
    if(cascade<targetCascade){deposit(cascade,probe,dir,vec3f(0),1.0);}
    else {deposit(cascade,probe,dir,radiance,0.0);}
  }
}

@compute @workgroup_size(8,8) fn splitRays(@builtin(global_invocation_id) gid: vec3u) {
  let giSize=vec2u(u32(frame.resolution.z),u32(frame.resolution.w));
  if(any(gid.xy>=giSize)||gid.z>=max(1u,passParams.value)){return;}
  let fullSize=vec2u(u32(frame.resolution.x),u32(frame.resolution.y));
  let pixel=min(fullSize-vec2u(1),gid.xy*fullSize/giSize);
  let world=textureLoad(worldTex,vec2i(pixel),0);
  if(world.w<0.5){return;}
  let normal=gbufferNormal(vec2i(pixel));
  let lods=lodSelection(world.xyz);
  let fine=u32(lods.x);
  let sample=sampleIndex(gid);
  traceAndSplit(world.xyz,normal,fine,sample);
  let coarse=u32(lods.y);
  if(coarse!=fine){traceAndSplit(world.xyz,normal,coarse,samplesPerFrame()+sample);}
}

fn sampleParentDirection(
  cascade:u32,position:vec3f,lod:u32,sheet:u32,parentDirection:u32
)->vec4f{
  let parent=cascade+1u;
  let spacing=cascadeSpacing(parent,lod);
  let grid=position/spacing-vec3f(0.5);
  let cell=vec3i(floor(grid));
  let fraction=fract(grid);
  var value=vec3f(0);
  var total=0.0;
  for(var corner=0u;corner<8u;corner++){
    let bits=vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u));
    let wv=vec3f(
      select(1.0-fraction.x,fraction.x,bits.x==1),
      select(1.0-fraction.y,fraction.y,bits.y==1),
      select(1.0-fraction.z,fraction.z,bits.z==1)
    );
    let weight=wv.x*wv.y*wv.z;
    let probe=lookupProbe(parent,keyFromCellSurface(cell+bits,lod,sheet));
    if(probe!=EMPTY){
      let cone=cones[dataIndex(parent,probe,parentDirection)];
      if(cone.w>0.5){
        value+=cone.xyz*weight;
        total+=weight;
      }
    }
  }
  if(total<1e-5){return vec4f(0);}
  return vec4f(value/total,1.0);
}

fn mergedParent(
  cascade:u32,direction:u32,position:vec3f,lod:u32,sheet:u32
)->vec4f{
  var sum=vec3f(0);
  var valid=0.0;
  for(var child=0u;child<4u;child++){
    // Morton ordering makes the four angular children of every lower
    // direction one contiguous aligned group, matching Section 6.
    let parentDirection=direction*4u+child;
    let parent=sampleParentDirection(cascade,position,lod,sheet,parentDirection);
    if(parent.w>0.5){
      sum+=parent.xyz;
      valid+=1.0;
    }
  }
  if(valid<0.5){return vec4f(0);}
  return vec4f(sum/valid,1.0);
}

@compute @workgroup_size(64) fn mergeCascade(@builtin(global_invocation_id) gid: vec3u) {
  let cascade=passParams.cascade;
  let count=DIR_COUNTS[cascade];
  let probe=gid.x/count;
  let direction=gid.x-probe*count;
  let activeCount=min(atomicLoad(&state[cascade]),PROBE_CAPS[cascade]);
  if(probe>=activeCount){return;}
  let index=dataIndex(cascade,probe,direction);
  let base=accumIndex(cascade,probe,direction);
  let samples=atomicLoad(&accum[base+4u]);
  var interval=vec3f(0);
  var beta=1.0;
  var hasInterval=samples>0u;
  var resolvedSamples=samples;
  if(samples>0u){
    let denominator=FIXED_SCALE*f32(samples);
    interval=vec3f(f32(atomicLoad(&accum[base])),f32(atomicLoad(&accum[base+1u])),f32(atomicLoad(&accum[base+2u])))/denominator;
    beta=f32(atomicLoad(&accum[base+3u]))/denominator;
  }
  let probeInfo=probeMeta[PROBE_OFFSETS[cascade]+probe];
  let lod=probeLod(probeInfo);
  let sheet=probeSurfaceClass(probeInfo);
  let key=probeKeyFromInfo(probeInfo,cascade);
  let previousFrame=1u-currentFrame();
  let previousProbe=lookupProbeFrame(cascade,key,previousFrame);
  if(historyWeight()>0.0&&previousProbe!=EMPTY&&previousProbe<PROBE_CAPS[cascade]){
    let previousBase=accumIndexFrame(cascade,previousProbe,direction,previousFrame);
    let previousSamples=atomicLoad(&accum[previousBase+4u]);
    if(previousSamples>0u){
      hasInterval=true;
      let previousDenominator=FIXED_SCALE*f32(previousSamples);
      let previousInterval=vec3f(
        f32(atomicLoad(&accum[previousBase])),
        f32(atomicLoad(&accum[previousBase+1u])),
        f32(atomicLoad(&accum[previousBase+2u]))
      )/previousDenominator;
      let previousBeta=f32(atomicLoad(&accum[previousBase+3u]))/previousDenominator;
      if(samples>0u){
        if(featureEnabled(32u)){
          // Moving lighting needs a bounded response time.
          let temporalWeight=intervalHistoryWeight();
          interval=mix(interval,previousInterval,temporalWeight);
          beta=mix(beta,previousBeta,temporalWeight);
          resolvedSamples=1u;
        }else{
          // Section 5.2 accumulates rays for semi-static scenes. Preserve an
          // effective sample count so repeated exact-key probes converge as a
          // true running average instead of a path-dependent fixed EMA.
          let boundedPrevious=min(previousSamples,16384u);
          let totalSamples=samples+boundedPrevious;
          interval=(
            interval*f32(samples)+previousInterval*f32(boundedPrevious)
          )/f32(max(1u,totalSamples));
          beta=(
            beta*f32(samples)+previousBeta*f32(boundedPrevious)
          )/f32(max(1u,totalSamples));
          resolvedSamples=min(16384u,totalSamples);
        }
      }else{
        interval=previousInterval;
        beta=previousBeta;
        resolvedSamples=previousSamples;
      }
    }
  }
  if(hasInterval){
    let safeInterval=clamp(interval,vec3f(0),vec3f(16));
    let storedSamples=max(1u,resolvedSamples);
    let storageScale=FIXED_SCALE*f32(storedSamples);
    atomicStore(&accum[base],u32(safeInterval.r*storageScale+0.5));
    atomicStore(&accum[base+1u],u32(safeInterval.g*storageScale+0.5));
    atomicStore(&accum[base+2u],u32(safeInterval.b*storageScale+0.5));
    atomicStore(&accum[base+3u],u32(clamp(beta,0.0,1.0)*storageScale+0.5));
    atomicStore(&accum[base+4u],storedSamples);
  }
  var distant=vec4f(frame.envBaseSpacing.xyz,1.0);
  if(cascade<3u){distant=mergedParent(cascade,direction,probeInfo.xyz,lod,sheet);}
  if(!hasInterval){
    // Full trilinear support introduces c0 probes that own no screen ray.
    // They have no near interval, so use the already filtered parent cone as
    // a conservative low-frequency estimate.  Other missing directions stay
    // invalid exactly as Section 5 requires.
    cones[index]=select(vec4f(0),distant,cascade==0u&&distant.w>0.5);
    return;
  }
  if(beta>0.999&&distant.w<0.5){
    cones[index]=vec4f(0);
    return;
  }
  cones[index]=vec4f(
    min(vec3f(16.0),interval+clamp(beta,0.0,1.0)*distant.xyz),
    1.0
  );
}

fn decodeOctahedral(uvIn:vec2f)->vec3f{
  var f=uvIn*2.0-1.0;
  var n=vec3f(f.x,f.y,1.0-abs(f.x)-abs(f.y));
  if(n.z<0.0){
    let old=n.xy;
    n.x=(1.0-abs(old.y))*select(-1.0,1.0,old.x>=0.0);
    n.y=(1.0-abs(old.x))*select(-1.0,1.0,old.y>=0.0);
  }
  return normalize(n);
}

@compute @workgroup_size(64) fn prefilterIrradiance(@builtin(global_invocation_id) gid: vec3u) {
  let probe=gid.x/64u;
  let texel=gid.x-probe*64u;
  let activeCount=min(atomicLoad(&state[0]),PROBE_CAPS[0]);
  if(probe>=activeCount){return;}
  let x=texel%8u;
  let y=texel/8u;
  // The 6x6 field occupies [1,6]^2. Evaluating the octahedral extension in
  // the one-texel border is equivalent to the paper's seam-copying pass.
  let normal=decodeOctahedral(vec2f((f32(x)-0.5)/6.0,(f32(y)-0.5)/6.0));
  var result=vec3f(0);
  for(var direction=0u;direction<32u;direction++){
    let ray=directionFromIndex(direction,0u);
    let cone=cones[dataIndex(0u,probe,direction)];
    if(cone.w>0.5){
      result+=cone.xyz*max(0.0,dot(normal,ray));
    }
  }
  let filtered=result*(4.0/32.0);
  let stored=vec4f(filtered,1.0);
  irradiance[currentFrame()*IRRADIANCE_FRAME_STRIDE+probe*64u+texel]=stored;
  let tile=vec2u(probe%64u,probe/64u)*8u;
  let atlasCoordinate=vec2i(tile+vec2u(x,y)+vec2u(0u,currentFrame()*2048u));
  textureStore(irradianceAtlasStorage,atlasCoordinate,stored);
}

// Mirror the production directional C(-1) interval merge in the path oracle.
fn primaryNeedsLocalInterval(
  pixel:vec2i,world:vec3f,normal:vec3f,backFacingSurface:bool
)->bool{
  if(backFacingSurface){return true;}
  let dimensions=vec2i(textureDimensions(worldTex));
  let offsets=array<vec2i,8>(
    vec2i(-4,0),vec2i(4,0),vec2i(0,-4),vec2i(0,4),
    vec2i(-4,-4),vec2i(4,-4),vec2i(-4,4),vec2i(4,4)
  );
  for(var index=0u;index<8u;index++){
    let samplePixel=pixel+offsets[index];
    if(any(samplePixel<vec2i(0))||any(samplePixel>=dimensions)){return true;}
    let neighbor=textureLoad(worldTex,samplePixel,0);
    if(neighbor.w<0.5){return true;}
    let neighborNormal=gbufferNormal(samplePixel);
    if(dot(normal,neighborNormal)<0.97){return true;}
    if(abs(dot(neighbor.xyz-world,normal))>frame.envBaseSpacing.w*0.075){
      return true;
    }
  }
  return false;
}

fn resolvedPrimaryIrradiance(
  pixel:vec2i,world:vec3f,normal:vec3f,backFacingSurface:bool
)->vec3f{
  let base=samplePrimaryIrradiance(world,normal).xyz;
  let intervalEnd=frame.envBaseSpacing.w;
  let origin=world+normal*max(0.006,intervalEnd*0.012);
  let nearEmission=primaryNearEmissiveIrradiance(
    origin,normal,intervalEnd
  );
  if(!primaryNeedsLocalInterval(pixel,world,normal,backFacingSurface)){
    return base+nearEmission;
  }
  let visibilityGuardEnd=select(intervalEnd,intervalEnd*15.0,backFacingSurface);
  var mergedResult=vec3f(0);
  for(var directionIndexValue=0u;directionIndexValue<32u;directionIndexValue++){
    let direction=directionFromIndex(directionIndexValue,0u);
    let cosine=max(0.0,dot(normal,direction));
    if(cosine<=0.0){continue;}
    // Probe interpolation may borrow a cone from the other side of geometry.
    // Validate the first local/c0 neighborhood in the same direction before
    // accepting the continuation. This is a per-direction replacement, not
    // a whole-pixel enclosure switch, so it remains continuous at box gaps.
    let hit=traceScene(origin,direction,visibilityGuardEnd+0.001);
    if(hit.t<visibilityGuardEnd){
      mergedResult+=max(
        vec3f(0),directAtHit(origin+direction*hit.t,hit)-hit.emissive
      )*cosine;
    }else{
      let continuation=samplePrimaryConeDirection(
        world,normal,directionIndexValue
      );
      mergedResult+=select(base,continuation.xyz,continuation.w>0.5)*cosine;
    }
  }
  return mergedResult*(4.0/32.0)+nearEmission;
}

// Deterministic one-bounce ground-truth gate. Each low-resolution validation
// pixel traces a cosine-weighted reference estimator against the same scene BVH
// and compares it with the reconstructed Split RC irradiance. The output is
// fixed-point so all workgroups can accumulate without floating atomics.
@compute @workgroup_size(8,8,1) fn validateReference(@builtin(global_invocation_id) gid:vec3u){
  let outputSize=vec2u(passParams.pad0,passParams.pad1);
  let sampleCount=max(1u,passParams.value);
  if(any(gid.xy>=outputSize)||gid.z>=sampleCount){return;}
  let fullSize=vec2u(u32(frame.resolution.x),u32(frame.resolution.y));
  let pixel=min(fullSize-vec2u(1),(gid.xy*fullSize+outputSize/2u)/outputSize);
  let world=textureLoad(worldTex,vec2i(pixel),0);
  if(world.w<0.5){return;}
  // Score indirect transport on receiving surfaces. Self-emissive pixels are
  // still present in every BVH ray as light sources, but their displayed
  // radiance is dominated by the separately known emission term and is not a
  // meaningful irradiance-reconstruction error sample.
  let packedNormal=textureLoad(normalTex,vec2i(pixel),0);
  if(world.w>1.25||max(packedNormal.z,packedNormal.w)>1e-5){return;}
  let normal=gbufferNormal(vec2i(pixel));
  let base=(gid.y*outputSize.x+gid.x)*8u;
  if(gid.z==0u){
    let current=clamp(
      resolvedPrimaryIrradiance(
        vec2i(pixel),world.xyz,normal,fract(world.w)>0.25
      ),
      vec3f(0),vec3f(16)
    );
    atomicStore(&accum[base+3u],u32(current.r*65536.0+0.5));
    atomicStore(&accum[base+4u],u32(current.g*65536.0+0.5));
    atomicStore(&accum[base+5u],u32(current.b*65536.0+0.5));
    atomicStore(&accum[base+6u],1u);
  }

  let pixelHash=hash32(gid.x+gid.y*outputSize.x);
  let sequence=gid.z+(pixelHash&65535u);
  let g=1.324717957244746;
  let uv=fract(vec2f(0.5)+f32(sequence+1u)*vec2f(1.0/g,1.0/(g*g)));
  let radius=sqrt(uv.x);
  let phi=TAU*uv.y;
  let local=vec3f(radius*cos(phi),radius*sin(phi),sqrt(max(0.0,1.0-uv.x)));
  let helper=select(vec3f(0,1,0),vec3f(1,0,0),abs(normal.y)>0.9);
  let tangent=normalize(cross(helper,normal));
  let bitangent=cross(normal,tangent);
  let direction=normalize(tangent*local.x+bitangent*local.y+normal*local.z);
  let lod=levelOfDetail(world.xyz);
  let maxDistance=frame.envBaseSpacing.w*exp2(f32(lod))*1.6*64.0;
  let origin=world.xyz+normal*max(0.008,frame.envBaseSpacing.w*0.012);
  let hit=traceScene(origin,direction,maxDistance+0.001);
  var radiance=frame.envBaseSpacing.xyz;
  if(hit.t<maxDistance){
    radiance=directAtHit(origin+direction*hit.t,hit);
  }
  let safe=clamp(radiance,vec3f(0),vec3f(16));
  atomicAdd(&accum[base],u32(safe.r*65536.0+0.5));
  atomicAdd(&accum[base+1u],u32(safe.g*65536.0+0.5));
  atomicAdd(&accum[base+2u],u32(safe.b*65536.0+0.5));
  atomicAdd(&accum[base+7u],1u);
}

fn pointShadowArrayCoordinate(fromLight:vec3f)->vec3f{
  let magnitude=abs(fromLight);
  if(magnitude.x>=magnitude.y&&magnitude.x>=magnitude.z){
    let face=select(1.0,0.0,fromLight.x>=0.0);
    let ndc=vec2f(
      select(-fromLight.z,fromLight.z,fromLight.x>=0.0),
      fromLight.y
    )/magnitude.x;
    return vec3f(ndc.x*0.5+0.5,0.5-ndc.y*0.5,face);
  }
  if(magnitude.y>=magnitude.z){
    let face=select(3.0,2.0,fromLight.y>=0.0);
    let ndc=vec2f(
      -fromLight.x,
      select(fromLight.z,-fromLight.z,fromLight.y>=0.0)
    )/magnitude.y;
    return vec3f(ndc.x*0.5+0.5,0.5-ndc.y*0.5,face);
  }
  let face=select(5.0,4.0,fromLight.z>=0.0);
  let ndc=vec2f(
    select(fromLight.x,-fromLight.x,fromLight.z>=0.0),
    fromLight.y
  )/magnitude.z;
  return vec3f(ndc.x*0.5+0.5,0.5-ndc.y*0.5,face);
}

fn pointShadowAuditVisibility(world:vec3f,normal:vec3f)->f32{
  let fromLight=world+normal*max(0.006,frame.envBaseSpacing.w*0.01)-frame.pointPosRange.xyz;
  let major=max(abs(fromLight.x),max(abs(fromLight.y),abs(fromLight.z)));
  let nearPlane=max(0.01,frame.pointPosRange.w*0.001);
  let farPlane=frame.pointPosRange.w;
  if(major<=nearPlane||major>=farPlane){return 1.0;}
  let reference=farPlane/(farPlane-nearPlane)
    -(farPlane*nearPlane)/((farPlane-nearPlane)*major);
  let coordinate=pointShadowArrayCoordinate(fromLight);
  return textureSampleCompareLevel(
    pointShadowAuditTex,pointShadowAuditSampler,coordinate.xy,i32(coordinate.z),reference
  );
}

fn sunSplitDepth(cascade:u32)->f32{
  if(cascade==0u){return sunShadow.splitDepths.x;}
  if(cascade==1u){return sunShadow.splitDepths.y;}
  if(cascade==2u){return sunShadow.splitDepths.z;}
  return sunShadow.splitDepths.w;
}

fn sunCascadeIndex(world:vec3f)->u32{
  let depth=max(0.0,dot(world-frame.cameraPos.xyz,sunShadow.cameraForward.xyz));
  if(depth<=sunShadow.splitDepths.x){return 0u;}
  if(depth<=sunShadow.splitDepths.y){return 1u;}
  if(depth<=sunShadow.splitDepths.z){return 2u;}
  return 3u;
}

fn sampleSunShadowAuditCascade(world:vec3f,normal:vec3f,cascade:u32)->f32{
  let texel=sunShadow.texelSizes[cascade];
  let lightDirection=normalize(-frame.sunDirTime.xyz);
  let normalOffset=texel*(0.35+0.75*(1.0-max(0.0,dot(normal,lightDirection))));
  let clip=sunShadow.matrices[cascade]*vec4f(world+normal*normalOffset,1.0);
  let ndc=clip.xyz/clip.w;
  let uv=vec2f(ndc.x*0.5+0.5,0.5-ndc.y*0.5);
  if(any(uv<vec2f(0))||any(uv>vec2f(1))||ndc.z<0.0||ndc.z>1.0){return 1.0;}
  let size=vec2f(textureDimensions(sunShadowAuditTex));
  var result=0.0;
  var total=0.0;
  for(var y=-2;y<=2;y++){
    for(var x=-2;x<=2;x++){
      let weight=(3.0-abs(f32(x)))*(3.0-abs(f32(y)));
      result+=weight*textureSampleCompareLevel(
        sunShadowAuditTex,sunShadowAuditSampler,
        uv+vec2f(f32(x),f32(y))/size,i32(cascade),ndc.z-0.00008
      );
      total+=weight;
    }
  }
  return result/total;
}

fn sunShadowAuditVisibility(world:vec3f,normal:vec3f)->f32{
  let cascade=sunCascadeIndex(world);
  var result=sampleSunShadowAuditCascade(world,normal,cascade);
  if(cascade<3u){
    let depth=max(0.0,dot(world-frame.cameraPos.xyz,sunShadow.cameraForward.xyz));
    let previous=select(sunSplitDepth(cascade-1u),0.0,cascade==0u);
    let blendStart=mix(previous,sunSplitDepth(cascade),0.85);
    let blend=clamp((depth-blendStart)/max(0.001,sunSplitDepth(cascade)-blendStart),0.0,1.0);
    if(blend>0.0){result=mix(result,sampleSunShadowAuditCascade(world,normal,cascade+1u),blend);}
  }
  return result;
}

// Raster-shadow correctness oracle. The depth maps are compared directly with
// exact software-BVH visibility at the same visible surface samples. This is
// intentionally separate from the GI/path-reference audit: a stable but wrong
// cube-map face convention or projection must fail independently.
@compute @workgroup_size(8,8,1) fn validateShadowMaps(@builtin(global_invocation_id) gid:vec3u){
  let outputSize=vec2u(passParams.pad0,passParams.pad1);
  if(any(gid.xy>=outputSize)){return;}
  let fullSize=vec2u(u32(frame.resolution.x),u32(frame.resolution.y));
  let pixel=min(fullSize-vec2u(1),(gid.xy*fullSize+outputSize/2u)/outputSize);
  let world=textureLoad(worldTex,vec2i(pixel),0);
  if(world.w<0.5){return;}
  let normal=gbufferNormal(vec2i(pixel));
  let base=(gid.y*outputSize.x+gid.x)*6u;

  let toPoint=frame.pointPosRange.xyz-world.xyz;
  let pointDistance=length(toPoint);
  let pointDirection=toPoint/max(pointDistance,1e-5);
  let pointValid=frame.pointColorIntensity.w>0.0
    && pointDistance<frame.pointPosRange.w
    && dot(normal,pointDirection)>0.02;
  if(pointValid){
    let pointHit=traceScene(
      world.xyz+normal*max(0.006,frame.envBaseSpacing.w*0.01),
      pointDirection,max(0.0,pointDistance-0.03)
    );
    let reference=select(0.0,1.0,pointHit.t>=pointDistance-0.04);
    atomicStore(&accum[base],u32(reference*65535.0+0.5));
    atomicStore(&accum[base+1u],u32(clamp(pointShadowAuditVisibility(world.xyz,normal),0.0,1.0)*65535.0+0.5));
    let face=pointShadowArrayCoordinate(world.xyz-frame.pointPosRange.xyz).z;
    atomicStore(&accum[base+4u],u32(face)+1u);
  }

  let sunDirection=normalize(-frame.sunDirTime.xyz);
  let sunValid=dot(normal,sunDirection)>0.02;
  if(sunValid){
    let sunHit=traceScene(world.xyz+normal*0.012,sunDirection,10000.0);
    let reference=select(0.0,1.0,sunHit.t>=9999.0);
    atomicStore(&accum[base+2u],u32(reference*65535.0+0.5));
    atomicStore(&accum[base+3u],u32(clamp(sunShadowAuditVisibility(world.xyz,normal),0.0,1.0)*65535.0+0.5));
    atomicStore(&accum[base+5u],1u);
  }
}
`;

export const finalShader = /* wgsl */`
struct FrameUniforms {
  viewProj: mat4x4<f32>,
  sunViewProj: mat4x4<f32>,
  cameraPos: vec4f,
  sunDirTime: vec4f,
  sunColorIntensity: vec4f,
  pointPosRange: vec4f,
  pointColorIntensity: vec4f,
  envBaseSpacing: vec4f,
  resolution: vec4f,
  controls: vec4f,
};
struct SunShadowUniforms {
  matrices: array<mat4x4<f32>,4>,
  splitDepths: vec4f,
  texelSizes: vec4f,
  cameraForward: vec4f,
  parameters: vec4f,
};
struct HashSlot { key: atomic<u32>, index: atomic<u32> };
struct BvhNode { minMeta:vec4f, maxMeta:vec4f };
struct Triangle {
  a:vec4f, b:vec4f, c:vec4f, albedo:vec4f, emissive:vec4f,
  uvAB:vec4f, uvCMaterial:vec4f, normalOct:vec4u,
};
struct ShortHit { t:f32, normal:vec3f, albedo:vec3f, emissive:vec3f };
@group(0) @binding(0) var<uniform> frame: FrameUniforms;
@group(0) @binding(1) var albedoTex: texture_2d<f32>;
@group(0) @binding(2) var normalTex: texture_2d<f32>;
@group(0) @binding(3) var worldTex: texture_2d<f32>;
@group(0) @binding(4) var shadowTex: texture_depth_2d_array;
@group(0) @binding(5) var shadowSampler: sampler_comparison;
@group(0) @binding(6) var<storage,read_write> slots: array<HashSlot>;
@group(0) @binding(7) var irradianceAtlas: texture_2d<f32>;
@group(0) @binding(8) var<storage,read> cones: array<vec4f>;
@group(0) @binding(9) var<storage,read_write> accum: array<atomic<u32>>;
@group(0) @binding(10) var irradianceSampler: sampler;
@group(0) @binding(11) var emissiveTex: texture_2d<f32>;
@group(0) @binding(12) var pointShadowTex: texture_depth_2d_array;
@group(0) @binding(13) var pointShadowSampler: sampler_comparison;
@group(0) @binding(14) var<uniform> sunShadow: SunShadowUniforms;
@group(0) @binding(15) var<storage,read> shortBvhNodes:array<BvhNode>;
@group(0) @binding(16) var<storage,read> shortTriangles:array<Triangle>;
@group(0) @binding(17) var shortAlbedoAtlas:texture_2d_array<f32>;
@group(0) @binding(18) var shortAlbedoSampler:sampler;
@group(0) @binding(19) var<storage,read> emissiveBvhNodes:array<BvhNode>;
@group(0) @binding(20) var<storage,read> emissiveTriangles:array<Triangle>;

fn gbufferNormal(pixel:vec2i)->vec3f{
  let oct=textureLoad(normalTex,pixel,0).xy;
  var normal=vec3f(oct,1.0-abs(oct.x)-abs(oct.y));
  if(normal.z<0.0){
    let old=normal.xy;
    normal.x=(1.0-abs(old.y))*select(-1.0,1.0,old.x>=0.0);
    normal.y=(1.0-abs(old.x))*select(-1.0,1.0,old.y>=0.0);
  }
  return normalize(normal);
}

const EMPTY:u32=0xffffffffu;
const HASH_FRAME_STRIDE:u32=44032u;
const IRRADIANCE_FRAME_STRIDE:u32=1048576u;
const ACCUM_FRAME_STRIDE:u32=13107200u;
const FIXED_SCALE:f32=4096.0;
const HASH_OFFSETS=array<u32,4>(0u,32768u,40960u,43008u);
const HASH_SIZES=array<u32,4>(32768u,8192u,2048u,1024u);
const DATA_OFFSETS=array<u32,4>(0u,524288u,1048576u,1572864u);
const DIR_COUNTS=array<u32,4>(32u,128u,512u,2048u);
fn hash32(value:u32)->u32{
  var x=value;x=x^(x>>16u);x=x*0x7feb352du;x=x^(x>>15u);x=x*0x846ca68bu;return x^(x>>16u);
}
fn surfaceClass(normalIn:vec3f)->u32{
  let normal=normalize(normalIn);
  let a=abs(normal);
  var axis=2u;
  var component=normal.z;
  if(a.x>=a.y&&a.x>=a.z){axis=0u;component=normal.x;}
  else if(a.y>=a.z){axis=1u;component=normal.y;}
  return axis*2u+select(0u,1u,component>=0.0);
}
fn keyFromCellSurface(cellIn:vec3i,lod:u32,surfaceClassValue:u32)->u32{
  if(any(cellIn<vec3i(-256))||any(cellIn>vec3i(255))){return EMPTY;}
  let c=cellIn+vec3i(256);
  return u32(c.x)|(u32(c.y)<<9u)|(u32(c.z)<<18u)|((lod&3u)<<27u)|((surfaceClassValue&7u)<<29u);
}
fn lookupProbeCascade(cascade:u32,key:u32)->u32{
  let mask=HASH_SIZES[cascade]-1u;
  let start=hash32(key)&mask;
  let frameIndex=min(1u,u32(floor(frame.controls.w)));
  let base=frameIndex*HASH_FRAME_STRIDE+HASH_OFFSETS[cascade];
  for(var step=0u;step<32u;step++){
    let slot=base+((start+step)&mask);
    let found=atomicLoad(&slots[slot].key);
    if(found==key){return atomicLoad(&slots[slot].index);}
    if(found==EMPTY){return EMPTY;}
  }
  return EMPTY;
}
fn lookupProbe(key:u32)->u32{return lookupProbeCascade(0u,key);}
fn lodDistance(position:vec3f)->f32{
  let delta=abs(position-frame.cameraPos.xyz);
  return max(delta.x,max(delta.y,delta.z));
}
fn levelOfDetail(position:vec3f)->u32{
  let ratio=max(1.0,lodDistance(position)/max(0.001,frame.envBaseSpacing.w*18.0));
  return u32(clamp(floor(log2(ratio)),0.0,3.0));
}
fn lodSelection(position:vec3f)->vec3f{
  let fine=levelOfDetail(position);
  if(fine>=3u){return vec3f(f32(fine),f32(fine),0.0);}
  let baseRange=max(0.001,frame.envBaseSpacing.w*18.0);
  let boundary=baseRange*exp2(f32(fine+1u));
  let start=boundary*0.9;
  let blend=clamp((lodDistance(position)-start)/max(0.001,boundary-start),0.0,1.0);
  let coarse=select(fine,fine+1u,blend>0.0);
  return vec3f(f32(fine),f32(coarse),blend);
}
fn octIndex(normalIn:vec3f)->u32{
  var n=normalize(normalIn);
  n/=abs(n.x)+abs(n.y)+abs(n.z);
  var f=n.xy;
  if(n.z<0.0){
    f=vec2f((1.0-abs(f.y))*select(-1.0,1.0,f.x>=0.0),(1.0-abs(f.x))*select(-1.0,1.0,f.y>=0.0));
  }
  let uv=clamp(f*0.5+0.5,vec2f(0),vec2f(0.99999));
  let xy=vec2u(floor(uv*6.0));
  return xy.x+xy.y*6u;
}
fn sampleIrradianceLod(position:vec3f,normal:vec3f,lod:u32)->vec4f{
  let spacing=frame.envBaseSpacing.w*exp2(f32(lod));
  let grid=position/spacing-vec3f(0.5);
  let cell=vec3i(floor(grid));
  let fraction=fract(grid);
  let fixedBits=vec3i(floor(position/spacing))-cell;
  let absoluteNormal=abs(normal);
  var normalAxis=2u;
  if(absoluteNormal.x>=absoluteNormal.y&&absoluteNormal.x>=absoluteNormal.z){
    normalAxis=0u;
  }else if(absoluteNormal.y>=absoluteNormal.z){
    normalAxis=1u;
  }
  var normalWeight=0.0;
  if(normalAxis==0u){normalWeight=select(1.0-fraction.x,fraction.x,fixedBits.x==1);}
  if(normalAxis==1u){normalWeight=select(1.0-fraction.y,fraction.y,fixedBits.y==1);}
  if(normalAxis==2u){normalWeight=select(1.0-fraction.z,fraction.z,fixedBits.z==1);}
  var octNormal=normalize(normal);
  octNormal/=max(1e-6,abs(octNormal.x)+abs(octNormal.y)+abs(octNormal.z));
  var oct=octNormal.xy;
  if(octNormal.z<0.0){
    oct=vec2f(
      (1.0-abs(oct.y))*select(-1.0,1.0,oct.x>=0.0),
      (1.0-abs(oct.x))*select(-1.0,1.0,oct.y>=0.0)
    );
  }
  let octCoordinate=clamp((oct*0.5+0.5)*6.0+vec2f(0.5),vec2f(0),vec2f(7));
  let frameIndex=min(1u,u32(floor(frame.controls.w)));
  var value=vec3f(0);
  var total=0.0;
  for(var corner=0u;corner<4u;corner++){
    var bits=fixedBits;
    if(normalAxis==0u){
      bits.y=i32(corner&1u);bits.z=i32((corner>>1u)&1u);
    }else if(normalAxis==1u){
      bits.x=i32(corner&1u);bits.z=i32((corner>>1u)&1u);
    }else{
      bits.x=i32(corner&1u);bits.y=i32((corner>>1u)&1u);
    }
    let wv=vec3f(
      select(1.0-fraction.x,fraction.x,bits.x==1),
      select(1.0-fraction.y,fraction.y,bits.y==1),
      select(1.0-fraction.z,fraction.z,bits.z==1)
    );
    let spatialWeight=wv.x*wv.y*wv.z;
    let probe=lookupProbe(keyFromCellSurface(cell+bits,lod,surfaceClass(normal)));
    if(probe!=EMPTY&&probe<16384u){
      let tile=vec2f(f32(probe%64u)*8.0,f32(probe/64u)*8.0+f32(frameIndex)*2048.0);
      let atlasUv=(tile+octCoordinate+vec2f(0.5))/vec2f(512.0,4096.0);
      let irradiance=textureSampleLevel(irradianceAtlas,irradianceSampler,atlasUv,0.0);
      let activeWeight=spatialWeight*irradiance.a;
      value+=irradiance.xyz*activeWeight;
      total+=activeWeight;
    }
  }
  if(total<1e-5){return vec4f(0,0,0,0);}
  return vec4f(value/total,clamp(total/max(normalWeight,1e-5),0.0,1.0));
}
fn sampleIrradiance(position:vec3f,normal:vec3f)->vec4f{
  let lods=lodSelection(position);
  let fine=sampleIrradianceLod(position,normal,u32(lods.x));
  if(u32(lods.y)==u32(lods.x)){return fine;}
  let coarse=sampleIrradianceLod(position,normal,u32(lods.y));
  if(fine.w<0.001){return coarse;}
  if(coarse.w<0.001){return fine;}
  return vec4f(mix(fine.xyz,coarse.xyz,lods.z),mix(fine.w,coarse.w,lods.z));
}
fn sampleConeDirectionLod(
  position:vec3f,normal:vec3f,lod:u32,direction:u32
)->vec4f{
  let spacing=frame.envBaseSpacing.w*exp2(f32(lod));
  let grid=position/spacing-vec3f(0.5);
  let cell=vec3i(floor(grid));
  let fraction=fract(grid);
  let fixedBits=vec3i(floor(position/spacing))-cell;
  let absoluteNormal=abs(normal);
  var normalAxis=2u;
  if(absoluteNormal.x>=absoluteNormal.y&&absoluteNormal.x>=absoluteNormal.z){
    normalAxis=0u;
  }else if(absoluteNormal.y>=absoluteNormal.z){
    normalAxis=1u;
  }
  var value=vec3f(0);
  var total=0.0;
  for(var corner=0u;corner<4u;corner++){
    var bits=fixedBits;
    if(normalAxis==0u){
      bits.y=i32(corner&1u);bits.z=i32((corner>>1u)&1u);
    }else if(normalAxis==1u){
      bits.x=i32(corner&1u);bits.z=i32((corner>>1u)&1u);
    }else{
      bits.x=i32(corner&1u);bits.y=i32((corner>>1u)&1u);
    }
    let wv=vec3f(
      select(1.0-fraction.x,fraction.x,bits.x==1),
      select(1.0-fraction.y,fraction.y,bits.y==1),
      select(1.0-fraction.z,fraction.z,bits.z==1)
    );
    let weight=wv.x*wv.y*wv.z;
    let probe=lookupProbe(keyFromCellSurface(
      cell+bits,lod,surfaceClass(normal)
    ));
    if(probe!=EMPTY&&probe<16384u){
      let cone=cones[dataIndex(0u,probe,direction)];
      if(cone.w>0.5){value+=cone.xyz*weight;total+=weight;}
    }
  }
  if(total<1e-5){return vec4f(0);}
  return vec4f(value/total,1.0);
}
fn sampleConeDirection(
  position:vec3f,normal:vec3f,direction:u32
)->vec4f{
  let lods=lodSelection(position);
  let fine=sampleConeDirectionLod(position,normal,u32(lods.x),direction);
  if(u32(lods.y)==u32(lods.x)){return fine;}
  let coarse=sampleConeDirectionLod(position,normal,u32(lods.y),direction);
  if(fine.w<0.5){return coarse;}
  if(coarse.w<0.5){return fine;}
  return vec4f(mix(fine.xyz,coarse.xyz,lods.z),1.0);
}
fn mortonDirectionIndexFinal(u:u32,v:u32,cascade:u32)->u32{
  let bits=2u+cascade;
  var result=0u;
  for(var bit=0u;bit<bits;bit++){
    result|=((u>>bit)&1u)<<(bit*2u);
    result|=((v>>bit)&1u)<<(bit*2u+1u);
  }
  return result|(((u>>bits)&1u)<<(bits*2u));
}
fn mortonDirectionCoordinatesFinal(index:u32,cascade:u32)->vec2u{
  let bits=2u+cascade;
  var u=0u;
  var v=0u;
  for(var bit=0u;bit<bits;bit++){
    u|=((index>>(bit*2u))&1u)<<bit;
    v|=((index>>(bit*2u+1u))&1u)<<bit;
  }
  u|=((index>>(bits*2u))&1u)<<bits;
  return vec2u(u,v);
}
fn directionIndex(direction:vec3f,cascade:u32)->u32{
  let theta=4u<<cascade;
  let width=theta*2u;
  let uFloat=fract(atan2(direction.y,direction.x)/6.283185307179586+1.0);
  let vFloat=clamp(direction.z*0.5+0.5,0.0,0.999999);
  let u=min(width-1u,u32(floor(uFloat*f32(width))));
  let v=min(theta-1u,u32(floor(vFloat*f32(theta))));
  return mortonDirectionIndexFinal(u,v,cascade);
}
fn decodeEqualArea(uv:vec2f)->vec3f{
  let phi=uv.x*6.283185307179586;
  let z=uv.y*2.0-1.0;
  let radius=sqrt(max(0.0,1.0-z*z));
  return vec3f(radius*cos(phi),radius*sin(phi),z);
}
fn directionFromIndex(index:u32,cascade:u32)->vec3f{
  let theta=4u<<cascade;
  let width=theta*2u;
  let coordinate=mortonDirectionCoordinatesFinal(index,cascade);
  return decodeEqualArea(vec2f(
    (f32(coordinate.x)+0.5)/f32(width),
    (f32(coordinate.y)+0.5)/f32(theta)
  ));
}
fn dataIndex(cascade:u32,probe:u32,direction:u32)->u32{
  return DATA_OFFSETS[cascade]+probe*DIR_COUNTS[cascade]+direction;
}
fn accumIndex(cascade:u32,probe:u32,direction:u32)->u32{
  let frameIndex=min(1u,u32(floor(frame.controls.w)));
  return frameIndex*ACCUM_FRAME_STRIDE+dataIndex(cascade,probe,direction)*5u;
}
fn sunSplitDepth(cascade:u32)->f32{
  if(cascade==0u){return sunShadow.splitDepths.x;}
  if(cascade==1u){return sunShadow.splitDepths.y;}
  if(cascade==2u){return sunShadow.splitDepths.z;}
  return sunShadow.splitDepths.w;
}
fn sunCascadeIndex(world:vec3f)->u32{
  let depth=max(0.0,dot(world-frame.cameraPos.xyz,sunShadow.cameraForward.xyz));
  if(depth<=sunShadow.splitDepths.x){return 0u;}
  if(depth<=sunShadow.splitDepths.y){return 1u;}
  if(depth<=sunShadow.splitDepths.z){return 2u;}
  return 3u;
}
fn sampleSunShadowCascade(world:vec3f,normal:vec3f,cascade:u32)->f32{
  let texel=sunShadow.texelSizes[cascade];
  let lightDirection=normalize(-frame.sunDirTime.xyz);
  let normalOffset=texel*(0.35+0.75*(1.0-max(0.0,dot(normal,lightDirection))));
  let clip=sunShadow.matrices[cascade]*vec4f(world+normal*normalOffset,1.0);
  let ndc=clip.xyz/clip.w;
  let uv=vec2f(ndc.x*0.5+0.5,0.5-ndc.y*0.5);
  if(any(uv<vec2f(0))||any(uv>vec2f(1))||ndc.z<0.0||ndc.z>1.0){return 1.0;}
  let size=vec2f(textureDimensions(shadowTex));
  var result=0.0;
  var total=0.0;
  for(var y=-2;y<=2;y++){
    for(var x=-2;x<=2;x++){
      let weight=(3.0-abs(f32(x)))*(3.0-abs(f32(y)));
      result+=weight*textureSampleCompareLevel(
        shadowTex,shadowSampler,uv+vec2f(f32(x),f32(y))/size,
        i32(cascade),ndc.z-0.00008
      );
      total+=weight;
    }
  }
  return result/total;
}
fn shadowVisibility(world:vec3f,normal:vec3f)->f32{
  let cascade=sunCascadeIndex(world);
  var result=sampleSunShadowCascade(world,normal,cascade);
  if(cascade<3u){
    let depth=max(0.0,dot(world-frame.cameraPos.xyz,sunShadow.cameraForward.xyz));
    let previous=select(sunSplitDepth(cascade-1u),0.0,cascade==0u);
    let split=sunSplitDepth(cascade);
    let blend=clamp((depth-mix(previous,split,0.85))/max(0.001,split-mix(previous,split,0.85)),0.0,1.0);
    if(blend>0.0){result=mix(result,sampleSunShadowCascade(world,normal,cascade+1u),blend);}
  }
  return result;
}
fn surfaceEmission(pixel:vec2i)->vec3f{
  let packed=vec3f(
    textureLoad(albedoTex,pixel,0).a,
    textureLoad(emissiveTex,pixel,0).zw
  );
  return exp2(packed*8.0)-vec3f(1.0);
}
fn pointShadowArrayCoordinate(fromLight:vec3f)->vec3f{
  let magnitude=abs(fromLight);
  if(magnitude.x>=magnitude.y&&magnitude.x>=magnitude.z){
    let face=select(1.0,0.0,fromLight.x>=0.0);
    let ndc=vec2f(
      select(-fromLight.z,fromLight.z,fromLight.x>=0.0),
      fromLight.y
    )/magnitude.x;
    return vec3f(ndc.x*0.5+0.5,0.5-ndc.y*0.5,face);
  }
  if(magnitude.y>=magnitude.z){
    let face=select(3.0,2.0,fromLight.y>=0.0);
    let ndc=vec2f(
      -fromLight.x,
      select(fromLight.z,-fromLight.z,fromLight.y>=0.0)
    )/magnitude.y;
    return vec3f(ndc.x*0.5+0.5,0.5-ndc.y*0.5,face);
  }
  let face=select(5.0,4.0,fromLight.z>=0.0);
  let ndc=vec2f(
    select(fromLight.x,-fromLight.x,fromLight.z>=0.0),
    fromLight.y
  )/magnitude.z;
  return vec3f(ndc.x*0.5+0.5,0.5-ndc.y*0.5,face);
}
fn pointShadowVisibility(world:vec3f,normal:vec3f)->f32{
  if(frame.pointColorIntensity.w<=0.0){return 1.0;}
  let fromLight=world+normal*max(0.006,frame.envBaseSpacing.w*0.01)-frame.pointPosRange.xyz;
  let major=max(abs(fromLight.x),max(abs(fromLight.y),abs(fromLight.z)));
  let nearPlane=max(0.01,frame.pointPosRange.w*0.001);
  let farPlane=frame.pointPosRange.w;
  if(major<=nearPlane||major>=farPlane){return 1.0;}
  let reference=farPlane/(farPlane-nearPlane)
    -(farPlane*nearPlane)/((farPlane-nearPlane)*major);
  let coordinate=pointShadowArrayCoordinate(fromLight);
  return textureSampleCompareLevel(
    pointShadowTex,pointShadowSampler,coordinate.xy,i32(coordinate.z),reference
  );
}
fn aces(x:vec3f)->vec3f{
  let a=2.51;let b=0.03;let c=2.43;let d=0.59;let e=0.14;
  return clamp((x*(a*x+b))/(x*(c*x+d)+e),vec3f(0),vec3f(1));
}
fn displayEncode(x:vec3f)->vec3f{
  // The canvas swap chain is an unorm target, not an sRGB render target.
  // Apply the display transfer after the filmic curve so shadow detail is not
  // accidentally presented as linear bytes and crushed on an sRGB monitor.
  return pow(aces(x),vec3f(1.0/2.2));
}

// Section 7.1's C(-1) merge restores detail below the c0 probe spacing. The
// paper proposes an ambient-term optimization after showing a screen-space
// prototype. We use that ambient form, but resolve its short interval against
// the exact scene BVH. It is therefore world-space, off-screen complete, and
// invariant under camera motion.
fn shortRayBoxNear(
  origin:vec3f,inverseDirection:vec3f,minB:vec3f,maxB:vec3f,maxDistance:f32
)->f32{
  let t0=(minB-origin)*inverseDirection;
  let t1=(maxB-origin)*inverseDirection;
  let near3=min(t0,t1);
  let far3=max(t0,t1);
  let nearT=max(max(near3.x,near3.y),max(near3.z,0.0));
  let farT=min(min(far3.x,far3.y),far3.z);
  return select(maxDistance,nearT,farT>=nearT&&nearT<maxDistance);
}
fn shortAtlas(uv:vec2f,material:f32)->vec4f{
  if(material<0.0){return vec4f(1);}
  let index=u32(material+0.5);
  let sampled=textureSampleLevel(
    shortAlbedoAtlas,shortAlbedoSampler,fract(uv),i32(index),2.0
  );
  if((u32(frame.cameraPos.w+0.5)&16u)==0u){return sampled;}
  let luminance=dot(sampled.rgb,vec3f(0.2126,0.7152,0.0722));
  let detail=clamp(0.72+luminance*0.34,0.62,1.04);
  let stone=vec3f(0.93,0.96,0.96)*detail;
  let cyan=vec3f(0.18,0.82,0.86)*mix(0.82,1.08,luminance);
  return vec4f(select(stone,cyan,index>=15u&&index<=19u),sampled.a);
}
fn shortNormal(packed:u32)->vec3f{
  let encoded=vec2f(f32(packed&65535u),f32(packed>>16u))/65535.0*2.0-1.0;
  var normal=vec3f(encoded,1.0-abs(encoded.x)-abs(encoded.y));
  if(normal.z<0.0){
    let old=normal.xy;
    normal.x=(1.0-abs(old.y))*select(-1.0,1.0,old.x>=0.0);
    normal.y=(1.0-abs(old.x))*select(-1.0,1.0,old.y>=0.0);
  }
  return normalize(normal);
}
fn shortTriangle(
  origin:vec3f,direction:vec3f,triangle:Triangle,maxDistance:f32
)->vec3f{
  let edge1=triangle.b.xyz-triangle.a.xyz;
  let edge2=triangle.c.xyz-triangle.a.xyz;
  let p=cross(direction,edge2);
  let determinant=dot(edge1,p);
  if(abs(determinant)<1e-7){return vec3f(maxDistance,0,0);}
  let inverse=1.0/determinant;
  let offset=origin-triangle.a.xyz;
  let u=dot(offset,p)*inverse;
  let edgeTolerance=2e-6;
  if(u < -edgeTolerance||u>1.0+edgeTolerance){
    return vec3f(maxDistance,0,0);
  }
  let q=cross(offset,edge1);
  let v=dot(direction,q)*inverse;
  if(v < -edgeTolerance||u+v>1.0+edgeTolerance){
    return vec3f(maxDistance,0,0);
  }
  let distance=dot(edge2,q)*inverse;
  // The final-stage origin is already displaced from its source face. Keep
  // this below a raster pixel's possible distance to an adjacent face: the
  // old millimetre-scale cutoff opened a one-pixel crack at shared box edges.
  // This is world/scene independent and only affects exact C(-1) visibility.
  // Keep it below one f32 ulp at the Cornell box corner: 1e-5 discarded the
  // adjacent face at a shared edge and left one bright pixel inside a sealed
  // volume even though the watertight edge functions accepted the triangle.
  if(distance>1e-7&&distance<maxDistance){
    return vec3f(distance,clamp(u,0.0,1.0),clamp(v,0.0,1.0));
  }
  return vec3f(maxDistance,0,0);
}
fn shortTriangleWatertight(
  origin:vec3f,direction:vec3f,triangle:Triangle,maxDistance:f32
)->vec3f{
  let absoluteDirection=abs(direction);
  var kz=0u;
  if(absoluteDirection.y>absoluteDirection.x){kz=1u;}
  if(absoluteDirection.z>absoluteDirection[kz]){kz=2u;}
  var kx=(kz+1u)%3u;
  var ky=(kx+1u)%3u;
  if(direction[kz]<0.0){let swap=kx;kx=ky;ky=swap;}
  let shearX=direction[kx]/direction[kz];
  let shearY=direction[ky]/direction[kz];
  let shearZ=1.0/direction[kz];
  let a=triangle.a.xyz-origin;
  let b=triangle.b.xyz-origin;
  let c=triangle.c.xyz-origin;
  let ax=a[kx]-shearX*a[kz]; let ay=a[ky]-shearY*a[kz];
  let bx=b[kx]-shearX*b[kz]; let by=b[ky]-shearY*b[kz];
  let cx=c[kx]-shearX*c[kz]; let cy=c[ky]-shearY*c[kz];
  let edgeA=bx*cy-by*cx;
  let edgeB=cx*ay-cy*ax;
  let edgeC=ax*by-ay*bx;
  let hasNegative=edgeA<0.0||edgeB<0.0||edgeC<0.0;
  let hasPositive=edgeA>0.0||edgeB>0.0||edgeC>0.0;
  if(hasNegative&&hasPositive){return vec3f(maxDistance,0,0);}
  let determinant=edgeA+edgeB+edgeC;
  if(determinant==0.0){return vec3f(maxDistance,0,0);}
  let distanceScaled=edgeA*(a[kz]*shearZ)
    +edgeB*(b[kz]*shearZ)+edgeC*(c[kz]*shearZ);
  if(determinant>0.0){
    if(distanceScaled<=1e-7*determinant||distanceScaled>=maxDistance*determinant){
      return vec3f(maxDistance,0,0);
    }
  }else if(distanceScaled>=1e-7*determinant||distanceScaled<=maxDistance*determinant){
    return vec3f(maxDistance,0,0);
  }
  let inverseDeterminant=1.0/determinant;
  return vec3f(
    distanceScaled*inverseDeterminant,
    edgeB*inverseDeterminant,
    edgeC*inverseDeterminant
  );
}
fn traceShortRangeImpl(
  origin:vec3f,directionIn:vec3f,maxDistance:f32,watertight:bool
)->ShortHit{
  let direction=normalize(directionIn);
  let inverseDirection=select(vec3f(-1e20),vec3f(1e20),direction>=vec3f(0))
    /max(vec3f(1),abs(direction)*1e20);
  var result=ShortHit(maxDistance,vec3f(0,1,0),vec3f(0),vec3f(0));
  var stack:array<u32,64>;
  var stackSize=1u;
  stack[0]=0u;
  loop{
    if(stackSize==0u){break;}
    stackSize-=1u;
    let nodeIndex=stack[stackSize];
    let node=shortBvhNodes[nodeIndex];
    if(shortRayBoxNear(
      origin,inverseDirection,node.minMeta.xyz,node.maxMeta.xyz,result.t
    )>=result.t){continue;}
    let left=bitcast<u32>(node.minMeta.w);
    let right=bitcast<u32>(node.maxMeta.w);
    if((left&0x80000000u)!=0u){
      let first=left&0x7fffffffu;
      for(var triangleOffset=0u;triangleOffset<right;triangleOffset++){
        let triangle=shortTriangles[first+triangleOffset];
        var intersection=shortTriangle(origin,direction,triangle,result.t);
        if(watertight){
          intersection=shortTriangleWatertight(origin,direction,triangle,result.t);
        }
        if(intersection.x<result.t){
          let uv=triangle.uvAB.xy*(1.0-intersection.y-intersection.z)
            +triangle.uvAB.zw*intersection.y+triangle.uvCMaterial.xy*intersection.z;
          let surface=shortAtlas(uv,triangle.uvCMaterial.z);
          if(triangle.uvCMaterial.w>0.0&&surface.a<triangle.uvCMaterial.w){continue;}
          let barycentric=vec3f(
            1.0-intersection.y-intersection.z,intersection.y,intersection.z
          );
          var hitNormal=normalize(
            shortNormal(triangle.normalOct.x)*barycentric.x
            +shortNormal(triangle.normalOct.y)*barycentric.y
            +shortNormal(triangle.normalOct.z)*barycentric.z
          );
          if(dot(hitNormal,direction)>0.0){hitNormal=-hitNormal;}
          result=ShortHit(
            intersection.x,hitNormal,triangle.albedo.xyz*surface.rgb,
            triangle.emissive.xyz
          );
        }
      }
    }else{
      let leftNear=shortRayBoxNear(
        origin,inverseDirection,shortBvhNodes[left].minMeta.xyz,
        shortBvhNodes[left].maxMeta.xyz,result.t
      );
      let rightNear=shortRayBoxNear(
        origin,inverseDirection,shortBvhNodes[right].minMeta.xyz,
        shortBvhNodes[right].maxMeta.xyz,result.t
      );
      if(leftNear<result.t&&rightNear<result.t){
        if(stackSize>61u){break;}
        if(leftNear<rightNear){stack[stackSize]=right;stack[stackSize+1u]=left;}
        else{stack[stackSize]=left;stack[stackSize+1u]=right;}
        stackSize+=2u;
      }else if(leftNear<result.t){
        if(stackSize>62u){break;}
        stack[stackSize]=left;stackSize+=1u;
      }else if(rightNear<result.t){
        if(stackSize>62u){break;}
        stack[stackSize]=right;stackSize+=1u;
      }
    }
  }
  return result;
}
fn traceShortRange(origin:vec3f,direction:vec3f,maxDistance:f32)->ShortHit{
  return traceShortRangeImpl(origin,direction,maxDistance,false);
}
fn traceShortRangeWatertight(
  origin:vec3f,direction:vec3f,maxDistance:f32
)->ShortHit{
  return traceShortRangeImpl(origin,direction,maxDistance,true);
}

fn fastSunVisibility(world:vec3f,normal:vec3f)->f32{
  let cascade=sunCascadeIndex(world);
  let clip=sunShadow.matrices[cascade]*vec4f(
    world+normal*max(0.006,sunShadow.texelSizes[cascade]*0.35),1.0
  );
  let ndc=clip.xyz/clip.w;
  let uv=vec2f(ndc.x*0.5+0.5,0.5-ndc.y*0.5);
  if(any(uv<vec2f(0))||any(uv>vec2f(1))||ndc.z<0.0||ndc.z>1.0){return 1.0;}
  return textureSampleCompareLevel(shadowTex,shadowSampler,uv,i32(cascade),ndc.z);
}

fn outgoingAtShortHit(hitWorld:vec3f,hit:ShortHit)->vec3f{
  let lightDirection=normalize(-frame.sunDirTime.xyz);
  let sun=frame.sunColorIntensity.xyz*frame.sunColorIntensity.w
    *max(0.0,dot(hit.normal,lightDirection))
    *fastSunVisibility(hitWorld,hit.normal);
  let toPoint=frame.pointPosRange.xyz-hitWorld;
  let distance=length(toPoint);
  let pointWindow=max(0.0,1.0-distance/frame.pointPosRange.w);
  let point=frame.pointColorIntensity.xyz*frame.pointColorIntensity.w
    *max(0.0,dot(hit.normal,toPoint/max(distance,1e-4)))
    *pointWindow*pointWindow/(1.0+0.06*distance*distance)
    *pointShadowVisibility(hitWorld,hit.normal);
  // Emissive geometry inside C(-1) is integrated analytically below. A
  // center-ray hit would quantize an entire angular bin and double count it.
  return hit.albedo*(sun+point);
}

fn pointAabbDistanceSquared(point:vec3f,minimum:vec3f,maximum:vec3f)->f32{
  let delta=max(max(minimum-point,vec3f(0)),point-maximum);
  return dot(delta,delta);
}
fn polygonEdgeIntegral(a:vec3f,b:vec3f)->vec3f{
  let edge=cross(a,b);
  let edgeLength=length(edge);
  if(edgeLength<1e-7){return vec3f(0);}
  return edge*(acos(clamp(dot(a,b),-1.0,1.0))/edgeLength);
}
fn clippedTriangleFormFactor(
  a:vec3f,b:vec3f,c:vec3f,normal:vec3f
)->f32{
  var input=array<vec3f,5>(a,b,c,vec3f(0),vec3f(0));
  var clipped:array<vec3f,5>;
  var clippedCount=0u;
  var previous=input[2];
  var previousDistance=dot(normal,previous);
  var previousInside=previousDistance>0.0;
  for(var index=0u;index<3u;index++){
    let current=input[index];
    let currentDistance=dot(normal,current);
    let currentInside=currentDistance>0.0;
    if(currentInside!=previousInside){
      let t=previousDistance/(previousDistance-currentDistance);
      clipped[clippedCount]=mix(previous,current,clamp(t,0.0,1.0));
      clippedCount++;
    }
    if(currentInside){clipped[clippedCount]=current;clippedCount++;}
    previous=current;
    previousDistance=currentDistance;
    previousInside=currentInside;
  }
  if(clippedCount<3u){return 0.0;}
  var vectorForm=vec3f(0);
  for(var index=0u;index<clippedCount;index++){
    let next=(index+1u)%clippedCount;
    vectorForm+=polygonEdgeIntegral(
      normalize(clipped[index]),normalize(clipped[next])
    );
  }
  return abs(dot(normal,vectorForm))*0.15915494309189535;
}
fn emissiveTriangleIrradiance(
  origin:vec3f,normal:vec3f,triangle:Triangle,radius:f32
)->vec3f{
  if(max(triangle.emissive.x,max(triangle.emissive.y,triangle.emissive.z))<=0.0){
    return vec3f(0);
  }
  let minimum=min(triangle.a.xyz,min(triangle.b.xyz,triangle.c.xyz));
  let maximum=max(triangle.a.xyz,max(triangle.b.xyz,triangle.c.xyz));
  let proximityDistance=sqrt(pointAabbDistanceSquared(origin,minimum,maximum));
  if(proximityDistance>=radius){return vec3f(0);}
  let aDelta=triangle.a.xyz-origin;
  let bDelta=triangle.b.xyz-origin;
  let cDelta=triangle.c.xyz-origin;
  if(min(length(aDelta),min(length(bDelta),length(cDelta)))<1e-5){return vec3f(0);}
  // Clip against the receiver horizon before evaluating Arvo's polygon form
  // factor. Without this, an emitter straddling the tangent plane cancels
  // itself and produces a dark band on near-coplanar receiving geometry.
  let formFactor=clippedTriangleFormFactor(aDelta,bDelta,cDelta,normal);
  if(formFactor<=1e-7){return vec3f(0);}
  let edgeAB=triangle.b.xyz-triangle.a.xyz;
  let edgeAC=triangle.c.xyz-triangle.a.xyz;
  let fromA=origin-triangle.a.xyz;
  let d00=dot(edgeAB,edgeAB);
  let d01=dot(edgeAB,edgeAC);
  let d11=dot(edgeAC,edgeAC);
  let d20=dot(fromA,edgeAB);
  let d21=dot(fromA,edgeAC);
  let inverseDenominator=1.0/max(1e-12,d00*d11-d01*d01);
  var barycentric=vec3f(
    1.0-(d11*d20-d01*d21)*inverseDenominator
      -(d00*d21-d01*d20)*inverseDenominator,
    (d11*d20-d01*d21)*inverseDenominator,
    (d00*d21-d01*d20)*inverseDenominator
  );
  barycentric=max(barycentric,vec3f(0));
  barycentric/=max(1e-8,barycentric.x+barycentric.y+barycentric.z);
  let closest=triangle.a.xyz*barycentric.x
    +triangle.b.xyz*barycentric.y+triangle.c.xyz*barycentric.z;
  let toLight=closest-origin;
  let lightDistance=length(toLight);
  if(lightDistance>0.012){
    let blocker=traceShortRangeWatertight(
      origin,toLight/lightDistance,max(1e-5,lightDistance-0.008)
    );
    if(blocker.t<lightDistance-0.01){return vec3f(0);}
  }
  let proximity=1.0-smoothstep(radius*0.72,radius,proximityDistance);
  return triangle.emissive.xyz*(formFactor*proximity);
}
fn nearEmissiveIrradiance(
  origin:vec3f,normal:vec3f,radius:f32
)->vec3f{
  var result=vec3f(0);
  var stack:array<u32,64>;
  var stackSize=1u;
  stack[0]=0u;
  let radiusSquared=radius*radius;
  loop{
    if(stackSize==0u){break;}
    stackSize-=1u;
    let nodeIndex=stack[stackSize];
    let node=emissiveBvhNodes[nodeIndex];
    if(pointAabbDistanceSquared(origin,node.minMeta.xyz,node.maxMeta.xyz)>radiusSquared){
      continue;
    }
    let left=bitcast<u32>(node.minMeta.w);
    let right=bitcast<u32>(node.maxMeta.w);
    if((left&0x80000000u)!=0u){
      let first=left&0x7fffffffu;
      for(var triangleOffset=0u;triangleOffset<right;triangleOffset++){
        result+=emissiveTriangleIrradiance(
          origin,normal,emissiveTriangles[first+triangleOffset],radius
        );
      }
    }else{
      if(stackSize>61u){break;}
      stack[stackSize]=left;
      stack[stackSize+1u]=right;
      stackSize+=2u;
    }
  }
  return result;
}

fn needsLocalInterval(
  pixel:vec2i,world:vec3f,normal:vec3f,backFacingSurface:bool
)->bool{
  if(backFacingSurface){return true;}
  let dimensions=vec2i(textureDimensions(worldTex));
  let offsets=array<vec2i,8>(
    vec2i(-4,0),vec2i(4,0),vec2i(0,-4),vec2i(0,4),
    vec2i(-4,-4),vec2i(4,-4),vec2i(-4,4),vec2i(4,4)
  );
  for(var index=0u;index<8u;index++){
    let samplePixel=pixel+offsets[index];
    if(any(samplePixel<vec2i(0))||any(samplePixel>=dimensions)){return true;}
    let neighbor=textureLoad(worldTex,samplePixel,0);
    if(neighbor.w<0.5){return true;}
    let neighborNormal=gbufferNormal(samplePixel);
    if(dot(normal,neighborNormal)<0.97){return true;}
    if(abs(dot(neighbor.xyz-world,normal))>frame.envBaseSpacing.w*0.075){
      return true;
    }
  }
  return false;
}

fn cMinusOneIrradiance(
  pixel:vec2i,world:vec3f,normal:vec3f,baseIrradiance:vec3f,
  backFacingSurface:bool
)->vec3f{
  // Section 7.1 defines C(-1) as the per-pixel interval from the surface to
  // t_-1 (normally delta_s0), followed by the directional c0 interval.  Do
  // that merge independently for every direction.  The former six-axis
  // enclosure classifier and "any miss" fallback switched an entire pixel
  // between unrelated estimators; box gaps therefore produced the large
  // black/white wedges reported in Cornell, and open rooms skipped C(-1)
  // altogether.  A miss now continues into the matching world-space c0 cone.
  let intervalEnd=frame.envBaseSpacing.w;
  // Thin-wall visibility guard for sparse trilinear interpolation.  It spans
  // the local c0/c1 neighborhood but only replaces the individual cone whose
  // direction has a verified first hit.  The fixed multiplier is universal;
  // it is neither a Cornell parameter nor selected from scene identity.
  let origin=world+normal*max(0.006,intervalEnd*0.012);
  let nearEmission=nearEmissiveIrradiance(origin,normal,intervalEnd);
  // C(-1) is only different from the already convolved c0 result near a
  // sub-cell depth/normal discontinuity. A conservative G-buffer footprint
  // skips expensive short BVH rays on smooth planar regions; this is the
  // screen-space locality optimization described for the paper's extension.
  if(!needsLocalInterval(pixel,world,normal,backFacingSurface)){
    return baseIrradiance+nearEmission;
  }
  let visibilityGuardEnd=select(
    intervalEnd,intervalEnd*15.0,backFacingSurface
  );
  var mergedResult=vec3f(0);
  for(var directionIndexValue=0u;directionIndexValue<32u;directionIndexValue++){
    let direction=directionFromIndex(directionIndexValue,0u);
    let cosine=max(0.0,dot(normal,direction));
    if(cosine<=0.0){continue;}
    let hit=traceShortRangeWatertight(
      origin,direction,visibilityGuardEnd+0.001
    );
    if(hit.t<visibilityGuardEnd){
      mergedResult+=outgoingAtShortHit(origin+direction*hit.t,hit)*cosine;
    }else{
      let continuation=sampleConeDirection(world,normal,directionIndexValue);
      mergedResult+=select(
        baseIrradiance,continuation.xyz,continuation.w>0.5
      )*cosine;
    }
  }
  return mergedResult*(4.0/32.0)+nearEmission;
}

@vertex fn fullscreenVS(@builtin(vertex_index) index:u32)->@builtin(position) vec4f{
  let uv=vec2f(f32((index<<1u)&2u),f32(index&2u));
  return vec4f(uv*2.0-1.0,0.0,1.0);
}
@fragment fn finalFS(@builtin(position) position:vec4f)->@location(0) vec4f{
  let pixel=vec2i(position.xy);
  let world=textureLoad(worldTex,pixel,0);
  let uv=position.xy/frame.resolution.xy;
  if(world.w<0.5){
    let sky=mix(frame.envBaseSpacing.xyz*0.35,frame.envBaseSpacing.xyz*1.5,clamp(1.0-uv.y,0.0,1.0));
    return vec4f(displayEncode(sky*frame.controls.y),1.0);
  }
  let albedoData=textureLoad(albedoTex,pixel,0);
  let albedo=albedoData.xyz;
  let normal=gbufferNormal(pixel);
  let sample=sampleIrradiance(world.xyz,normal);
  let resolvedIrradiance=cMinusOneIrradiance(
    pixel,world.xyz,normal,sample.xyz,fract(world.w)>0.25
  );
  let indirect=albedo*resolvedIrradiance*frame.controls.x;
  let L=normalize(-frame.sunDirTime.xyz);
  let sun=albedo*frame.sunColorIntensity.xyz*frame.sunColorIntensity.w*max(0.0,dot(normal,L))*shadowVisibility(world.xyz,normal);
  let toPoint=frame.pointPosRange.xyz-world.xyz;
  let distance=length(toPoint);
  let pointWindow=max(0.0,1.0-distance/frame.pointPosRange.w);
  let point=albedo*frame.pointColorIntensity.xyz*frame.pointColorIntensity.w*max(0.0,dot(normal,toPoint/max(distance,1e-4)))*pointWindow*pointWindow/(1.0+0.06*distance*distance)*pointShadowVisibility(world.xyz,normal);
  let emissive=surfaceEmission(pixel);
  let direct=sun+point+emissive;
  let mode=u32(frame.controls.z+0.5);
  var color=direct+indirect;
  if(mode==1u){color=indirect;}
  if(mode==2u){color=direct;}
  if(mode==3u){color=normal*0.5+0.5;}
  if(mode==4u){color=mix(vec3f(0.35,0.025,0.015),vec3f(0.05,1.0,0.55),sample.w);}
  if(mode==5u){color=albedo;}
  // Test-only mode 6 removes material color from the comparison so the
  // motion gate measures the reconstructed irradiance field itself.
  if(mode==6u){color=resolvedIrradiance;}
  if(mode<=2u||mode==6u){color=displayEncode(color*frame.controls.y);}
  return vec4f(color,1.0);
}
`;

// Split RC's temporal accumulation is entirely world-space (Section 5.2).
// Present the current composite directly so screen-space history cannot retain
// a camera-path-dependent image after the sparse field has already converged.
export const presentShader = /* wgsl */`
@vertex fn presentVS(@builtin(vertex_index) index:u32)->@builtin(position) vec4f{
  let uv=vec2f(f32((index<<1u)&2u),f32(index&2u));
  return vec4f(uv*2.0-1.0,0.0,1.0);
}

@group(0) @binding(0) var currentComposite: texture_2d<f32>;
@group(0) @binding(1) var currentCompositeSampler: sampler;
fn presentLuma(rgb:vec3f)->f32{return dot(rgb,vec3f(0.299,0.587,0.114));}
@fragment fn presentFS(@builtin(position) position:vec4f)->@location(0) vec4f{
  let size=vec2f(textureDimensions(currentComposite));
  let inverseSize=1.0/size;
  let uv=position.xy*inverseSize;
  let rgbM=textureSampleLevel(currentComposite,currentCompositeSampler,uv,0.0).xyz;
  let rgbNW=textureSampleLevel(currentComposite,currentCompositeSampler,uv+vec2f(-0.5,-0.5)*inverseSize,0.0).xyz;
  let rgbNE=textureSampleLevel(currentComposite,currentCompositeSampler,uv+vec2f(0.5,-0.5)*inverseSize,0.0).xyz;
  let rgbSW=textureSampleLevel(currentComposite,currentCompositeSampler,uv+vec2f(-0.5,0.5)*inverseSize,0.0).xyz;
  let rgbSE=textureSampleLevel(currentComposite,currentCompositeSampler,uv+vec2f(0.5,0.5)*inverseSize,0.0).xyz;
  let lumaM=presentLuma(rgbM);
  let lumaNW=presentLuma(rgbNW);
  let lumaNE=presentLuma(rgbNE);
  let lumaSW=presentLuma(rgbSW);
  let lumaSE=presentLuma(rgbSE);
  let lumaMin=min(lumaM,min(min(lumaNW,lumaNE),min(lumaSW,lumaSE)));
  let lumaMax=max(lumaM,max(max(lumaNW,lumaNE),max(lumaSW,lumaSE)));
  if(lumaMax-lumaMin<max(0.025,lumaMax*0.10)){return vec4f(rgbM,1.0);}
  var direction=vec2f(
    -((lumaNW+lumaNE)-(lumaSW+lumaSE)),
    (lumaNW+lumaSW)-(lumaNE+lumaSE)
  );
  let directionReduce=max((lumaNW+lumaNE+lumaSW+lumaSE)*(0.25*0.125),1.0/128.0);
  let inverseDirectionMin=1.0/(min(abs(direction.x),abs(direction.y))+directionReduce);
  direction=clamp(direction*inverseDirectionMin,vec2f(-8),vec2f(8))*inverseSize;
  let rgbA=0.5*(
    textureSampleLevel(currentComposite,currentCompositeSampler,uv+direction*(1.0/3.0-0.5),0.0).xyz+
    textureSampleLevel(currentComposite,currentCompositeSampler,uv+direction*(2.0/3.0-0.5),0.0).xyz
  );
  let rgbB=rgbA*0.5+0.25*(
    textureSampleLevel(currentComposite,currentCompositeSampler,uv+direction*-0.5,0.0).xyz+
    textureSampleLevel(currentComposite,currentCompositeSampler,uv+direction*0.5,0.0).xyz
  );
  let lumaB=presentLuma(rgbB);
  return vec4f(select(rgbB,rgbA,lumaB<lumaMin||lumaB>lumaMax),1.0);
}
`;
