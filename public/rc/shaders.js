export const shaderConstants = {
  hashOffsets: [0, 8192, 12288, 14336],
  hashSizes: [8192, 4096, 2048, 1024],
  probeOffsets: [0, 4096, 5632, 6144],
  probeCaps: [4096, 1536, 512, 256],
  dataOffsets: [0, 131072, 327680, 589824],
  directions: [32, 128, 512, 2048],
  totalHashSlots: 15360,
  hashFrames: 2,
  totalProbeMeta: 6400,
  totalDirectionData: 1114112,
  irradianceTexels: 64,
  irradianceFrames: 2,
  accumFrames: 2,
  stateWords: 19216,
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
@group(0) @binding(1) var albedoAtlas: texture_2d<f32>;
@group(0) @binding(2) var atlasSampler: sampler;
@group(0) @binding(3) var<uniform> pointShadowViewProj: mat4x4<f32>;

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
  @location(3) emissive: vec4f,
};
fn atlasUv(uv:vec2f,material:f32)->vec2f{
  let index=u32(max(0.0,material)+0.5);
  let cell=vec2u(index%5u,index/5u);
  return (vec2f(cell)*819.0+vec2f(4.0)+fract(uv)*811.0)/4096.0;
}
fn materialSample(uv:vec2f,material:f32)->vec4f{
  let sampled=textureSample(albedoAtlas,atlasSampler,atlasUv(uv,material));
  if(material<0.0){return vec4f(1);}
  if((u32(frame.cameraPos.w+0.5)&16u)==0u){return sampled;}
  let index=u32(material+0.5);
  let luminance=dot(sampled.rgb,vec3f(0.2126,0.7152,0.0722));
  let detail=clamp(0.72+luminance*0.34,0.62,1.04);
  let stone=vec3f(0.93,0.96,0.96)*detail;
  let cyan=vec3f(0.18,0.82,0.86)*mix(0.82,1.08,luminance);
  let paperColor=select(stone,cyan,index>=15u&&index<=19u);
  return vec4f(paperColor,sampled.a);
}
@fragment fn gbufferFS(v: VertexOut, @builtin(front_facing) frontFacing: bool) -> GBufferOut {
  var o: GBufferOut;
  let surface=materialSample(v.uv,v.materialCutoff.x);
  if(v.materialCutoff.y>0.0&&surface.a<v.materialCutoff.y){discard;}
  o.albedo = vec4f(v.albedo*surface.rgb, 1.0);
  o.normal = vec4f(normalize(select(-v.normal, v.normal, frontFacing)), 1.0);
  o.world = vec4f(v.world, 1.0);
  o.emissive = vec4f(v.emissive, 1.0);
  return o;
}
struct ShadowOut {
  @builtin(position) clip: vec4f,
  @location(0) uv: vec2f,
  @location(1) @interpolate(flat) materialCutoff: vec2f,
};
@vertex fn shadowVS(v: VertexIn) -> ShadowOut {
  var o:ShadowOut;
  o.clip=frame.sunViewProj*vec4f(v.position,1.0);
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
  o.clip=pointShadowViewProj*vec4f(v.position,1.0);
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
@group(0) @binding(12) var albedoAtlas: texture_2d<f32>;
@group(0) @binding(13) var albedoSampler: sampler;
@group(0) @binding(14) var irradianceAtlasStorage: texture_storage_2d<rgba16float,write>;
@group(0) @binding(15) var irradianceAtlasSampled: texture_2d<f32>;
@group(0) @binding(16) var irradianceAtlasSampler: sampler;
@group(0) @binding(17) var pointShadowAuditTex: texture_depth_2d_array;
@group(0) @binding(18) var pointShadowAuditSampler: sampler_comparison;
@group(0) @binding(19) var sunShadowAuditTex: texture_depth_2d;
@group(0) @binding(20) var sunShadowAuditSampler: sampler_comparison;

const EMPTY: u32 = 0xffffffffu;
const PI: f32 = 3.141592653589793;
const TAU: f32 = 6.283185307179586;
const FIXED_SCALE: f32 = 4096.0;
const HASH_OFFSETS = array<u32,4>(0u,8192u,12288u,14336u);
const HASH_SIZES = array<u32,4>(8192u,4096u,2048u,1024u);
const PROBE_OFFSETS = array<u32,4>(0u,4096u,5632u,6144u);
const PROBE_CAPS = array<u32,4>(4096u,1536u,512u,256u);
const DATA_OFFSETS = array<u32,4>(0u,131072u,327680u,589824u);
const DIR_COUNTS = array<u32,4>(32u,128u,512u,2048u);
const HASH_FRAME_STRIDE: u32 = 15360u;
const IRRADIANCE_FRAME_STRIDE: u32 = 262144u;
const ACCUM_FRAME_STRIDE: u32 = 5570560u;
const RAY_COUNT_OFFSET: u32 = 16u;
const RAY_OFFSET_OFFSET: u32 = 6416u;
const RAY_CURSOR_OFFSET: u32 = 12816u;
const BLOCK_COUNT_OFFSET: u32 = 19216u;
const TOTAL_PROBE_META: u32 = 6400u;
const SECONDARY_TAG: u32 = 256u;

fn hash32(value: u32) -> u32 {
  var x = value;
  x = x ^ (x >> 16u);
  x = x * 0x7feb352du;
  x = x ^ (x >> 15u);
  x = x * 0x846ca68bu;
  return x ^ (x >> 16u);
}

fn keyFromCell(cellIn: vec3i, lod: u32) -> u32 {
  if(any(cellIn<vec3i(-256))||any(cellIn>vec3i(255))){return EMPTY;}
  let c = cellIn + vec3i(256);
  return u32(c.x) | (u32(c.y) << 9u) | (u32(c.z) << 18u)
    | ((lod & 7u) << 27u) | (((lod >> 8u) & 1u) << 30u);
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
  return frame.envBaseSpacing.w * exp2(f32(cascade + (lod & 7u)));
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

fn insertProbeKey(cascade: u32, position: vec3f, lod: u32) {
  insertProbeRaw(cascade,keyFromCell(probeCell(position,cascade,lod),lod));
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
  let cell=vec2u(index%5u,index/5u);
  let atlasUv=(vec2f(cell)*819.0+vec2f(4.0)+fract(uv)*811.0)/4096.0;
  let sampled=textureSampleLevel(albedoAtlas,albedoSampler,atlasUv,0.0);
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
  let e1=tri.b.xyz-tri.a.xyz;
  let e2=tri.c.xyz-tri.a.xyz;
  let p=cross(direction,e2);
  let det=dot(e1,p);
  if(abs(det)<1e-7){return vec3f(maxDistance,0,0);}
  let inv=1.0/det;
  let tv=origin-tri.a.xyz;
  let u=dot(tv,p)*inv;
  if(u<0.0||u>1.0){return vec3f(maxDistance,0,0);}
  let q=cross(tv,e1);
  let v=dot(direction,q)*inv;
  if(v<0.0||u+v>1.0){return vec3f(maxDistance,0,0);}
  let t=dot(e2,q)*inv;
  if(t>0.001&&t<maxDistance){return vec3f(t,u,v);}
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

fn sampleAtlasIrradiance(probe:u32,normalIn:vec3f,frameIndex:u32)->vec3f{
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
  let tile=vec2f(f32(probe%64u)*8.0,f32(probe/64u)*8.0+f32(frameIndex)*512.0);
  let atlasUv=(tile+octCoordinate+vec2f(0.5))/vec2f(512.0,1024.0);
  return textureSampleLevel(
    irradianceAtlasSampled,irradianceAtlasSampler,atlasUv,0.0
  ).xyz;
}

fn samplePrimaryIrradianceLod(position:vec3f,normal:vec3f,lod:u32)->vec4f{
  let spacing=cascadeSpacing(0u,lod);
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
    let spatialWeight=wv.x*wv.y*wv.z;
    let probe=lookupProbe(0u,keyFromCell(cell+bits,lod));
    if(probe!=EMPTY&&probe<PROBE_CAPS[0]){
      value+=sampleAtlasIrradiance(probe,normal,currentFrame())*spatialWeight;
      total+=spatialWeight;
    }
  }
  if(total<1e-5){return vec4f(0,0,0,0);}
  return vec4f(value/total,clamp(total,0.0,1.0));
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

// Secondary cache from the previous frame. It uses the same sparse machinery
// with an independent key tag and a base LOD two levels coarser, as described
// in the paper's multibounce implementation.
fn sampleSecondaryHistory(position:vec3f,normal:vec3f)->vec3f{
  if(!featureEnabled(1u)||historyWeight()<=0.0){return vec3f(0);}
  let lod=(min(levelOfDetail(position)+2u,7u)|SECONDARY_TAG);
  let spacing=cascadeSpacing(0u,lod);
  let grid=position/spacing-vec3f(0.5);
  let cell=vec3i(floor(grid));
  let fraction=fract(grid);
  let previousFrame=1u-currentFrame();
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
    let probe=lookupProbeFrame(0u,keyFromCell(cell+bits,lod),previousFrame);
    if(probe!=EMPTY&&probe<PROBE_CAPS[0]){
      value+=sampleAtlasIrradiance(probe,normal,previousFrame)*weight;
      total+=weight;
    }
  }
  if(total<1e-5){return vec3f(0);}
  return value/total;
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
  let secondary=sampleSecondaryHistory(position,hit.normal);
  return min(vec3f(16.0),hit.emissive.xyz+hit.albedo.xyz*(sun+point+secondary*0.82));
}
`;

export const computeShader = sharedCompute + /* wgsl */`
@compute @workgroup_size(256) fn resetSlots(@builtin(global_invocation_id) gid: vec3u) {
  if(gid.x>=HASH_FRAME_STRIDE){return;}
  let slot=currentFrame()*HASH_FRAME_STRIDE+gid.x;
  atomicStore(&slots[slot].key,EMPTY);
  atomicStore(&slots[slot].index,EMPTY);
}

@compute @workgroup_size(64) fn retainPreviousProbes(@builtin(global_invocation_id) gid:vec3u){
  if(!featureEnabled(8u)||historyWeight()<=0.0||gid.x>=HASH_FRAME_STRIDE){return;}
  var cascade=0u;
  if(gid.x>=HASH_OFFSETS[3]){cascade=3u;}
  else if(gid.x>=HASH_OFFSETS[2]){cascade=2u;}
  else if(gid.x>=HASH_OFFSETS[1]){cascade=1u;}
  let previousBase=(1u-currentFrame())*HASH_FRAME_STRIDE;
  let key=atomicLoad(&slots[previousBase+gid.x].key);
  if(key==EMPTY){return;}
  let lod=lodFromKey(key);
  let position=probePositionFromCell(cellFromKey(key),cascade,lod);
  let clip=frame.viewProj*vec4f(position,1.0);
  if(clip.w<=0.0){return;}
  let ndc=clip.xyz/clip.w;
  // Preserve the complete cascade ancestry, including the secondary cache,
  // across camera motion. A guard band keeps this a bounded sparse volume.
  if(abs(ndc.x)>1.18||abs(ndc.y)>1.18||ndc.z<0.0||ndc.z>1.0){return;}
  insertProbeRaw(cascade,key);
}

@compute @workgroup_size(8,8) fn initBase(@builtin(global_invocation_id) gid: vec3u) {
  let fullSize=vec2u(u32(frame.resolution.x),u32(frame.resolution.y));
  if(any(gid.xy>=fullSize)){return;}
  // Algorithm 1 initializes the nearest c0 probe for every visible pixel.
  // Ray generation remains independently budgeted by the lower-resolution
  // GI sampling grid, but probe discovery must not discard thin or distant
  // surfaces that happen to fall between those samples.
  let world=textureLoad(worldTex,vec2i(gid.xy),0);
  if(world.w<0.5){return;}
  let lods=lodSelection(world.xyz);
  let fine=u32(lods.x);
  insertProbeKey(0u,world.xyz,fine);
  let coarse=u32(lods.y);
  if(coarse!=fine){insertProbeKey(0u,world.xyz,coarse);}
}

fn hitRecordIndex(gid:vec3u,frameIndex:u32)->u32{
  let giSize=vec2u(u32(frame.resolution.z),u32(frame.resolution.w));
  let samplesPerFrame=giSize.x*giSize.y*max(1u,passParams.value);
  let sample=(gid.z*giSize.y+gid.y)*giSize.x+gid.x;
  return TOTAL_PROBE_META+(frameIndex*samplesPerFrame+sample)*2u;
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
  return TOTAL_PROBE_META+samplesPerFrame()*4u;
}

fn rayBlockSize()->u32{
  return max(1u,passParams.pad0);
}

fn rayBlockCount()->u32{
  return (samplesPerFrame()*3u+rayBlockSize()-1u)/rayBlockSize();
}

fn rayBlockStateIndex(probe:u32,block:u32)->u32{
  return BLOCK_COUNT_OFFSET+probe*rayBlockCount()+block;
}

@compute @workgroup_size(8,8) fn initSecondary(@builtin(global_invocation_id) gid: vec3u) {
  let giSize=vec2u(u32(frame.resolution.z),u32(frame.resolution.w));
  if(!featureEnabled(1u)||historyWeight()<=0.0||any(gid.xy>=giSize)||gid.z>=max(1u,passParams.value)){return;}
  let record=hitRecordIndex(gid,1u-currentFrame());
  let hitPosition=probeMeta[record];
  if(hitPosition.w<0.5){return;}
  let lod=min(levelOfDetail(hitPosition.xyz)+2u,7u)|SECONDARY_TAG;
  insertProbeKey(0u,hitPosition.xyz,lod);
}

@compute @workgroup_size(64) fn initHigher(@builtin(global_invocation_id) gid: vec3u) {
  let cascade=passParams.cascade;
  if(cascade==0u||cascade>3u){return;}
  let previous=cascade-1u;
  let activeCount=min(atomicLoad(&state[previous]),PROBE_CAPS[previous]);
  if(gid.x>=activeCount){return;}
  let probeInfo=probeMeta[PROBE_OFFSETS[previous]+gid.x];
  insertProbeKey(cascade,probeInfo.xyz,bitcast<u32>(probeInfo.w));
}

fn cellFromKey(key:u32)->vec3i{
  return vec3i(
    i32(key&511u)-256,
    i32((key>>9u)&511u)-256,
    i32((key>>18u)&511u)-256
  );
}

fn lodFromKey(key:u32)->u32{
  return ((key>>27u)&7u)|(((key>>30u)&1u)<<8u);
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
    bitcast<f32>(lod)
  );
}

@compute @workgroup_size(8,8) fn countBaseRays(@builtin(global_invocation_id) gid:vec3u){
  let giSize=vec2u(u32(frame.resolution.z),u32(frame.resolution.w));
  if(any(gid.xy>=giSize)){return;}
  let fullSize=vec2u(u32(frame.resolution.x),u32(frame.resolution.y));
  let pixel=min(fullSize-vec2u(1),gid.xy*fullSize/giSize);
  let world=textureLoad(worldTex,vec2i(pixel),0);
  if(world.w<0.5){return;}
  let lods=lodSelection(world.xyz);
  let fine=u32(lods.x);
  let fineProbe=lookupProbe(0u,keyFromCell(probeCell(world.xyz,0u,fine),fine));
  if(fineProbe!=EMPTY){
    atomicAdd(&state[probeStateIndex(RAY_COUNT_OFFSET,0u,fineProbe)],max(1u,passParams.value));
  }
  let coarse=u32(lods.y);
  if(coarse!=fine){
    let coarseProbe=lookupProbe(0u,keyFromCell(probeCell(world.xyz,0u,coarse),coarse));
    if(coarseProbe!=EMPTY){
      atomicAdd(&state[probeStateIndex(RAY_COUNT_OFFSET,0u,coarseProbe)],max(1u,passParams.value));
    }
  }
}

@compute @workgroup_size(8,8) fn countSecondaryRays(@builtin(global_invocation_id) gid:vec3u){
  let giSize=vec2u(u32(frame.resolution.z),u32(frame.resolution.w));
  if(!featureEnabled(1u)||historyWeight()<=0.0||any(gid.xy>=giSize)||gid.z>=max(1u,passParams.value)){return;}
  let record=hitRecordIndex(gid,1u-currentFrame());
  let hitPosition=probeMeta[record];
  if(hitPosition.w<0.5){return;}
  let lod=min(levelOfDetail(hitPosition.xyz)+2u,7u)|SECONDARY_TAG;
  let probe=lookupProbe(0u,keyFromCell(probeCell(hitPosition.xyz,0u,lod),lod));
  if(probe!=EMPTY){atomicAdd(&state[probeStateIndex(RAY_COUNT_OFFSET,0u,probe)],1u);}
}

@compute @workgroup_size(64) fn countHigherRays(@builtin(global_invocation_id) gid:vec3u){
  let cascade=passParams.cascade;
  if(cascade==0u||cascade>3u){return;}
  let previous=cascade-1u;
  let childCount=min(atomicLoad(&state[previous]),PROBE_CAPS[previous]);
  if(gid.x>=childCount){return;}
  let childInfo=probeMeta[PROBE_OFFSETS[previous]+gid.x];
  let lod=bitcast<u32>(childInfo.w);
  let parent=lookupProbe(cascade,keyFromCell(probeCell(childInfo.xyz,cascade,lod),lod));
  if(parent!=EMPTY){
    let childRays=atomicLoad(&state[probeStateIndex(RAY_COUNT_OFFSET,previous,gid.x)]);
    atomicAdd(&state[probeStateIndex(RAY_COUNT_OFFSET,cascade,parent)],childRays);
  }
}

fn probeKeyFromInfo(info:vec4f,cascade:u32)->u32{
  let lod=bitcast<u32>(info.w);
  return keyFromCell(probeCell(info.xyz,cascade,lod),lod);
}

fn parentKeyFromInfo(info:vec4f,cascade:u32)->u32{
  let lod=bitcast<u32>(info.w);
  return keyFromCell(probeCell(info.xyz,cascade+1u,lod),lod);
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
    let lod=bitcast<u32>(info.w);
    let parentCell=probeCell(info.xyz,cascade+1u,lod);
    for(var child=0u;child<8u;child++){
      let bits=vec3i(i32(child&1u),i32((child>>1u)&1u),i32((child>>2u)&1u));
      let siblingKey=keyFromCell(parentCell*2+bits,lod);
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

fn mapRaySample(world:vec3f,lod:u32,stableSlot:u32){
  let probe=lookupProbe(0u,keyFromCell(probeCell(world,0u,lod),lod));
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
  let lods=lodSelection(world.xyz);
  let sample=sampleIndex(gid);
  mapRaySample(world.xyz,u32(lods.x),sample);
  if(u32(lods.y)!=u32(lods.x)){
    mapRaySample(world.xyz,u32(lods.y),samplesPerFrame()+sample);
  }
}

@compute @workgroup_size(8,8) fn mapSecondaryRaySamples(@builtin(global_invocation_id) gid:vec3u){
  let giSize=vec2u(u32(frame.resolution.z),u32(frame.resolution.w));
  if(!featureEnabled(1u)||historyWeight()<=0.0||any(gid.xy>=giSize)||gid.z>=max(1u,passParams.value)){return;}
  let record=hitRecordIndex(gid,1u-currentFrame());
  let hitPosition=probeMeta[record];
  if(hitPosition.w<0.5){return;}
  let lod=min(levelOfDetail(hitPosition.xyz)+2u,7u)|SECONDARY_TAG;
  mapRaySample(hitPosition.xyz,lod,samplesPerFrame()*2u+sampleIndex(gid));
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

fn traceAndSplit(world:vec3f,normal:vec3f,lod:u32,stableSlot:u32,record:u32,writeHit:bool){
  let key0=keyFromCell(probeCell(world,0u,lod),lod);
  let baseProbe=lookupProbe(0u,key0);
  if(baseProbe==EMPTY){
    if(writeHit){probeMeta[record]=vec4f(0);probeMeta[record+1u]=vec4f(0);}
    return;
  }
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
  // every retained probe keeps receiving new samples instead of becoming
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

  let baseLength=frame.envBaseSpacing.w*exp2(f32(lod&7u))*1.6;
  let maxDistance=baseLength*64.0;
  let nearOffset=select(0.0,frame.envBaseSpacing.w*exp2(f32(lod&7u)),featureEnabled(4u));
  let origin=world+normal*max(0.008,frame.envBaseSpacing.w*0.012)+direction*nearOffset;
  let hit=traceScene(origin,direction,maxDistance+0.001);
  let didHit=hit.t<maxDistance;
  if(writeHit){
    if(didHit){
      probeMeta[record]=vec4f(origin+direction*hit.t,1.0);
      probeMeta[record+1u]=vec4f(hit.normal,1.0);
    } else {
      probeMeta[record]=vec4f(0);
      probeMeta[record+1u]=vec4f(0);
    }
  }
  var targetCascade=3u;
  if(didHit){
    targetCascade=0u;
    var end=baseLength;
    let intervalDistance=hit.t+nearOffset;
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
    let key=keyFromCell(probeCell(world,cascade,lod),lod);
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
  let record=hitRecordIndex(gid,currentFrame());
  let world=textureLoad(worldTex,vec2i(pixel),0);
  if(world.w<0.5){probeMeta[record]=vec4f(0);probeMeta[record+1u]=vec4f(0);return;}
  let normal=normalize(textureLoad(normalTex,vec2i(pixel),0).xyz);
  let lods=lodSelection(world.xyz);
  let fine=u32(lods.x);
  let sample=sampleIndex(gid);
  traceAndSplit(world.xyz,normal,fine,sample,record,true);
  let coarse=u32(lods.y);
  if(coarse!=fine){traceAndSplit(world.xyz,normal,coarse,samplesPerFrame()+sample,0u,false);}
}

@compute @workgroup_size(8,8) fn splitSecondaryRays(@builtin(global_invocation_id) gid: vec3u) {
  let giSize=vec2u(u32(frame.resolution.z),u32(frame.resolution.w));
  if(!featureEnabled(1u)||historyWeight()<=0.0||any(gid.xy>=giSize)||gid.z>=max(1u,passParams.value)){return;}
  let record=hitRecordIndex(gid,1u-currentFrame());
  let hitPosition=probeMeta[record];
  let hitNormal=probeMeta[record+1u];
  if(hitPosition.w<0.5||hitNormal.w<0.5){return;}
  let lod=min(levelOfDetail(hitPosition.xyz)+2u,7u)|SECONDARY_TAG;
  traceAndSplit(
    hitPosition.xyz,
    normalize(hitNormal.xyz),
    lod,
    samplesPerFrame()*2u+sampleIndex(gid),
    0u,
    false
  );
}

fn sampleParentDirection(cascade:u32,position:vec3f,lod:u32,parentDirection:u32)->vec4f{
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
    let probe=lookupProbe(parent,keyFromCell(cell+bits,lod));
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

fn mergedParent(cascade:u32,direction:u32,position:vec3f,lod:u32)->vec4f{
  var sum=vec3f(0);
  var valid=0.0;
  for(var child=0u;child<4u;child++){
    // Morton ordering makes the four angular children of every lower
    // direction one contiguous aligned group, matching Section 6.
    let parentDirection=direction*4u+child;
    let parent=sampleParentDirection(cascade,position,lod,parentDirection);
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
  if(samples>0u){
    let denominator=FIXED_SCALE*f32(samples);
    interval=vec3f(f32(atomicLoad(&accum[base])),f32(atomicLoad(&accum[base+1u])),f32(atomicLoad(&accum[base+2u])))/denominator;
    beta=f32(atomicLoad(&accum[base+3u]))/denominator;
  }
  let probeInfo=probeMeta[PROBE_OFFSETS[cascade]+probe];
  let lod=bitcast<u32>(probeInfo.w);
  let key=keyFromCell(probeCell(probeInfo.xyz,cascade,lod),lod);
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
        let temporalWeight=intervalHistoryWeight();
        interval=mix(interval,previousInterval,temporalWeight);
        beta=mix(beta,previousBeta,temporalWeight);
      }else{
        interval=previousInterval;
        beta=previousBeta;
      }
    }
  }
  if(hasInterval){
    let safeInterval=clamp(interval,vec3f(0),vec3f(16));
    atomicStore(&accum[base],u32(safeInterval.r*FIXED_SCALE+0.5));
    atomicStore(&accum[base+1u],u32(safeInterval.g*FIXED_SCALE+0.5));
    atomicStore(&accum[base+2u],u32(safeInterval.b*FIXED_SCALE+0.5));
    atomicStore(&accum[base+3u],u32(clamp(beta,0.0,1.0)*FIXED_SCALE+0.5));
    atomicStore(&accum[base+4u],1u);
  }
  if(!hasInterval){
    // Section 5 explicitly ignores zero-count directions. Treating one as a
    // transparent interval incorrectly exposes the environment and produces
    // stable but severe bright leaks.
    cones[index]=vec4f(0);
    return;
  }
  var distant=vec4f(frame.envBaseSpacing.xyz,1.0);
  if(cascade<3u){distant=mergedParent(cascade,direction,probeInfo.xyz,lod);}
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
  let atlasCoordinate=vec2i(tile+vec2u(x,y)+vec2u(0u,currentFrame()*512u));
  textureStore(irradianceAtlasStorage,atlasCoordinate,stored);
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
  let normal=normalize(textureLoad(normalTex,vec2i(pixel),0).xyz);
  let base=(gid.y*outputSize.x+gid.x)*8u;
  if(gid.z==0u){
    let current=clamp(samplePrimaryIrradiance(world.xyz,normal).xyz,vec3f(0),vec3f(16));
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

fn sunShadowAuditVisibility(world:vec3f,normal:vec3f)->f32{
  let clip=frame.sunViewProj*vec4f(world+normal*0.012,1.0);
  let ndc=clip.xyz/clip.w;
  let uv=vec2f(ndc.x*0.5+0.5,0.5-ndc.y*0.5);
  if(any(uv<vec2f(0))||any(uv>vec2f(1))||ndc.z<0.0||ndc.z>1.0){return 1.0;}
  let size=vec2f(textureDimensions(sunShadowAuditTex));
  var result=0.0;
  for(var y=-1;y<=1;y++){
    for(var x=-1;x<=1;x++){
      result+=textureSampleCompareLevel(
        sunShadowAuditTex,sunShadowAuditSampler,
        uv+vec2f(f32(x),f32(y))/size,ndc.z-0.0015
      );
    }
  }
  return result/9.0;
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
  let normal=normalize(textureLoad(normalTex,vec2i(pixel),0).xyz);
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
struct HashSlot { key: atomic<u32>, index: atomic<u32> };
@group(0) @binding(0) var<uniform> frame: FrameUniforms;
@group(0) @binding(1) var albedoTex: texture_2d<f32>;
@group(0) @binding(2) var normalTex: texture_2d<f32>;
@group(0) @binding(3) var worldTex: texture_2d<f32>;
@group(0) @binding(4) var shadowTex: texture_depth_2d;
@group(0) @binding(5) var shadowSampler: sampler_comparison;
@group(0) @binding(6) var<storage,read_write> slots: array<HashSlot>;
@group(0) @binding(7) var irradianceAtlas: texture_2d<f32>;
@group(0) @binding(8) var<storage,read> cones: array<vec4f>;
@group(0) @binding(9) var<storage,read_write> accum: array<atomic<u32>>;
@group(0) @binding(10) var irradianceSampler: sampler;
@group(0) @binding(11) var emissiveTex: texture_2d<f32>;
@group(0) @binding(12) var pointShadowTex: texture_depth_2d_array;
@group(0) @binding(13) var pointShadowSampler: sampler_comparison;

const EMPTY:u32=0xffffffffu;
const HASH_FRAME_STRIDE:u32=15360u;
const IRRADIANCE_FRAME_STRIDE:u32=262144u;
const ACCUM_FRAME_STRIDE:u32=5570560u;
const FIXED_SCALE:f32=4096.0;
const HASH_OFFSETS=array<u32,4>(0u,8192u,12288u,14336u);
const HASH_SIZES=array<u32,4>(8192u,4096u,2048u,1024u);
const DATA_OFFSETS=array<u32,4>(0u,131072u,327680u,589824u);
const DIR_COUNTS=array<u32,4>(32u,128u,512u,2048u);
fn hash32(value:u32)->u32{
  var x=value;x=x^(x>>16u);x=x*0x7feb352du;x=x^(x>>15u);x=x*0x846ca68bu;return x^(x>>16u);
}
fn keyFromCell(cellIn:vec3i,lod:u32)->u32{
  if(any(cellIn<vec3i(-256))||any(cellIn>vec3i(255))){return EMPTY;}
  let c=cellIn+vec3i(256);
  return u32(c.x)|(u32(c.y)<<9u)|(u32(c.z)<<18u)|((lod&7u)<<27u);
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
  for(var corner=0u;corner<8u;corner++){
    let bits=vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u));
    let wv=vec3f(
      select(1.0-fraction.x,fraction.x,bits.x==1),
      select(1.0-fraction.y,fraction.y,bits.y==1),
      select(1.0-fraction.z,fraction.z,bits.z==1)
    );
    let spatialWeight=wv.x*wv.y*wv.z;
    let probe=lookupProbe(keyFromCell(cell+bits,lod));
    if(probe!=EMPTY&&probe<4096u){
      let tile=vec2f(f32(probe%64u)*8.0,f32(probe/64u)*8.0+f32(frameIndex)*512.0);
      let atlasUv=(tile+octCoordinate+vec2f(0.5))/vec2f(512.0,1024.0);
      value+=textureSampleLevel(irradianceAtlas,irradianceSampler,atlasUv,0.0).xyz*spatialWeight;
      total+=spatialWeight;
    }
  }
  if(total<1e-5){return vec4f(0,0,0,0);}
  return vec4f(value/total,clamp(total,0.0,1.0));
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
fn sampleConeLod(cascade:u32,position:vec3f,direction:vec3f,lod:u32)->vec4f{
  let spacing=frame.envBaseSpacing.w*exp2(f32(cascade+lod));
  let grid=position/spacing-vec3f(0.5);
  let cell=vec3i(floor(grid));
  let fraction=fract(grid);
  let directionId=directionIndex(direction,cascade);
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
    let probe=lookupProbeCascade(cascade,keyFromCell(cell+bits,lod));
    if(probe!=EMPTY){
      let cone=cones[dataIndex(cascade,probe,directionId)];
      if(cone.w>0.5){
        value+=cone.xyz*weight;
        total+=weight;
      }
    }
  }
  if(total<1e-5){return vec4f(frame.envBaseSpacing.xyz,0.0);}
  return vec4f(value/total,total);
}
fn sampleIntervalLod(cascade:u32,position:vec3f,direction:vec3f,lod:u32)->vec4f{
  let spacing=frame.envBaseSpacing.w*exp2(f32(cascade+lod));
  let grid=position/spacing-vec3f(0.5);
  let cell=vec3i(floor(grid));
  let fraction=fract(grid);
  let directionId=directionIndex(direction,cascade);
  var radiance=vec3f(0);
  var beta=0.0;
  var total=0.0;
  for(var corner=0u;corner<8u;corner++){
    let bits=vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u));
    let wv=vec3f(
      select(1.0-fraction.x,fraction.x,bits.x==1),
      select(1.0-fraction.y,fraction.y,bits.y==1),
      select(1.0-fraction.z,fraction.z,bits.z==1)
    );
    let weight=wv.x*wv.y*wv.z;
    let probe=lookupProbeCascade(cascade,keyFromCell(cell+bits,lod));
    if(probe!=EMPTY){
      let base=accumIndex(cascade,probe,directionId);
      let samples=atomicLoad(&accum[base+4u]);
      if(samples>0u){
        let denominator=FIXED_SCALE*f32(samples);
        radiance+=vec3f(
          f32(atomicLoad(&accum[base])),
          f32(atomicLoad(&accum[base+1u])),
          f32(atomicLoad(&accum[base+2u]))
        )/denominator*weight;
        beta+=f32(atomicLoad(&accum[base+3u]))/denominator*weight;
        total+=weight;
      }
    }
  }
  if(total<1e-5){return vec4f(0,0,0,1);}
  return vec4f(radiance/total,clamp(beta/total,0.0,1.0));
}
fn roughSpecularLod(position:vec3f,normal:vec3f,lod:u32)->vec3f{
  let viewDirection=normalize(frame.cameraPos.xyz-position);
  let reflection=reflect(-viewDirection,normal);
  var radiance=sampleConeLod(2u,position,reflection,lod).xyz;
  for(var cascade=1i;cascade>=0i;cascade--){
    let interval=sampleIntervalLod(u32(cascade),position,reflection,lod);
    radiance=interval.xyz+interval.w*radiance;
  }
  let fresnel=0.04+0.18*pow(1.0-max(0.0,dot(normal,viewDirection)),5.0);
  return min(vec3f(8),radiance*fresnel);
}
fn roughSpecular(position:vec3f,normal:vec3f)->vec3f{
  let lods=lodSelection(position);
  let fine=roughSpecularLod(position,normal,u32(lods.x));
  if(u32(lods.y)==u32(lods.x)){return fine;}
  let coarse=roughSpecularLod(position,normal,u32(lods.y));
  return mix(fine,coarse,lods.z);
}
fn shadowVisibility(world:vec3f,normal:vec3f)->f32{
  let clip=frame.sunViewProj*vec4f(world+normal*0.012,1.0);
  let ndc=clip.xyz/clip.w;
  let uv=vec2f(ndc.x*0.5+0.5,0.5-ndc.y*0.5);
  if(any(uv<vec2f(0))||any(uv>vec2f(1))||ndc.z<0.0||ndc.z>1.0){return 1.0;}
  let size=vec2f(textureDimensions(shadowTex));
  var result=0.0;
  for(var y=-1;y<=1;y++){
    for(var x=-1;x<=1;x++){
      result+=textureSampleCompareLevel(shadowTex,shadowSampler,uv+vec2f(f32(x),f32(y))/size,ndc.z-0.0015);
    }
  }
  return result/9.0;
}
fn surfaceEmission(pixel:vec2i)->vec3f{
  return textureLoad(emissiveTex,pixel,0).xyz;
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
fn directRadianceAtPixel(pixel:vec2i)->vec3f{
  let world=textureLoad(worldTex,pixel,0);
  let albedoData=textureLoad(albedoTex,pixel,0);
  let albedo=albedoData.xyz;
  let normal=normalize(textureLoad(normalTex,pixel,0).xyz);
  let lightDirection=normalize(-frame.sunDirTime.xyz);
  let sun=albedo*frame.sunColorIntensity.xyz*frame.sunColorIntensity.w
    *max(0.0,dot(normal,lightDirection))*shadowVisibility(world.xyz,normal);
  let toPoint=frame.pointPosRange.xyz-world.xyz;
  let distance=length(toPoint);
  let pointWindow=max(0.0,1.0-distance/frame.pointPosRange.w);
  let point=albedo*frame.pointColorIntensity.xyz*frame.pointColorIntensity.w
    *max(0.0,dot(normal,toPoint/max(distance,1e-4)))*pointWindow*pointWindow
    /(1.0+0.06*distance*distance)*pointShadowVisibility(world.xyz,normal);
  return sun+point+surfaceEmission(pixel);
}
fn screenSpaceNearInterval(position:vec3f,direction:vec3f)->vec4f{
  let size=vec2i(textureDimensions(worldTex));
  let nearLength=frame.envBaseSpacing.w;
  for(var step=1u;step<=8u;step++){
    let distance=nearLength*f32(step)/8.0;
    let expected=position+direction*distance;
    let clip=frame.viewProj*vec4f(expected,1.0);
    if(clip.w<=0.0){continue;}
    let ndc=clip.xy/clip.w;
    let uv=vec2f(ndc.x*0.5+0.5,0.5-ndc.y*0.5);
    if(any(uv<vec2f(0))||any(uv>vec2f(1))){continue;}
    let pixel=clamp(vec2i(uv*vec2f(size)),vec2i(0),size-vec2i(1));
    let candidate=textureLoad(worldTex,pixel,0);
    if(candidate.w<0.5){continue;}
    let delta=candidate.xyz-position;
    let along=dot(delta,direction);
    let perpendicular=length(delta-direction*along);
    let thickness=max(frame.envBaseSpacing.w*0.08,distance*0.035);
    if(along>0.01&&abs(along-distance)<thickness&&perpendicular<thickness){
      return vec4f(directRadianceAtPixel(pixel),0.0);
    }
  }
  return vec4f(0,0,0,1);
}
fn directionalCMinusOneLod(position:vec3f,normal:vec3f,lod:u32)->vec3f{
  var result=vec3f(0);
  for(var index=0u;index<32u;index++){
    let direction=directionFromIndex(index,0u);
    let cosine=max(0.0,dot(normal,direction));
    if(cosine<=0.0){continue;}
    let near=screenSpaceNearInterval(position,direction);
    let distant=sampleConeLod(0u,position,direction,lod).xyz;
    result+=(near.xyz+near.w*distant)*cosine;
  }
  return result*(4.0/32.0);
}
fn directionalCMinusOne(position:vec3f,normal:vec3f)->vec3f{
  let lods=lodSelection(position);
  let fine=directionalCMinusOneLod(position,normal,u32(lods.x));
  if(u32(lods.y)==u32(lods.x)){return fine;}
  let coarse=directionalCMinusOneLod(position,normal,u32(lods.y));
  return mix(fine,coarse,lods.z);
}
fn aces(x:vec3f)->vec3f{
  let a=2.51;let b=0.03;let c=2.43;let d=0.59;let e=0.14;
  return clamp((x*(a*x+b))/(x*(c*x+d)+e),vec3f(0),vec3f(1));
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
    return vec4f(aces(sky*frame.controls.y),1.0);
  }
  let albedoData=textureLoad(albedoTex,pixel,0);
  let albedo=albedoData.xyz;
  let normal=normalize(textureLoad(normalTex,pixel,0).xyz);
  let features=u32(frame.cameraPos.w+0.5);
  var sample=sampleIrradiance(world.xyz,normal);
  if((features&4u)!=0u){sample=vec4f(directionalCMinusOne(world.xyz,normal),sample.w);}
  let indirect=albedo*sample.xyz*frame.controls.x;
  let L=normalize(-frame.sunDirTime.xyz);
  let sun=albedo*frame.sunColorIntensity.xyz*frame.sunColorIntensity.w*max(0.0,dot(normal,L))*shadowVisibility(world.xyz,normal);
  let toPoint=frame.pointPosRange.xyz-world.xyz;
  let distance=length(toPoint);
  let pointWindow=max(0.0,1.0-distance/frame.pointPosRange.w);
  let point=albedo*frame.pointColorIntensity.xyz*frame.pointColorIntensity.w*max(0.0,dot(normal,toPoint/max(distance,1e-4)))*pointWindow*pointWindow/(1.0+0.06*distance*distance)*pointShadowVisibility(world.xyz,normal);
  let emissive=surfaceEmission(pixel);
  let direct=sun+point+emissive;
  let specular=select(vec3f(0),roughSpecular(world.xyz,normal),(features&2u)!=0u);
  let mode=u32(frame.controls.z+0.5);
  var color=direct+indirect+specular;
  if(mode==1u){color=indirect;}
  if(mode==2u){color=direct;}
  if(mode==3u){color=normal*0.5+0.5;}
  if(mode==4u){color=mix(vec3f(0.35,0.025,0.015),vec3f(0.05,1.0,0.55),sample.w);}
  if(mode==5u){color=albedo;}
  if(mode<=2u){color=aces(color*frame.controls.y);}
  return vec4f(color,1.0);
}
`;

// The paper's interval accumulation stabilizes world-space radiance. The final
// sparse-to-screen reconstruction still crosses probe/LOD footprints under
// camera motion, so production presentation adds a conservative temporal
// resolve. History is accepted only when the reprojected world position agrees;
// disocclusions and newly visible surfaces therefore use the current frame.
export const temporalShader = /* wgsl */`
struct TemporalFrameUniforms {
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
  previousViewProj: mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> frame: TemporalFrameUniforms;
@group(0) @binding(1) var currentComposite: texture_2d<f32>;
@group(0) @binding(2) var previousComposite: texture_2d<f32>;
@group(0) @binding(3) var currentWorld: texture_2d<f32>;
@group(0) @binding(4) var previousWorld: texture_2d<f32>;
@group(0) @binding(5) var currentNormal: texture_2d<f32>;
@group(0) @binding(6) var previousNormal: texture_2d<f32>;

@vertex fn temporalVS(@builtin(vertex_index) index:u32)->@builtin(position) vec4f{
  let uv=vec2f(f32((index<<1u)&2u),f32(index&2u));
  return vec4f(uv*2.0-1.0,0.0,1.0);
}

@fragment fn temporalFS(@builtin(position) position:vec4f)->@location(0) vec4f{
  let pixel=vec2i(position.xy);
  let size=vec2i(textureDimensions(currentComposite));
  let current=textureLoad(currentComposite,pixel,0);
  let world=textureLoad(currentWorld,pixel,0);
  let historyAvailable=fract(frame.controls.w)>0.0;
  if(!historyAvailable||world.w<0.5){return current;}

  let clip=frame.previousViewProj*vec4f(world.xyz,1.0);
  if(clip.w<=1e-6){return current;}
  let ndc=clip.xy/clip.w;
  let projected=(vec2f(ndc.x*0.5+0.5,0.5-ndc.y*0.5))*vec2f(size);
  let center=vec2i(round(projected-vec2f(0.5)));
  let normal=normalize(textureLoad(currentNormal,pixel,0).xyz);
  var bestPixel=vec2i(-1);
  var bestDistance=1e30;
  for(var y=-1;y<=1;y++){
    for(var x=-1;x<=1;x++){
      let candidatePixel=center+vec2i(x,y);
      if(any(candidatePixel<vec2i(0))||any(candidatePixel>=size)){continue;}
      let candidateWorld=textureLoad(previousWorld,candidatePixel,0);
      if(candidateWorld.w<0.5){continue;}
      let candidateNormal=normalize(textureLoad(previousNormal,candidatePixel,0).xyz);
      if(dot(normal,candidateNormal)<0.88){continue;}
      let delta=candidateWorld.xyz-world.xyz;
      let distance=dot(delta,delta);
      if(distance<bestDistance){
        bestDistance=distance;
        bestPixel=candidatePixel;
      }
    }
  }
  let tolerance=max(0.015,frame.envBaseSpacing.w*0.06);
  if(bestPixel.x<0||bestDistance>tolerance*tolerance){return current;}

  let previous=textureLoad(previousComposite,bestPixel,0).rgb;
  let features=u32(frame.cameraPos.w+0.5);
  let maximumWeight=select(0.97,0.88,(features&32u)!=0u);
  let weight=min(fract(frame.controls.w),maximumWeight);
  return vec4f(mix(current.rgb,previous,weight),1.0);
}

@group(0) @binding(7) var resolvedComposite: texture_2d<f32>;
@fragment fn presentFS(@builtin(position) position:vec4f)->@location(0) vec4f{
  return textureLoad(resolvedComposite,vec2i(position.xy),0);
}
`;
