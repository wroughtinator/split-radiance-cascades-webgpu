export const shaderConstants = {
  hashOffsets: [0, 4096, 8192, 10240],
  hashSizes: [4096, 4096, 2048, 2048],
  probeOffsets: [0, 2048, 3072, 3584],
  probeCaps: [2048, 1024, 512, 256],
  dataOffsets: [0, 65536, 196608, 458752],
  directions: [32, 128, 512, 2048],
  totalHashSlots: 12288,
  hashFrames: 2,
  totalProbeMeta: 3840,
  totalDirectionData: 983040,
  irradianceTexels: 36,
  irradianceFrames: 2,
  stateWords: 11536,
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
fn atlasUv(uv:vec2f,material:f32)->vec2f{
  let index=u32(max(0.0,material)+0.5);
  let cell=vec2u(index%5u,index/5u);
  return (vec2f(cell)*819.0+vec2f(4.0)+fract(uv)*811.0)/4096.0;
}
fn materialSample(uv:vec2f,material:f32)->vec4f{
  let sampled=textureSample(albedoAtlas,atlasSampler,atlasUv(uv,material));
  return select(sampled,vec4f(1),material<0.0);
}
@fragment fn gbufferFS(v: VertexOut, @builtin(front_facing) frontFacing: bool) -> GBufferOut {
  var o: GBufferOut;
  let surface=materialSample(v.uv,v.materialCutoff.x);
  if(v.materialCutoff.y>0.0&&surface.a<v.materialCutoff.y){discard;}
  o.albedo = vec4f(v.albedo*surface.rgb, max(max(v.emissive.r, v.emissive.g), v.emissive.b));
  o.normal = vec4f(normalize(select(-v.normal, v.normal, frontFacing)), 1.0);
  o.world = vec4f(v.world, 1.0);
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
struct PassParams { cascade: u32, value: u32, pad0: u32, pad1: u32 };
struct BvhNode { minMeta: vec4f, maxMeta: vec4f };
struct Triangle {
  a: vec4f, b: vec4f, c: vec4f, albedo: vec4f, emissive: vec4f,
  uvAB: vec4f, uvCMaterial: vec4f,
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

const EMPTY: u32 = 0xffffffffu;
const PI: f32 = 3.141592653589793;
const TAU: f32 = 6.283185307179586;
const FIXED_SCALE: f32 = 4096.0;
const HASH_OFFSETS = array<u32,4>(0u,4096u,8192u,10240u);
const HASH_SIZES = array<u32,4>(4096u,4096u,2048u,2048u);
const PROBE_OFFSETS = array<u32,4>(0u,2048u,3072u,3584u);
const PROBE_CAPS = array<u32,4>(2048u,1024u,512u,256u);
const DATA_OFFSETS = array<u32,4>(0u,65536u,196608u,458752u);
const DIR_COUNTS = array<u32,4>(32u,128u,512u,2048u);
const HASH_FRAME_STRIDE: u32 = 12288u;
const IRRADIANCE_FRAME_STRIDE: u32 = 73728u;
const RAY_COUNT_OFFSET: u32 = 16u;
const RAY_OFFSET_OFFSET: u32 = 3856u;
const RAY_CURSOR_OFFSET: u32 = 7696u;
const TOTAL_PROBE_META: u32 = 3840u;
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
  let c = clamp(cellIn, vec3i(-256), vec3i(255)) + vec3i(256);
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

fn insertProbeKey(cascade: u32, position: vec3f, lod: u32) {
  let cell = probeCell(position, cascade, lod);
  let key = keyFromCell(cell, lod);
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

fn dataIndex(cascade: u32, probe: u32, direction: u32) -> u32 {
  return DATA_OFFSETS[cascade] + probe * DIR_COUNTS[cascade] + direction;
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

fn directionIndex(direction: vec3f, cascade: u32) -> u32 {
  let theta = 4u << cascade;
  let width = theta * 2u;
  let uFloat = fract(atan2(direction.y, direction.x) / TAU + 1.0);
  let vFloat = clamp(direction.z * 0.5 + 0.5, 0.0, 0.999999);
  return min(width-1u,u32(floor(uFloat*f32(width)))) + min(theta-1u,u32(floor(vFloat*f32(theta))))*width;
}

fn directionFromIndex(index: u32, cascade: u32) -> vec3f {
  let theta = 4u << cascade;
  let width = theta * 2u;
  let u = index % width;
  let v = index / width;
  return decodeEqualArea(vec2f((f32(u)+0.5)/f32(width),(f32(v)+0.5)/f32(theta)));
}

fn rayBox(origin: vec3f, inverseDirection: vec3f, minB: vec3f, maxB: vec3f, maxDistance: f32) -> bool {
  let t0 = (minB-origin)*inverseDirection;
  let t1 = (maxB-origin)*inverseDirection;
  let near3 = min(t0,t1);
  let far3 = max(t0,t1);
  let nearT = max(max(near3.x,near3.y),max(near3.z,0.0));
  let farT = min(min(far3.x,far3.y),far3.z);
  return farT >= nearT && nearT < maxDistance;
}

fn sampleAtlasNearest(uv:vec2f,material:f32)->vec4f{
  if(material<0.0){return vec4f(1);}
  let index=u32(material+0.5);
  let cell=vec2u(index%5u,index/5u);
  let atlasUv=(vec2f(cell)*819.0+vec2f(4.0)+fract(uv)*811.0)/4096.0;
  let dimensions=vec2f(textureDimensions(albedoAtlas));
  let texel=vec2i(clamp(floor(atlasUv*dimensions),vec2f(0),dimensions-vec2f(1)));
  return textureLoad(albedoAtlas,texel,0);
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
  let inverseDirection=1.0/max(abs(direction),vec3f(1e-8))*sign(direction);
  var result=Hit(maxDistance,vec3f(0,1,0),vec3f(0),vec3f(0));
  var stack: array<u32,32>;
  var stackSize=1u;
  stack[0]=0u;
  for(var iteration=0u;iteration<512u&&stackSize>0u;iteration++){
    stackSize-=1u;
    let nodeIndex=stack[stackSize];
    let node=bvhNodes[nodeIndex];
    if(!rayBox(origin,inverseDirection,node.minMeta.xyz,node.maxMeta.xyz,result.t)){continue;}
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
          let surface=sampleAtlasNearest(uv,tri.uvCMaterial.z);
          if(tri.uvCMaterial.w>0.0&&surface.a<tri.uvCMaterial.w){continue;}
          var n=normalize(cross(tri.b.xyz-tri.a.xyz,tri.c.xyz-tri.a.xyz));
          if(dot(n,direction)>0.0){n=-n;}
          result=Hit(intersection.x,n,tri.albedo.xyz*surface.rgb,tri.emissive.xyz);
        }
      }
    } else if(stackSize<30u) {
      stack[stackSize]=left; stack[stackSize+1u]=right; stackSize+=2u;
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
  let xy=vec2u(floor(uv*6.0));
  return xy.x+xy.y*6u;
}

fn samplePrimaryIrradianceLod(position:vec3f,normal:vec3f,lod:u32)->vec4f{
  let spacing=cascadeSpacing(0u,lod);
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
  let octCoordinate=clamp((oct*0.5+0.5)*6.0-vec2f(0.5),vec2f(0),vec2f(5));
  let octBase=vec2i(floor(octCoordinate));
  let octFraction=fract(octCoordinate);
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
      for(var tap=0u;tap<4u;tap++){
        let offset=vec2i(i32(tap&1u),i32((tap>>1u)&1u));
        let coordinate=clamp(octBase+offset,vec2i(0),vec2i(5));
        let angularWeight=select(1.0-octFraction.x,octFraction.x,offset.x==1)
          *select(1.0-octFraction.y,octFraction.y,offset.y==1);
        let texel=u32(coordinate.x+coordinate.y*6);
        let weight=spatialWeight*angularWeight;
        value+=irradiance[currentFrame()*IRRADIANCE_FRAME_STRIDE+probe*36u+texel].xyz*weight;
        total+=weight;
      }
    }
  }
  if(total<1e-5){return vec4f(frame.envBaseSpacing.xyz*0.6,0.0);}
  return vec4f(value/total,clamp(total,0.0,1.0));
}

fn samplePrimaryIrradiance(position:vec3f,normal:vec3f)->vec4f{
  let lods=lodSelection(position);
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
  let texel=octTexel(normal);
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
      value+=irradiance[previousFrame*IRRADIANCE_FRAME_STRIDE+probe*36u+texel].xyz*weight;
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
  if(gid.x>=12288u){return;}
  let slot=currentFrame()*HASH_FRAME_STRIDE+gid.x;
  atomicStore(&slots[slot].key,EMPTY);
  atomicStore(&slots[slot].index,EMPTY);
}

@compute @workgroup_size(8,8) fn initBase(@builtin(global_invocation_id) gid: vec3u) {
  let giSize=vec2u(u32(frame.resolution.z),u32(frame.resolution.w));
  if(any(gid.xy>=giSize)){return;}
  let fullSize=vec2u(u32(frame.resolution.x),u32(frame.resolution.y));
  let pixel=min(fullSize-vec2u(1),gid.xy*fullSize/giSize);
  let world=textureLoad(worldTex,vec2i(pixel),0);
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

// Key insertion and compact-index assignment are separate passes. Ranking
// occupied hash slots in slot order makes key -> probe-index publication
// independent of GPU workgroup scheduling, eliminating a subtle frame race.
@compute @workgroup_size(64) fn canonicalizeProbes(@builtin(global_invocation_id) gid:vec3u){
  let cascade=passParams.cascade;
  if(gid.x>=HASH_SIZES[cascade]){return;}
  let base=currentFrame()*HASH_FRAME_STRIDE+HASH_OFFSETS[cascade];
  let slot=base+gid.x;
  let key=atomicLoad(&slots[slot].key);
  if(key==EMPTY){return;}
  var rank=0u;
  for(var previous=0u;previous<gid.x;previous++){
    if(atomicLoad(&slots[base+previous].key)!=EMPTY){rank+=1u;}
  }
  if(rank>=PROBE_CAPS[cascade]){
    atomicStore(&slots[slot].index,EMPTY);
    atomicAdd(&state[6],1u);
    return;
  }
  atomicStore(&slots[slot].index,rank);
  let lod=lodFromKey(key);
  probeMeta[PROBE_OFFSETS[cascade]+rank]=vec4f(
    probePositionFromCell(cellFromKey(key),cascade,lod),
    bitcast<f32>(lod)
  );
  atomicMax(&state[cascade],rank+1u);
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
    for(var other=0u;other<activeCount;other++){
      let otherInfo=probeMeta[PROBE_OFFSETS[cascade]+other];
      if(parentKeyFromInfo(otherInfo,cascade)==parentKey&&probeKeyFromInfo(otherInfo,cascade)<key){
        prefix+=atomicLoad(&state[probeStateIndex(RAY_COUNT_OFFSET,cascade,other)]);
      }
    }
  }
  atomicStore(&state[probeStateIndex(RAY_OFFSET_OFFSET,cascade,gid.x)],prefix);
}

fn deposit(cascade:u32,probe:u32,direction:u32,radiance:vec3f,beta:f32){
  let base=dataIndex(cascade,probe,direction)*5u;
  let safe=min(max(radiance,vec3f(0)),vec3f(16));
  atomicAdd(&accum[base],u32(safe.r*FIXED_SCALE+0.5));
  atomicAdd(&accum[base+1u],u32(safe.g*FIXED_SCALE+0.5));
  atomicAdd(&accum[base+2u],u32(safe.b*FIXED_SCALE+0.5));
  atomicAdd(&accum[base+3u],u32(clamp(beta,0.0,1.0)*FIXED_SCALE+0.5));
  atomicAdd(&accum[base+4u],1u);
}

fn traceAndSplit(world:vec3f,normal:vec3f,lod:u32,lane:u32,sequenceSeed:u32,record:u32,writeHit:bool){
  let key0=keyFromCell(probeCell(world,0u,lod),lod);
  let baseProbe=lookupProbe(0u,key0);
  if(baseProbe==EMPTY){
    if(writeHit){probeMeta[record]=vec4f(0);probeMeta[record+1u]=vec4f(0);}
    return;
  }
  // GPU atomic arrival order is deliberately excluded from direction
  // assignment. The stable screen-sample seed makes the same camera path
  // bit-reproducible while Algorithm 3's prefix still decorrelates probes.
  let sequenceIndex=atomicLoad(&state[probeStateIndex(RAY_OFFSET_OFFSET,0u,baseProbe)])+sequenceSeed;
  let g=1.324717957244746;
  let alpha=vec2f(1.0/g,1.0/(g*g));
  var jitter=vec2f(0);
  if(!featureEnabled(8u)){
    let temporal=hash32(u32(frame.sunDirTime.w*60.0)+lane*0x9e3779b9u);
    jitter=vec2f(f32(temporal&65535u),f32(temporal>>16u))/65536.0;
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
  let sequenceSeed=(gid.y*giSize.x+gid.x)*max(1u,passParams.value)+gid.z;
  traceAndSplit(world.xyz,normal,fine,gid.z,sequenceSeed,record,true);
  let coarse=u32(lods.y);
  if(coarse!=fine){traceAndSplit(world.xyz,normal,coarse,gid.z,sequenceSeed,0u,false);}
}

@compute @workgroup_size(8,8) fn splitSecondaryRays(@builtin(global_invocation_id) gid: vec3u) {
  let giSize=vec2u(u32(frame.resolution.z),u32(frame.resolution.w));
  if(!featureEnabled(1u)||historyWeight()<=0.0||any(gid.xy>=giSize)||gid.z>=max(1u,passParams.value)){return;}
  let record=hitRecordIndex(gid,1u-currentFrame());
  let hitPosition=probeMeta[record];
  let hitNormal=probeMeta[record+1u];
  if(hitPosition.w<0.5||hitNormal.w<0.5){return;}
  let lod=min(levelOfDetail(hitPosition.xyz)+2u,7u)|SECONDARY_TAG;
  let sequenceSeed=(gid.y*giSize.x+gid.x)*max(1u,passParams.value)+gid.z;
  traceAndSplit(hitPosition.xyz,normalize(hitNormal.xyz),lod,gid.z,sequenceSeed,0u,false);
}

fn sampleParentDirection(cascade:u32,position:vec3f,lod:u32,parentDirection:u32)->vec3f{
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
      value+=cones[dataIndex(parent,probe,parentDirection)].xyz*weight;
      total+=weight;
    }
  }
  if(total<1e-5){return frame.envBaseSpacing.xyz;}
  return value/total;
}

fn mergedParent(cascade:u32,direction:u32,position:vec3f,lod:u32)->vec3f{
  let theta=4u<<cascade;
  let width=theta*2u;
  let u=direction%width;
  let v=direction/width;
  let parentWidth=width*2u;
  var sum=vec3f(0);
  for(var child=0u;child<4u;child++){
    let du=child&1u;
    let dv=(child>>1u)&1u;
    let parentDirection=(v*2u+dv)*parentWidth+(u*2u+du);
    sum+=sampleParentDirection(cascade,position,lod,parentDirection);
  }
  return sum*0.25;
}

@compute @workgroup_size(64) fn mergeCascade(@builtin(global_invocation_id) gid: vec3u) {
  let cascade=passParams.cascade;
  let count=DIR_COUNTS[cascade];
  let probe=gid.x/count;
  let direction=gid.x-probe*count;
  let activeCount=min(atomicLoad(&state[cascade]),PROBE_CAPS[cascade]);
  if(probe>=activeCount){return;}
  let index=dataIndex(cascade,probe,direction);
  let base=index*5u;
  let samples=atomicLoad(&accum[base+4u]);
  var interval=vec3f(0);
  var beta=1.0;
  if(samples>0u){
    let denominator=FIXED_SCALE*f32(samples);
    interval=vec3f(f32(atomicLoad(&accum[base])),f32(atomicLoad(&accum[base+1u])),f32(atomicLoad(&accum[base+2u])))/denominator;
    beta=f32(atomicLoad(&accum[base+3u]))/denominator;
  }
  let probeInfo=probeMeta[PROBE_OFFSETS[cascade]+probe];
  var distant=frame.envBaseSpacing.xyz;
  if(cascade<3u){distant=mergedParent(cascade,direction,probeInfo.xyz,bitcast<u32>(probeInfo.w));}
  cones[index]=vec4f(min(vec3f(16.0),interval+clamp(beta,0.0,1.0)*distant),1.0);
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
  let probe=gid.x/36u;
  let texel=gid.x-probe*36u;
  let activeCount=min(atomicLoad(&state[0]),PROBE_CAPS[0]);
  if(probe>=activeCount){return;}
  let x=texel%6u;
  let y=texel/6u;
  let normal=decodeOctahedral(vec2f((f32(x)+0.5)/6.0,(f32(y)+0.5)/6.0));
  var result=vec3f(0);
  for(var direction=0u;direction<32u;direction++){
    let ray=directionFromIndex(direction,0u);
    result+=cones[dataIndex(0u,probe,direction)].xyz*max(0.0,dot(normal,ray));
  }
  var filtered=result*(4.0/32.0);
  let blend=historyWeight();
  if(blend>0.0){
    let probeInfo=probeMeta[PROBE_OFFSETS[0]+probe];
    let lod=bitcast<u32>(probeInfo.w);
    let key=keyFromCell(probeCell(probeInfo.xyz,0u,lod),lod);
    let previousFrame=1u-currentFrame();
    let previousProbe=lookupProbeFrame(0u,key,previousFrame);
    if(previousProbe!=EMPTY&&previousProbe<PROBE_CAPS[0]){
      let history=irradiance[previousFrame*IRRADIANCE_FRAME_STRIDE+previousProbe*36u+texel].xyz;
      filtered=mix(filtered,history,blend);
    }
  }
  irradiance[currentFrame()*IRRADIANCE_FRAME_STRIDE+probe*36u+texel]=vec4f(filtered,1.0);
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
@group(0) @binding(7) var<storage,read> irradiance: array<vec4f>;
@group(0) @binding(8) var<storage,read> cones: array<vec4f>;
@group(0) @binding(9) var<storage,read_write> accum: array<atomic<u32>>;

const EMPTY:u32=0xffffffffu;
const HASH_FRAME_STRIDE:u32=12288u;
const IRRADIANCE_FRAME_STRIDE:u32=73728u;
const FIXED_SCALE:f32=4096.0;
const HASH_OFFSETS=array<u32,4>(0u,4096u,8192u,10240u);
const HASH_SIZES=array<u32,4>(4096u,4096u,2048u,2048u);
const DATA_OFFSETS=array<u32,4>(0u,65536u,196608u,458752u);
const DIR_COUNTS=array<u32,4>(32u,128u,512u,2048u);
fn hash32(value:u32)->u32{
  var x=value;x=x^(x>>16u);x=x*0x7feb352du;x=x^(x>>15u);x=x*0x846ca68bu;return x^(x>>16u);
}
fn keyFromCell(cellIn:vec3i,lod:u32)->u32{
  let c=clamp(cellIn,vec3i(-256),vec3i(255))+vec3i(256);
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
  let octCoordinate=clamp((oct*0.5+0.5)*6.0-vec2f(0.5),vec2f(0),vec2f(5));
  let octBase=vec2i(floor(octCoordinate));
  let octFraction=fract(octCoordinate);
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
    if(probe!=EMPTY&&probe<2048u){
      for(var tap=0u;tap<4u;tap++){
        let offset=vec2i(i32(tap&1u),i32((tap>>1u)&1u));
        let coordinate=clamp(octBase+offset,vec2i(0),vec2i(5));
        let angularWeight=select(1.0-octFraction.x,octFraction.x,offset.x==1)
          *select(1.0-octFraction.y,octFraction.y,offset.y==1);
        let texel=u32(coordinate.x+coordinate.y*6);
        let weight=spatialWeight*angularWeight;
        value+=irradiance[frameIndex*IRRADIANCE_FRAME_STRIDE+probe*36u+texel].xyz*weight;
        total+=weight;
      }
    }
  }
  if(total<1e-5){return vec4f(frame.envBaseSpacing.xyz*0.6,0.0);}
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
fn directionIndex(direction:vec3f,cascade:u32)->u32{
  let theta=4u<<cascade;
  let width=theta*2u;
  let uFloat=fract(atan2(direction.y,direction.x)/6.283185307179586+1.0);
  let vFloat=clamp(direction.z*0.5+0.5,0.0,0.999999);
  return min(width-1u,u32(floor(uFloat*f32(width))))+min(theta-1u,u32(floor(vFloat*f32(theta))))*width;
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
  return decodeEqualArea(vec2f(
    (f32(index%width)+0.5)/f32(width),
    (f32(index/width)+0.5)/f32(theta)
  ));
}
fn dataIndex(cascade:u32,probe:u32,direction:u32)->u32{
  return DATA_OFFSETS[cascade]+probe*DIR_COUNTS[cascade]+direction;
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
      value+=cones[dataIndex(cascade,probe,directionId)].xyz*weight;
      total+=weight;
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
      let base=dataIndex(cascade,probe,directionId)*5u;
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
    /(1.0+0.06*distance*distance);
  return sun+point+albedoData.www*vec3f(2.2,1.8,1.35);
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
  let point=albedo*frame.pointColorIntensity.xyz*frame.pointColorIntensity.w*max(0.0,dot(normal,toPoint/max(distance,1e-4)))*pointWindow*pointWindow/(1.0+0.06*distance*distance);
  let emissive=albedoData.www*vec3f(2.2,1.8,1.35);
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
