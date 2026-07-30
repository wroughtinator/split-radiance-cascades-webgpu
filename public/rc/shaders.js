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

struct VertexIn {
  @location(0) position: vec3f,
  @location(1) normal: vec3f,
  @location(2) albedo: vec3f,
  @location(3) emissive: vec3f,
};
struct VertexOut {
  @builtin(position) clip: vec4f,
  @location(0) world: vec3f,
  @location(1) normal: vec3f,
  @location(2) albedo: vec3f,
  @location(3) emissive: vec3f,
};
@vertex fn gbufferVS(v: VertexIn) -> VertexOut {
  var o: VertexOut;
  o.clip = frame.viewProj * vec4f(v.position, 1.0);
  o.world = v.position;
  o.normal = normalize(v.normal);
  o.albedo = v.albedo;
  o.emissive = v.emissive;
  return o;
}
struct GBufferOut {
  @location(0) albedo: vec4f,
  @location(1) normal: vec4f,
  @location(2) world: vec4f,
};
@fragment fn gbufferFS(v: VertexOut) -> GBufferOut {
  var o: GBufferOut;
  o.albedo = vec4f(v.albedo, max(max(v.emissive.r, v.emissive.g), v.emissive.b));
  o.normal = vec4f(normalize(v.normal), 1.0);
  o.world = vec4f(v.world, 1.0);
  return o;
}
@vertex fn shadowVS(v: VertexIn) -> @builtin(position) vec4f {
  return frame.sunViewProj * vec4f(v.position, 1.0);
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
struct HashSlot { key: atomic<u32>, index: u32 };
struct PassParams { cascade: u32, value: u32, pad0: u32, pad1: u32 };
struct BvhNode { minMeta: vec4f, maxMeta: vec4f };
struct Triangle { a: vec4f, b: vec4f, c: vec4f, albedo: vec4f, emissive: vec4f };
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
  return u32(c.x) | (u32(c.y) << 9u) | (u32(c.z) << 18u) | ((lod & 7u) << 27u);
}

fn levelOfDetail(position: vec3f) -> u32 {
  let delta = abs(position - frame.cameraPos.xyz);
  let chebyshev = max(delta.x, max(delta.y, delta.z));
  let ratio = max(1.0, chebyshev / max(0.001, frame.envBaseSpacing.w * 18.0));
  return u32(clamp(floor(log2(ratio)), 0.0, 3.0));
}

fn cascadeSpacing(cascade: u32, lod: u32) -> f32 {
  return frame.envBaseSpacing.w * exp2(f32(cascade + lod));
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

fn lookupProbeFrame(cascade: u32, key: u32, frameIndex: u32) -> u32 {
  let base = frameIndex * HASH_FRAME_STRIDE + HASH_OFFSETS[cascade];
  let mask = HASH_SIZES[cascade] - 1u;
  let start = hash32(key) & mask;
  for (var step=0u; step<32u; step++) {
    let slot = base + ((start + step) & mask);
    let found = atomicLoad(&slots[slot].key);
    if (found == key) { return slots[slot].index; }
    if (found == EMPTY) { return EMPTY; }
  }
  return EMPTY;
}

fn lookupProbe(cascade: u32, key: u32) -> u32 {
  return lookupProbeFrame(cascade, key, currentFrame());
}

fn insertProbe(cascade: u32, position: vec3f, lod: u32) -> u32 {
  let cell = probeCell(position, cascade, lod);
  let key = keyFromCell(cell, lod);
  let base = currentFrame() * HASH_FRAME_STRIDE + HASH_OFFSETS[cascade];
  let mask = HASH_SIZES[cascade] - 1u;
  let start = hash32(key) & mask;
  for (var step=0u; step<32u; step++) {
    let slot = base + ((start + step) & mask);
    let result = atomicCompareExchangeWeak(&slots[slot].key, EMPTY, key);
    if (result.exchanged) {
      let index = atomicAdd(&state[cascade], 1u);
      if (index >= PROBE_CAPS[cascade]) {
        slots[slot].index = EMPTY;
        atomicAdd(&state[6], 1u);
        return EMPTY;
      }
      slots[slot].index = index;
      probeMeta[PROBE_OFFSETS[cascade] + index] = vec4f(probePositionFromCell(cell,cascade,lod), bitcast<f32>(lod));
      return index;
    }
    if (result.old_value == key) { return slots[slot].index; }
  }
  atomicAdd(&state[6], 1u);
  return EMPTY;
}

fn dataIndex(cascade: u32, probe: u32, direction: u32) -> u32 {
  return DATA_OFFSETS[cascade] + probe * DIR_COUNTS[cascade] + direction;
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

fn traceTriangle(origin: vec3f, direction: vec3f, tri: Triangle, maxDistance: f32) -> f32 {
  let e1=tri.b.xyz-tri.a.xyz;
  let e2=tri.c.xyz-tri.a.xyz;
  let p=cross(direction,e2);
  let det=dot(e1,p);
  if(abs(det)<1e-7){return maxDistance;}
  let inv=1.0/det;
  let tv=origin-tri.a.xyz;
  let u=dot(tv,p)*inv;
  if(u<0.0||u>1.0){return maxDistance;}
  let q=cross(tv,e1);
  let v=dot(direction,q)*inv;
  if(v<0.0||u+v>1.0){return maxDistance;}
  let t=dot(e2,q)*inv;
  if(t>0.001&&t<maxDistance){return t;}
  return maxDistance;
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
        let t=traceTriangle(origin,direction,tri,result.t);
        if(t<result.t){
          var n=normalize(cross(tri.b.xyz-tri.a.xyz,tri.c.xyz-tri.a.xyz));
          if(dot(n,direction)>0.0){n=-n;}
          result=Hit(t,n,tri.albedo.xyz,tri.emissive.xyz);
        }
      }
    } else if(stackSize<30u) {
      stack[stackSize]=left; stack[stackSize+1u]=right; stackSize+=2u;
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
  let point=max(0.0,dot(hit.normal,pointDirection))*pointAttenuation*frame.pointColorIntensity.xyz*frame.pointColorIntensity.w;
  let sun=frame.sunColorIntensity.xyz*frame.sunColorIntensity.w*ndl*visibility;
  return min(vec3f(16.0),hit.emissive.xyz+hit.albedo.xyz*(sun+point));
}
`;

export const computeShader = sharedCompute + /* wgsl */`
@compute @workgroup_size(256) fn resetSlots(@builtin(global_invocation_id) gid: vec3u) {
  if(gid.x>=12288u){return;}
  let slot=currentFrame()*HASH_FRAME_STRIDE+gid.x;
  atomicStore(&slots[slot].key,EMPTY);
  slots[slot].index=EMPTY;
}

@compute @workgroup_size(8,8) fn initBase(@builtin(global_invocation_id) gid: vec3u) {
  let giSize=vec2u(u32(frame.resolution.z),u32(frame.resolution.w));
  if(any(gid.xy>=giSize)){return;}
  let fullSize=vec2u(u32(frame.resolution.x),u32(frame.resolution.y));
  let pixel=min(fullSize-vec2u(1),gid.xy*fullSize/giSize);
  let world=textureLoad(worldTex,vec2i(pixel),0);
  if(world.w<0.5){return;}
  let lod=levelOfDetail(world.xyz);
  _=insertProbe(0u,world.xyz,lod);
}

@compute @workgroup_size(64) fn initHigher(@builtin(global_invocation_id) gid: vec3u) {
  let cascade=passParams.cascade;
  if(cascade==0u||cascade>3u){return;}
  let previous=cascade-1u;
  let activeCount=min(atomicLoad(&state[previous]),PROBE_CAPS[previous]);
  if(gid.x>=activeCount){return;}
  let probeInfo=probeMeta[PROBE_OFFSETS[previous]+gid.x];
  _=insertProbe(cascade,probeInfo.xyz,bitcast<u32>(probeInfo.w));
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

@compute @workgroup_size(8,8) fn splitRays(@builtin(global_invocation_id) gid: vec3u) {
  let giSize=vec2u(u32(frame.resolution.z),u32(frame.resolution.w));
  if(any(gid.xy>=giSize)||gid.z>=max(1u,passParams.value)){return;}
  let fullSize=vec2u(u32(frame.resolution.x),u32(frame.resolution.y));
  let pixel=min(fullSize-vec2u(1),gid.xy*fullSize/giSize);
  let world=textureLoad(worldTex,vec2i(pixel),0);
  if(world.w<0.5){return;}
  let normal=normalize(textureLoad(normalTex,vec2i(pixel),0).xyz);
  let lod=levelOfDetail(world.xyz);
  let stableScale=max(0.001,cascadeSpacing(0u,lod)/16.0);
  let stableCell=vec3i(floor(world.xyz/stableScale));
  let normalCell=vec3u(clamp(floor((normal*0.5+0.5)*15.0),vec3f(0),vec3f(15)));
  let normalKey=normalCell.x|(normalCell.y<<4u)|(normalCell.z<<8u);
  var stableKey=hash32(bitcast<u32>(stableCell.x)*0x9e3779b9u);
  stableKey=hash32(stableKey^(bitcast<u32>(stableCell.y)*0x85ebca6bu));
  stableKey=hash32(stableKey^(bitcast<u32>(stableCell.z)*0xc2b2ae35u));
  stableKey=hash32((stableKey^normalKey)^(gid.z*0x27d4eb2du));
  let sequence=f32(stableKey&0x00ffffffu);
  let g=1.324717957244746;
  let uv=fract(vec2f(0.5+sequence/g,0.5+sequence/(g*g)));
  var direction=decodeEqualArea(uv);
  if(dot(direction,normal)<0.0){direction=-direction;}
  direction=normalize(direction+normal*0.03);

  let baseLength=frame.envBaseSpacing.w*1.6;
  let maxDistance=baseLength*64.0;
  let origin=world.xyz+normal*max(0.008,frame.envBaseSpacing.w*0.012);
  let hit=traceScene(origin,direction,maxDistance+0.001);
  let didHit=hit.t<maxDistance;
  var targetCascade=3u;
  if(didHit){
    targetCascade=0u;
    var end=baseLength;
    loop {
      if(hit.t<=end||targetCascade>=3u){break;}
      targetCascade+=1u; end*=4.0;
    }
  }
  let radiance=select(frame.envBaseSpacing.xyz,directAtHit(origin+direction*hit.t,hit),didHit);
  atomicAdd(&state[4],1u);
  if(didHit){atomicAdd(&state[5],1u);}
  for(var cascade=0u;cascade<4u;cascade++){
    if(cascade>targetCascade){break;}
    let key=keyFromCell(probeCell(world.xyz,cascade,lod),lod);
    let probe=lookupProbe(cascade,key);
    if(probe==EMPTY){continue;}
    let dir=directionIndex(direction,cascade);
    if(cascade<targetCascade){deposit(cascade,probe,dir,vec3f(0),1.0);}
    else {deposit(cascade,probe,dir,radiance,0.0);}
  }
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
struct HashSlot { key: atomic<u32>, index: u32 };
@group(0) @binding(0) var<uniform> frame: FrameUniforms;
@group(0) @binding(1) var albedoTex: texture_2d<f32>;
@group(0) @binding(2) var normalTex: texture_2d<f32>;
@group(0) @binding(3) var worldTex: texture_2d<f32>;
@group(0) @binding(4) var shadowTex: texture_depth_2d;
@group(0) @binding(5) var shadowSampler: sampler_comparison;
@group(0) @binding(6) var<storage,read_write> slots: array<HashSlot>;
@group(0) @binding(7) var<storage,read> irradiance: array<vec4f>;

const EMPTY:u32=0xffffffffu;
const HASH_FRAME_STRIDE:u32=12288u;
const IRRADIANCE_FRAME_STRIDE:u32=73728u;
fn hash32(value:u32)->u32{
  var x=value;x=x^(x>>16u);x=x*0x7feb352du;x=x^(x>>15u);x=x*0x846ca68bu;return x^(x>>16u);
}
fn keyFromCell(cellIn:vec3i,lod:u32)->u32{
  let c=clamp(cellIn,vec3i(-256),vec3i(255))+vec3i(256);
  return u32(c.x)|(u32(c.y)<<9u)|(u32(c.z)<<18u)|((lod&7u)<<27u);
}
fn lookupProbe(key:u32)->u32{
  let start=hash32(key)&4095u;
  let frameIndex=min(1u,u32(floor(frame.controls.w)));
  let base=frameIndex*HASH_FRAME_STRIDE;
  for(var step=0u;step<32u;step++){
    let slot=base+((start+step)&4095u);
    let found=atomicLoad(&slots[slot].key);
    if(found==key){return slots[slot].index;}
    if(found==EMPTY){return EMPTY;}
  }
  return EMPTY;
}
fn levelOfDetail(position:vec3f)->u32{
  let delta=abs(position-frame.cameraPos.xyz);
  let distance=max(delta.x,max(delta.y,delta.z));
  let ratio=max(1.0,distance/max(0.001,frame.envBaseSpacing.w*18.0));
  return u32(clamp(floor(log2(ratio)),0.0,3.0));
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
fn sampleIrradiance(position:vec3f,normal:vec3f)->vec4f{
  let lod=levelOfDetail(position);
  let spacing=frame.envBaseSpacing.w*exp2(f32(lod));
  let grid=position/spacing-vec3f(0.5);
  let cell=vec3i(floor(grid));
  let fraction=fract(grid);
  let texel=octIndex(normal);
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
    let weight=wv.x*wv.y*wv.z;
    let probe=lookupProbe(keyFromCell(cell+bits,lod));
    if(probe!=EMPTY&&probe<2048u){value+=irradiance[frameIndex*IRRADIANCE_FRAME_STRIDE+probe*36u+texel].xyz*weight;total+=weight;}
  }
  if(total<1e-5){return vec4f(frame.envBaseSpacing.xyz*0.6,0.0);}
  return vec4f(value/total,clamp(total,0.0,1.0));
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
  let sample=sampleIrradiance(world.xyz,normal);
  let indirect=albedo*sample.xyz*frame.controls.x;
  let L=normalize(-frame.sunDirTime.xyz);
  let sun=albedo*frame.sunColorIntensity.xyz*frame.sunColorIntensity.w*max(0.0,dot(normal,L))*shadowVisibility(world.xyz,normal);
  let toPoint=frame.pointPosRange.xyz-world.xyz;
  let distance=length(toPoint);
  let pointWindow=max(0.0,1.0-distance/frame.pointPosRange.w);
  let point=albedo*frame.pointColorIntensity.xyz*frame.pointColorIntensity.w*max(0.0,dot(normal,toPoint/max(distance,1e-4)))*pointWindow*pointWindow/(1.0+0.06*distance*distance);
  let emissive=albedoData.www*vec3f(2.2,1.8,1.35);
  let direct=sun+point+emissive;
  let mode=u32(frame.controls.z+0.5);
  var color=direct+indirect;
  if(mode==1u){color=indirect;}
  if(mode==2u){color=direct;}
  if(mode==3u){color=normal*0.5+0.5;}
  if(mode==4u){color=mix(vec3f(0.35,0.025,0.015),vec3f(0.05,1.0,0.55),sample.w);}
  if(mode==5u){color=albedo;}
  if(mode<=2u){color=aces(color*frame.controls.y);}
  return vec4f(color,1.0);
}
`;
