export const shaderConstants = {
  hashOffsets: [0, 32768, 40960, 43008],
  hashSizes: [32768, 8192, 2048, 1024],
  probeOffsets: [0, 16384, 20480, 21504],
  probeCaps: [16384, 4096, 1024, 512],
  dataOffsets: [0, 524288, 1048576, 1572864],
  directions: [32, 128, 512, 2048],
  totalHashSlots: 44032,
  hashFrames: 4,
  totalProbeMeta: 22016,
  totalDirectionData: 2621440,
  irradianceTexels: 64,
  irradianceAtlasWidth: 512,
  irradianceAtlasFrameHeight: 2048,
  irradianceFrames: 4,
  accumFrames: 2,
  persistentHashSlots: 32768,
  persistentMetaWords: 131072,
  persistentMapOffset: 5373952,
  persistentWords: 5390336,
  // Base counters/prefixes plus one deterministic winner per c0 probe and
  // each of the 64 fixed current-state lanes used by dynamic transport.
  stateWords: 2163216,
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
  sceneBounds: vec4f,
  dynamicInfo: vec4u,
  previousViewProj: mat4x4<f32>,
  lodCamera: vec4f,
};
@group(0) @binding(0) var<uniform> frame: FrameUniforms;
@group(0) @binding(1) var albedoAtlas: texture_2d_array<f32>;
@group(0) @binding(2) var atlasSampler: sampler;
@group(0) @binding(3) var<uniform> lightViewProj: mat4x4<f32>;
struct RasterDynamicRecord {
  a:vec4f, b:vec4f, c:vec4f, albedo:vec4f, emissive:vec4f,
  uvAB:vec4f, uvCMaterial:vec4f, normalOct:vec4u,
};
@group(0) @binding(4) var<storage,read> rasterDynamicData:array<RasterDynamicRecord>;

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
  // 0xffffffff is immutable scene geometry; dynamic records use their stable
  // instance slot.  The fragment packs this into world.w so every later pass
  // can distinguish overlapping rigid receivers without another attachment.
  @location(6) @interpolate(flat) dynamicOwner: u32,
  @location(7) @interpolate(flat) rasterPrimitive: u32,
};
@vertex fn gbufferVS(
  v: VertexIn,@builtin(vertex_index) vertexIndex:u32
) -> VertexOut {
  var o: VertexOut;
  o.clip = frame.viewProj * vec4f(v.position, 1.0);
  o.world = v.position;
  o.normal = normalize(v.normal);
  o.albedo = v.albedo;
  o.emissive = v.emissive;
  o.uv = v.uv;
  o.materialCutoff = v.materialCutoff;
  o.dynamicOwner = 0xffffffffu;
  o.rasterPrimitive = vertexIndex/3u;
  return o;
}
fn rasterQuaternionRotate(vector:vec3f,quaternion:vec4f)->vec3f{
  let doubled=2.0*cross(quaternion.xyz,vector);
  return vector+quaternion.w*doubled+cross(quaternion.xyz,doubled);
}
@vertex fn dynamicGbufferVS(
  v:VertexIn,@builtin(vertex_index) vertexIndex:u32
)->VertexOut{
  let instanceIndex=u32(v.materialCutoff.x+0.5);
  let instance=rasterDynamicData[frame.dynamicInfo.y+instanceIndex];
  let world=instance.a.xyz+rasterQuaternionRotate(
    v.position*instance.c.xyz,instance.b
  );
  let normal=normalize(rasterQuaternionRotate(
    v.normal/instance.c.xyz,instance.b
  ));
  var o:VertexOut;
  o.clip=frame.viewProj*vec4f(world,1);
  o.world=world;
  o.normal=normal;
  o.albedo=v.albedo*instance.albedo.xyz;
  o.emissive=instance.emissive.xyz;
  o.uv=v.uv;
  o.materialCutoff=vec2f(-1.0,v.materialCutoff.y);
  o.dynamicOwner=instanceIndex;
  o.rasterPrimitive=vertexIndex/3u;
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
@fragment fn gbufferFS(
  v: VertexOut,
  @builtin(front_facing) frontFacing: bool
) -> GBufferOut {
  var o: GBufferOut;
  let surface=materialSample(v.uv,v.materialCutoff.x);
  if(v.materialCutoff.y>0.0&&surface.a<v.materialCutoff.y){discard;}
  // Procedural and imported area emitters are oriented surfaces. Treating
  // their back face as luminous lets a recessed ceiling panel illuminate the
  // ceiling above it and creates the conspicuous square "halo" that is not
  // present in the reference scene. Closed emissive meshes still radiate in
  // every outward direction because each outward-facing facet remains lit.
  // Use the authored surface normal for radiometric sidedness. The raster
  // front_facing convention can be inverted by the view/projection handedness
  // on a backend even though the physical source orientation is unchanged.
  let sourceVisible=dot(normalize(v.normal),frame.cameraPos.xyz-v.world)>0.0;
  let visibleEmission=select(vec3f(0),v.emissive,sourceVisible);
  let emission=encodeSurfaceEmission(visibleEmission);
  o.albedo = vec4f(v.albedo*surface.rgb, emission.r);
  o.normal = vec4f(
    encodeNormalOct(select(-v.normal,v.normal,frontFacing)),emission.gb
  );
  // Closed-topology metadata comes from the mesh, not the current view. Only
  // a back face belonging to a declared closed volume receives the interior
  // visibility bit. Open and two-sided surfaces therefore render identically
  // from either side. The independent 1.0 emitter bit preserves packed RGB.
  let closedBackFace=!frontFacing&&v.materialCutoff.y< -0.5;
  let flags=select(0u,1u,closedBackFace)
    |select(0u,2u,any(visibleEmission>vec3f(0)));
  // Preserve exact primitive/topology identity without another MRT. Static
  // primitive IDs occupy the low 23 bits. Dynamic IDs set bit 23 and pack a
  // six-bit owner plus the draw's stable 12-bit primitive index. The complete
  // code remains below 2^24 and is therefore represented exactly by f32.
  // Bit 2 is the static-surface validity sentinel. Using +1 as the sentinel
  // collided with bit 0: a closed back face added another 1, carried into the
  // emitter bit, and lost its closed-volume classification.
  var packedSurface=4u+(v.rasterPrimitive<<2u)+flags;
  if(v.dynamicOwner!=0xffffffffu){
    packedSurface=0x800000u
      |((v.dynamicOwner&63u)<<14u)
      |((v.rasterPrimitive&4095u)<<2u)
      |flags;
  }
  o.world = vec4f(v.world,f32(packedSurface));
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
@vertex fn dynamicShadowVS(v:VertexIn)->ShadowOut{
  let instanceIndex=u32(v.materialCutoff.x+0.5);
  let instance=rasterDynamicData[frame.dynamicInfo.y+instanceIndex];
  let world=instance.a.xyz+rasterQuaternionRotate(
    v.position*instance.c.xyz,instance.b
  );
  var o:ShadowOut;
  o.clip=lightViewProj*vec4f(world,1);
  o.uv=v.uv;
  o.materialCutoff=vec2f(-1.0,v.materialCutoff.y);
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
@vertex fn dynamicPointShadowVS(v:VertexIn)->ShadowOut{
  let instanceIndex=u32(v.materialCutoff.x+0.5);
  let instance=rasterDynamicData[frame.dynamicInfo.y+instanceIndex];
  let world=instance.a.xyz+rasterQuaternionRotate(
    v.position*instance.c.xyz,instance.b
  );
  var o:ShadowOut;
  o.clip=lightViewProj*vec4f(world,1);
  o.uv=v.uv;
  o.materialCutoff=vec2f(-1.0,v.materialCutoff.y);
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
  sceneBounds: vec4f,
  dynamicInfo: vec4u,
  previousViewProj: mat4x4<f32>,
  lodCamera: vec4f,
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
struct Hit {
  t: f32, normal: vec3f, albedo: vec3f, emissive: vec3f,
  sourceMinimum: vec3f, sourceMaximum: vec3f, triangleIndex: u32,
};

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
@group(0) @binding(22) var<storage,read_write> persistentIrradiance:array<atomic<u32>>;
@group(0) @binding(23) var dynamicReceiverAuditTex:texture_2d<f32>;

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
const HAZARD_WINNER_OFFSET: u32 = 66064u;
const HAZARD_WINNER_LANES: u32 = 64u;
const HAZARD_SECOND_OFFSET: u32 = 1114640u;
const BLOCK_COUNT_OFFSET: u32 = 2163216u;
const TOTAL_PROBE_META: u32 = 22016u;
const PERSISTENT_BUCKETS:u32=8192u;
const PERSISTENT_WAYS:u32=4u;
const PERSISTENT_HASH_SLOTS:u32=32768u;
const PERSISTENT_META_WORDS:u32=4u;
const PERSISTENT_DATA_OFFSET:u32=131072u;
const PERSISTENT_DIRECTION_WORDS:u32=5u;
const PERSISTENT_MAP_OFFSET:u32=5373952u;

fn clearPersistentDirections(slot:u32){
  for(var direction=0u;direction<32u;direction++){
    let base=PERSISTENT_DATA_OFFSET
      +(slot*32u+direction)*PERSISTENT_DIRECTION_WORDS;
    atomicStore(&persistentIrradiance[base+4u],0u);
  }
}

fn initializePersistentSlot(slot:u32,generation:u32){
  let address=slot*PERSISTENT_META_WORDS;
  atomicStore(&persistentIrradiance[address+1u],generation);
  atomicStore(&persistentIrradiance[address+2u],passParams.sampleEpoch);
  clearPersistentDirections(slot);
}

fn persistentCandidateSlot(firstBucket:u32,secondBucket:u32,index:u32)->u32{
  let bucket=select(firstBucket,secondBucket,index>=PERSISTENT_WAYS);
  return bucket*PERSISTENT_WAYS+(index&(PERSISTENT_WAYS-1u));
}

fn resolvePersistentProbeSlot(key:u32)->u32{
  let bucketMask=PERSISTENT_BUCKETS-1u;
  let firstBucket=hash32(key)&bucketMask;
  var secondBucket=hash32(key^0x9e3779b9u)&bucketMask;
  if(secondBucket==firstBucket){secondBucket=(secondBucket+1u)&bucketMask;}
  let generation=passParams.sampleFrame+1u;
  // Two independent four-way buckets give bounded lookup cost. Claim only
  // after scanning every candidate for an existing match; one canonical c0
  // probe therefore owns at most one persistent slot.
  for(var attempt=0u;attempt<4u;attempt++){
    var emptySlot=EMPTY;
    var victimSlot=EMPTY;
    var victimKey=EMPTY;
    var victimAge=0u;
    for(var candidate=0u;candidate<PERSISTENT_WAYS*2u;candidate++){
      let slot=persistentCandidateSlot(firstBucket,secondBucket,candidate);
      let keyAddress=slot*PERSISTENT_META_WORDS;
      let storedKey=atomicLoad(&persistentIrradiance[keyAddress]);
      if(storedKey==key){
        let storedEpoch=atomicLoad(&persistentIrradiance[keyAddress+2u]);
        if(storedEpoch!=passParams.sampleEpoch){
          initializePersistentSlot(slot,generation);
        }else{
          atomicStore(&persistentIrradiance[keyAddress+1u],generation);
        }
        return slot;
      }
      if(storedKey==EMPTY){
        if(emptySlot==EMPTY){emptySlot=slot;}
      }else if(lookupProbe(0u,storedKey)==EMPTY){
        let lastSeen=atomicLoad(&persistentIrradiance[keyAddress+1u]);
        let age=generation-lastSeen;
        if(victimSlot==EMPTY||age>victimAge){
          victimSlot=slot;
          victimKey=storedKey;
          victimAge=age;
        }
      }
    }
    if(emptySlot!=EMPTY){
      let keyAddress=emptySlot*PERSISTENT_META_WORDS;
      for(var retry=0u;retry<8u;retry++){
        let claim=atomicCompareExchangeWeak(
          &persistentIrradiance[keyAddress],EMPTY,key
        );
        if(claim.exchanged){
          initializePersistentSlot(emptySlot,generation);
          return emptySlot;
        }
        if(claim.old_value==key){
          let storedEpoch=atomicLoad(&persistentIrradiance[keyAddress+2u]);
          if(storedEpoch!=passParams.sampleEpoch){
            initializePersistentSlot(emptySlot,generation);
          }else{
            atomicStore(&persistentIrradiance[keyAddress+1u],generation);
          }
          return emptySlot;
        }
        if(claim.old_value!=EMPTY){break;}
      }
      let afterRetry=atomicLoad(&persistentIrradiance[keyAddress]);
      // Do not choose from stale metadata after another resolver changed the
      // candidate. A still-empty weak-CAS failure is a safe cache miss.
      if(afterRetry==EMPTY){
        atomicAdd(&state[9],1u);
        return EMPTY;
      }
      continue;
    }
    if(victimSlot!=EMPTY){
      let victimAddress=victimSlot*PERSISTENT_META_WORDS;
      for(var retry=0u;retry<8u;retry++){
        let claim=atomicCompareExchangeWeak(
          &persistentIrradiance[victimAddress],victimKey,key
        );
        if(claim.exchanged){
          initializePersistentSlot(victimSlot,generation);
          return victimSlot;
        }
        if(claim.old_value==key){
          let storedEpoch=atomicLoad(&persistentIrradiance[victimAddress+2u]);
          if(storedEpoch!=passParams.sampleEpoch){
            initializePersistentSlot(victimSlot,generation);
          }else{
            atomicStore(&persistentIrradiance[victimAddress+1u],generation);
          }
          return victimSlot;
        }
        if(claim.old_value!=victimKey){break;}
      }
    }
  }
  // Contention is diagnostic only; the ordinary paper path remains valid.
  atomicAdd(&state[9],1u);
  return EMPTY;
}

fn persistentDirectionBase(slot:u32,direction:u32)->u32{
  return PERSISTENT_DATA_OFFSET
    +(slot*32u+direction)*PERSISTENT_DIRECTION_WORDS;
}

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

// Sheet-class override for the rigid pixel the current invocation serves.
// A tumbling instance's world normal crosses dominant-axis sheet boundaries
// continuously, and every crossing swapped the pixel's entire interpolation
// support to a different sheet's probes — the largest single source of
// per-frame lighting jumps on moving receivers. Keying rigid pixels by the
// owner-CANONICAL normal makes the sheet a fixed material property: rotation
// can no longer change which probes describe a given face, while world cells
// still provide the spatial identity. 8 means "no override" (static pixel).
var<private> rigidSheetOverride:u32=8u;

fn pixelSurfaceClass(normalIn:vec3f)->u32{
  return select(
    surfaceClass(normalIn),rigidSheetOverride,rigidSheetOverride!=8u
  );
}

fn setRigidSheetOverride(world:vec4f,normal:vec3f){
  rigidSheetOverride=8u;
  let owner=packedDynamicOwner(world.w);
  if(owner==EMPTY){return;}
  let instance=triangles[frame.dynamicInfo.y+owner];
  let canonicalNormal=normalize(
    quaternionRotate(normal,vec4f(-instance.b.xyz,instance.b.w))*instance.c.xyz
  );
  rigidSheetOverride=surfaceClass(canonicalNormal);
}

fn lodDistance(position: vec3f) -> f32 {
  let delta = abs(position - frame.lodCamera.xyz);
  return max(delta.x, max(delta.y, delta.z));
}

fn levelOfDetail(position: vec3f) -> u32 {
  let ratio = max(1.0, lodDistance(position) / max(0.001, frame.envBaseSpacing.w * 18.0));
  return u32(clamp(floor(log2(ratio)), 0.0, 3.0));
}

// Keep neighboring sparse LODs resident across a deliberately wider universal
// overlap, then blend linearly. x=fine LOD, y=coarse LOD, z=blend.
fn lodSelection(position: vec3f) -> vec3f {
  let fine = levelOfDetail(position);
  if (fine >= 3u) { return vec3f(f32(fine), f32(fine), 0.0); }
  let baseRange = max(0.001, frame.envBaseSpacing.w * 18.0);
  let boundary = baseRange * exp2(f32(fine + 1u));
  // A 25% residency overlap survives ordinary wheel impulses while preserving
  // the same two-LOD bounded workload. The former 10% band could be skipped
  // in one event, leaving both selected sparse fields cold.
  let overlapStart = boundary * 0.75;
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
  return u32(floor(frame.controls.w))&3u;
}

fn historyWeight() -> f32 {
  return fract(frame.controls.w);
}

fn featureEnabled(bit:u32)->bool {
  return (u32(frame.cameraPos.w+0.5)&bit)!=0u;
}

fn intervalHistoryWeight()->f32{
  // Dynamic scenes retain converged static cones, but even a mover outside a
  // cone can change the direct-light visibility at its endpoint. Bound that
  // residual response to a short EMA; cones intersecting swept geometry are
  // rejected completely by dynamicConeHistoryValid below.
  return select(historyWeight(),min(historyWeight(),0.88),featureEnabled(128u));
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
  var step=0u;
  var attempts=0u;
  // WebGPU exposes only the weak compare/exchange operation. A spurious CAS
  // failure while the slot is still EMPTY must retry that exact slot: moving
  // on would leave a hole in the linear-probe cluster, and lookupProbeFrame
  // deliberately terminates at its first EMPTY slot. Such a hole made probe
  // coverage depend on GPU scheduling and produced rare temporal sparkles.
  while(step<32u&&attempts<128u){
    let slot = base + ((start + step) & mask);
    let result = atomicCompareExchangeWeak(&slots[slot].key, EMPTY, key);
    if (result.exchanged || result.old_value == key) { return; }
    attempts+=1u;
    if(result.old_value!=EMPTY){step+=1u;}
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
  return (frameIndex&1u)*ACCUM_FRAME_STRIDE+dataIndex(cascade,probe,direction)*5u;
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

fn quaternionRotate(vector:vec3f,quaternion:vec4f)->vec3f{
  let doubled=2.0*cross(quaternion.xyz,vector);
  return vector+quaternion.w*doubled+cross(quaternion.xyz,doubled);
}

fn inverseInstanceVector(vector:vec3f,instance:Triangle)->vec3f{
  return quaternionRotate(vector,vec4f(-instance.b.xyz,instance.b.w))/instance.c.xyz;
}

fn instancePoint(localPoint:vec3f,instance:Triangle)->vec3f{
  return instance.a.xyz+quaternionRotate(localPoint*instance.c.xyz,instance.b);
}

fn traceDynamicInstance(
  origin:vec3f,direction:vec3f,instanceIndex:u32,best:Hit
)->Hit{
  let instance=triangles[frame.dynamicInfo.y+instanceIndex];
  let localOrigin=inverseInstanceVector(origin-instance.a.xyz,instance);
  let localDirection=inverseInstanceVector(direction,instance);
  let inverseLocalDirection=select(
    vec3f(-1e20),vec3f(1e20),localDirection>=vec3f(0)
  )/max(vec3f(1),abs(localDirection)*1e20);
  var result=best;
  var stack:array<u32,64>;
  var stackSize=1u;
  stack[0]=instance.normalOct.x;
  loop{
    if(stackSize==0u){break;}
    stackSize-=1u;
    let nodeIndex=stack[stackSize];
    let node=bvhNodes[nodeIndex];
    if(rayBoxNear(
      localOrigin,inverseLocalDirection,node.minMeta.xyz,node.maxMeta.xyz,result.t
    )>=result.t){continue;}
    let left=bitcast<u32>(node.minMeta.w);
    let right=bitcast<u32>(node.maxMeta.w);
    if((left&0x80000000u)!=0u){
      let first=left&0x7fffffffu;
      for(var triangleOffset=0u;triangleOffset<right;triangleOffset++){
        let triangleIndex=first+triangleOffset;
        let triangle=triangles[triangleIndex];
        let intersection=traceTriangle(
          localOrigin,localDirection,triangle,result.t
        );
        if(intersection.x<result.t){
          let barycentric=vec3f(
            1.0-intersection.y-intersection.z,intersection.y,intersection.z
          );
          let localNormal=normalize(
            decodeOctNormal(triangle.normalOct.x)*barycentric.x
            +decodeOctNormal(triangle.normalOct.y)*barycentric.y
            +decodeOctNormal(triangle.normalOct.z)*barycentric.z
          );
          var worldNormal=normalize(quaternionRotate(
            localNormal/instance.c.xyz,instance.b
          ));
          if(dot(worldNormal,direction)>0.0){worldNormal=-worldNormal;}
          let sourceFrontFace=dot(cross(
            triangle.b.xyz-triangle.a.xyz,
            triangle.c.xyz-triangle.a.xyz
          ),localDirection)<0.0;
          let worldA=instancePoint(triangle.a.xyz,instance);
          let worldB=instancePoint(triangle.b.xyz,instance);
          let worldC=instancePoint(triangle.c.xyz,instance);
          let isEmitter=(instance.normalOct.w&1u)!=0u&&sourceFrontFace;
          result=Hit(
            intersection.x,worldNormal,
            triangle.albedo.xyz*instance.albedo.xyz,
            select(vec3f(0),instance.emissive.xyz,isEmitter),
            min(worldA,min(worldB,worldC)),max(worldA,max(worldB,worldC)),
            0x80000000u|triangleIndex
          );
        }
      }
    }else{
      let leftNear=rayBoxNear(
        localOrigin,inverseLocalDirection,bvhNodes[left].minMeta.xyz,
        bvhNodes[left].maxMeta.xyz,result.t
      );
      let rightNear=rayBoxNear(
        localOrigin,inverseLocalDirection,bvhNodes[right].minMeta.xyz,
        bvhNodes[right].maxMeta.xyz,result.t
      );
      if(leftNear<result.t&&rightNear<result.t){
        if(stackSize>61u){atomicAdd(&state[7],1u);break;}
        if(leftNear<rightNear){stack[stackSize]=right;stack[stackSize+1u]=left;}
        else{stack[stackSize]=left;stack[stackSize+1u]=right;}
        stackSize+=2u;
      }else if(leftNear<result.t){
        if(stackSize>62u){atomicAdd(&state[7],1u);break;}
        stack[stackSize]=left;stackSize+=1u;
      }else if(rightNear<result.t){
        if(stackSize>62u){atomicAdd(&state[7],1u);break;}
        stack[stackSize]=right;stackSize+=1u;
      }
    }
  }
  return result;
}

fn traceScene(origin: vec3f, directionIn: vec3f, maxDistance: f32) -> Hit {
  let direction=normalize(directionIn);
  let inverseDirection=select(vec3f(-1e20),vec3f(1e20),direction>=vec3f(0))
    /max(vec3f(1),abs(direction)*1e20);
  var result=Hit(
    maxDistance,vec3f(0,1,0),vec3f(0),vec3f(0),
    vec3f(0),vec3f(0),0xffffffffu
  );
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
          let sourceFrontFace=dot(
            cross(tri.b.xyz-tri.a.xyz,tri.c.xyz-tri.a.xyz),direction
          )<0.0;
          result=Hit(
            intersection.x,n,tri.albedo.xyz*surface.rgb,
            select(vec3f(0),tri.emissive.xyz,sourceFrontFace),
            min(tri.a.xyz,min(tri.b.xyz,tri.c.xyz)),
            max(tri.a.xyz,max(tri.b.xyz,tri.c.xyz)),first+j
          );
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
  if(frame.dynamicInfo.x!=0xffffffffu){
    var dynamicStack:array<u32,32>;
    var dynamicStackSize=1u;
    dynamicStack[0]=frame.dynamicInfo.x;
    loop{
      if(dynamicStackSize==0u){break;}
      dynamicStackSize-=1u;
      let nodeIndex=dynamicStack[dynamicStackSize];
      let node=bvhNodes[nodeIndex];
      if(rayBoxNear(
        origin,inverseDirection,node.minMeta.xyz,node.maxMeta.xyz,result.t
      )>=result.t){continue;}
      let left=bitcast<u32>(node.minMeta.w);
      let right=bitcast<u32>(node.maxMeta.w);
      if((left&0x80000000u)!=0u){
        let first=left&0x7fffffffu;
        for(var instanceOffset=0u;instanceOffset<right;instanceOffset++){
          result=traceDynamicInstance(
            origin,direction,first+instanceOffset,result
          );
        }
      }else{
        let leftNear=rayBoxNear(
          origin,inverseDirection,bvhNodes[left].minMeta.xyz,
          bvhNodes[left].maxMeta.xyz,result.t
        );
        let rightNear=rayBoxNear(
          origin,inverseDirection,bvhNodes[right].minMeta.xyz,
          bvhNodes[right].maxMeta.xyz,result.t
        );
        if(leftNear<result.t&&rightNear<result.t){
          if(dynamicStackSize>29u){atomicAdd(&state[7],1u);break;}
          if(leftNear<rightNear){
            dynamicStack[dynamicStackSize]=right;
            dynamicStack[dynamicStackSize+1u]=left;
          }else{
            dynamicStack[dynamicStackSize]=left;
            dynamicStack[dynamicStackSize+1u]=right;
          }
          dynamicStackSize+=2u;
        }else if(leftNear<result.t){
          dynamicStack[dynamicStackSize]=left;dynamicStackSize+=1u;
        }else if(rightNear<result.t){
          dynamicStack[dynamicStackSize]=right;dynamicStackSize+=1u;
        }
      }
    }
  }
  return result;
}

fn segmentIntersectsExpandedBox(
  origin:vec3f,inverseDirection:vec3f,minimum:vec3f,maximum:vec3f,
  startDistance:f32,endDistance:f32,expansion:f32
)->bool{
  let expandedMinimum=minimum-vec3f(expansion);
  let expandedMaximum=maximum+vec3f(expansion);
  let t0=(expandedMinimum-origin)*inverseDirection;
  let t1=(expandedMaximum-origin)*inverseDirection;
  let near3=min(t0,t1);
  let far3=max(t0,t1);
  let nearDistance=max(max(near3.x,near3.y),near3.z);
  let farDistance=min(min(far3.x,far3.y),far3.z);
  return farDistance>=max(0.0,startDistance)&&nearDistance<=endDistance;
}

fn dynamicConeRootClear(
  root:u32,origin:vec3f,directionIndexIn:u32,cascade:u32,lod:u32
)->bool{
  if(root==0xffffffffu){return true;}
  let direction=directionFromIndex(directionIndexIn,cascade);
  let inverseDirection=select(vec3f(-1e20),vec3f(1e20),direction>=vec3f(0))
    /max(vec3f(1),abs(direction)*1e20);
  let baseLength=frame.envBaseSpacing.w*exp2(f32(lod&3u))*1.6;
  let intervalScale=exp2(f32(cascade)*2.0);
  let endDistance=baseLength*intervalScale;
  let startDistance=select(0.0,baseLength*intervalScale*0.25,cascade>0u);
  // Test the complete spatio-angular support, not only the bin's center ray.
  // Algorithm 3 can contribute from any point in the probe cell and any
  // equal-area direction inside this bin. The half-cell diagonal covers the
  // former; the far-end angular radius derived from all four bin corners
  // covers the latter. This is a conservative, scene-independent swept-cone
  // predicate and prevents a thin mover or emitter from surviving in the
  // untested corner of a high-cascade directional bin.
  let theta=4u<<cascade;
  let width=theta*2u;
  let coordinate=mortonDirectionCoordinates(directionIndexIn,cascade);
  var minimumCosine=1.0;
  for(var corner=0u;corner<4u;corner++){
    let cornerUv=vec2f(
      (f32(coordinate.x)+f32(corner&1u))/f32(width),
      (f32(coordinate.y)+f32((corner>>1u)&1u))/f32(theta)
    );
    minimumCosine=min(minimumCosine,dot(direction,decodeEqualArea(cornerUv)));
  }
  let angularRadius=endDistance*sqrt(max(0.0,1.0-minimumCosine*minimumCosine));
  let footprint=cascadeSpacing(cascade,lod)*0.8660254+angularRadius;
  var stack:array<u32,32>;
  var stackSize=1u;
  stack[0]=root;
  loop{
    if(stackSize==0u){break;}
    stackSize-=1u;
    let nodeIndex=stack[stackSize];
    let node=bvhNodes[nodeIndex];
    if(!segmentIntersectsExpandedBox(
      origin,inverseDirection,node.minMeta.xyz,node.maxMeta.xyz,
      startDistance,endDistance,footprint
    )){continue;}
    let left=bitcast<u32>(node.minMeta.w);
    let right=bitcast<u32>(node.maxMeta.w);
    if((left&0x80000000u)!=0u){return false;}
    if(stackSize>29u){return false;}
    stack[stackSize]=left;
    stack[stackSize+1u]=right;
    stackSize+=2u;
  }
  return true;
}

fn dynamicConeHistoryValid(
  origin:vec3f,directionIndexIn:u32,cascade:u32,lod:u32
)->bool{
  return dynamicConeRootClear(
    frame.dynamicInfo.w,origin,directionIndexIn,cascade,lod
  );
}

fn directionBinSine(directionIndexIn:u32,cascade:u32)->f32{
  let direction=directionFromIndex(directionIndexIn,cascade);
  let theta=4u<<cascade;
  let width=theta*2u;
  let coordinate=mortonDirectionCoordinates(directionIndexIn,cascade);
  var minimumCosine=1.0;
  for(var corner=0u;corner<4u;corner++){
    let cornerUv=vec2f(
      (f32(coordinate.x)+f32(corner&1u))/f32(width),
      (f32(coordinate.y)+f32((corner>>1u)&1u))/f32(theta)
    );
    minimumCosine=min(minimumCosine,dot(direction,decodeEqualArea(cornerUv)));
  }
  return sqrt(max(0.0,1.0-minimumCosine*minimumCosine));
}

fn dynamicShadowCorridorClear(
  origin:vec3f,direction:vec3f,distanceLimit:f32,radius:f32
)->bool{
  let root=frame.dynamicInfo.w;
  if(root==0xffffffffu){return true;}
  let inverseDirection=select(vec3f(-1e20),vec3f(1e20),direction>=vec3f(0))
    /max(vec3f(1),abs(direction)*1e20);
  var stack:array<u32,32>;
  var stackSize=1u;
  stack[0]=root;
  loop{
    if(stackSize==0u){break;}
    stackSize-=1u;
    let node=bvhNodes[stack[stackSize]];
    if(!segmentIntersectsExpandedBox(
      origin,inverseDirection,node.minMeta.xyz,node.maxMeta.xyz,
      0.0,distanceLimit,radius
    )){continue;}
    let left=bitcast<u32>(node.minMeta.w);
    let right=bitcast<u32>(node.maxMeta.w);
    if((left&0x80000000u)!=0u){return false;}
    if(stackSize>29u){return false;}
    stack[stackSize]=left;
    stack[stackSize+1u]=right;
    stackSize+=2u;
  }
  return true;
}

// A stored interval's J embeds direct light at its hit point, and that hit
// lies inside the cascade's own [start,end] span. A mover crossing the
// LIGHT path to the endpoint changes J without ever intersecting the receiver
// cone, so the swept-cone predicate alone freezes stale mover shadows into
// converged history. Test conservative light corridors from four interval
// sample points toward each analytic source; sample radius covers the
// half-spacing between samples plus the bin's angular spread, so the swept
// union of the four capsules contains the full corridor volume.
fn dynamicEndpointShadingValid(
  origin:vec3f,directionIndexIn:u32,cascade:u32,lod:u32
)->bool{
  if(frame.dynamicInfo.w==0xffffffffu){return true;}
  let direction=directionFromIndex(directionIndexIn,cascade);
  let baseLength=frame.envBaseSpacing.w*exp2(f32(lod&3u))*1.6;
  let end=baseLength*exp2(f32(cascade)*2.0);
  let start=select(0.0,end*0.25,cascade>0u);
  let span=end-start;
  let binSine=directionBinSine(directionIndexIn,cascade);
  let towardSun=normalize(-frame.sunDirTime.xyz);
  let sunActive=frame.sunColorIntensity.w>0.0;
  let pointActive=frame.pointColorIntensity.w>0.0;
  if(!sunActive&&!pointActive){return true;}
  for(var sampleIndex=0u;sampleIndex<4u;sampleIndex++){
    let hitDistance=start+(f32(sampleIndex)+0.5)*0.25*span;
    let hitPoint=origin+direction*hitDistance;
    let sampleRadius=cascadeSpacing(cascade,lod)*0.8660254
      +hitDistance*binSine+span*0.125;
    if(sunActive&&!dynamicShadowCorridorClear(
      hitPoint,towardSun,frame.sceneBounds.w,sampleRadius
    )){return false;}
    if(pointActive){
      let toPoint=frame.pointPosRange.xyz-hitPoint;
      let pointDistance=length(toPoint);
      if(pointDistance>1e-4&&!dynamicShadowCorridorClear(
        hitPoint,toPoint/pointDistance,pointDistance,sampleRadius
      )){return false;}
    }
  }
  return true;
}

fn dynamicPointHistoryValid(position:vec3f,radius:f32)->bool{
  let sweptRoot=frame.dynamicInfo.w;
  if(sweptRoot==0xffffffffu){return true;}
  var stack:array<u32,32>;
  var stackSize=1u;
  stack[0]=sweptRoot;
  loop{
    if(stackSize==0u){break;}
    stackSize-=1u;
    let node=bvhNodes[stack[stackSize]];
    let delta=max(max(node.minMeta.xyz-position,vec3f(0)),position-node.maxMeta.xyz);
    if(dot(delta,delta)>radius*radius){continue;}
    let left=bitcast<u32>(node.minMeta.w);
    let right=bitcast<u32>(node.maxMeta.w);
    if((left&0x80000000u)!=0u){return false;}
    if(stackSize>29u){return false;}
    stack[stackSize]=left;
    stack[stackSize+1u]=right;
    stackSize+=2u;
  }
  return true;
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

fn loadAtlasIrradianceBilinear(
  probe:u32,frameIndex:u32,localIn:vec2f
)->vec4f{
  // Compact probe indices are intentionally unordered GPU storage addresses.
  // Interpolate entirely in the 8x8 tile's local coordinates before adding
  // the integer atlas origin, so assigning the same logical key to a different
  // compact index cannot change fractional precision or sample a neighbor tile.
  let local=clamp(localIn,vec2f(0),vec2f(7));
  let low=vec2i(floor(local));
  let high=min(low+vec2i(1),vec2i(7));
  let fraction=local-vec2f(low);
  let tile=vec2i(
    i32(probe%64u)*8,
    i32(probe/64u)*8+i32(frameIndex)*2048
  );
  let v00=textureLoad(irradianceAtlasSampled,tile+vec2i(low.x,low.y),0);
  let v10=textureLoad(irradianceAtlasSampled,tile+vec2i(high.x,low.y),0);
  let v01=textureLoad(irradianceAtlasSampled,tile+vec2i(low.x,high.y),0);
  let v11=textureLoad(irradianceAtlasSampled,tile+high,0);
  return mix(mix(v00,v10,fraction.x),mix(v01,v11,fraction.x),fraction.y);
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
  return loadAtlasIrradianceBilinear(probe,frameIndex,octCoordinate);
}

fn samplePrimaryIrradianceLod(position:vec3f,normal:vec3f,lod:u32)->vec4f{
  let spacing=cascadeSpacing(0u,lod);
  let grid=position/spacing-vec3f(0.5);
  let cell=vec3i(floor(grid));
  let linearFraction=fract(grid);
  // C1 surface reconstruction. Linear weights are only C0: a rigid surface
  // moving through the probe grid crosses a cell boundary somewhere on its
  // body almost every frame, and each crossing converts the field's spatial
  // curvature into a second-difference spike of the reconstructed lighting
  // (measured as ~14% of Lagrangian samples spiking per frame at Sponza mover
  // speeds). Hermite weights have zero derivative at cell boundaries, so the
  // same sparse field is C1 along any smooth motion path — for the camera,
  // for rigid receivers, and for static geometry alike.
  let fraction=linearFraction*linearFraction*(vec3f(3.0)-2.0*linearFraction);
  let fixedBits=vec3i(floor(position/spacing))-cell;
  let absoluteNormal=abs(normal);
  let allowHistoricalSupport=dynamicPointHistoryValid(position,spacing*1.75);
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
    let key=keyFromCellSurface(cell+bits,lod,pixelSurfaceClass(normal));
    let probe=lookupProbe(0u,key);
    var irradiance=vec4f(0);
    if(probe!=EMPTY&&probe<PROBE_CAPS[0]){
      irradiance=sampleAtlasIrradiance(probe,normal,currentFrame());
    }
    // A tangent support key can disappear from the current sparse hash when
    // primary-ray ownership shifts by one pixel. Search exact world/sheet
    // history whether the current probe is empty or merely has no ray yet.
    // This is the continuity bridge; restricting it to an existing current
    // probe made coverage itself flicker under camera translation.
    if(irradiance.a<0.001&&allowHistoricalSupport){
      for(var age=1u;age<4u;age++){
        let historyFrame=(currentFrame()+4u-age)&3u;
        let historyProbe=lookupProbeFrame(0u,key,historyFrame);
        if(historyProbe!=EMPTY&&historyProbe<PROBE_CAPS[0]){
          let historyIrradiance=sampleAtlasIrradiance(
            historyProbe,normal,historyFrame
          );
          if(historyIrradiance.a>0.001){
            irradiance=historyIrradiance;
            break;
          }
        }
      }
    }
    if(irradiance.a>0.001){
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
  let fineLod=u32(lods.x);
  let coarseLod=u32(lods.y);
  let fine=samplePrimaryIrradianceLod(position,normal,fineLod);
  var coarse=vec4f(0);
  if(coarseLod!=fineLod){
    coarse=samplePrimaryIrradianceLod(position,normal,coarseLod);
  }
  // Keep the path-reference/current-state oracle identical to production
  // final shading. A selected LOD can be temporarily cold even though an
  // exact world/sheet field remains resident at another bounded LOD; treating
  // that as black made the oracle report broad false under-lighting and was
  // the same abrupt viewport/zoom failure seen at the doorway.
  var resident=vec4f(0);
  var residentDistance=5u;
  for(var candidateLod=0u;candidateLod<4u;candidateLod++){
    if(candidateLod==fineLod||candidateLod==coarseLod){continue;}
    let candidate=samplePrimaryIrradianceLod(
      position,normal,candidateLod
    );
    let distance=u32(abs(i32(candidateLod)-i32(fineLod)));
    if(candidate.w>resident.w+0.001
      ||(abs(candidate.w-resident.w)<=0.001&&distance<residentDistance)){
      resident=candidate;
      residentDistance=distance;
    }
  }
  if(coarseLod==fineLod){
    if(fine.w<0.001){return resident;}
    if(resident.w<0.001||fine.w>=0.35){return fine;}
    let readiness=smoothstep(0.02,0.35,fine.w);
    return vec4f(
      mix(resident.xyz,fine.xyz,readiness),
      mix(resident.w,fine.w,readiness)
    );
  }
  if(fine.w<0.001){
    if(coarse.w>=0.001){return coarse;}
    return resident;
  }
  if(coarse.w<0.001){
    if(resident.w<0.001||fine.w>=0.35){return fine;}
    let readiness=smoothstep(0.02,0.35,fine.w);
    return vec4f(
      mix(resident.xyz,fine.xyz,readiness),
      mix(resident.w,fine.w,readiness)
    );
  }
  return vec4f(
    mix(fine.xyz,coarse.xyz,lods.z),mix(fine.w,coarse.w,lods.z)
  );
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
  var environmentTransmittance=0.0;
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
      cell+bits,lod,pixelSurfaceClass(normal)
    ));
    if(probe!=EMPTY&&probe<PROBE_CAPS[0]){
      let cone=cones[dataIndex(0u,probe,direction)];
      if(cone.w>0.5){
        value+=cone.xyz*weight;
        environmentTransmittance+=max(0.0,cone.w-1.0)*weight;
        total+=weight;
      }
    }
  }
  if(total<1e-5){return vec4f(0);}
  return vec4f(value/total,1.0+environmentTransmittance/total);
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
  return mix(fine,coarse,lods.z);
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
fn primaryEmissivePatchIrradiance(
  origin:vec3f,normal:vec3f,a:vec3f,b:vec3f,c:vec3f,
  emission:vec3f
)->vec3f{
  let centroid=(a+b+c)/3.0;
  let toLight=centroid-origin;
  let lightDistance=length(toLight);
  if(lightDistance<=1e-7){return vec3f(0);}
  let patchExtent=max(length(b-a),max(length(c-b),length(a-c)));
  let formFactor=primaryClippedTriangleFormFactor(
    a-origin,b-origin,c-origin,normal
  );
  if(formFactor<=1e-8){return vec3f(0);}
  // Visibility is integrated over stable world-space subtriangles. A moving
  // blocker can therefore reveal part of an emitter instead of toggling the
  // whole polygon through one closest-point ray. The endpoint contraction is
  // proportional to the ray/patch scale, never a scene-unit constant.
  let rayEpsilon=max(
    1e-7,min(lightDistance*0.001,patchExtent*0.001)
  );
  let traceDistance=max(1e-7,lightDistance-rayEpsilon);
  let blocker=traceScene(origin,toLight/lightDistance,traceDistance);
  if(blocker.t<traceDistance){return vec3f(0);}
  return emission*formFactor;
}
fn primaryEmissiveVisibility(
  origin:vec3f,lightPoint:vec3f,sourceExtent:f32
)->f32{
  let toLight=lightPoint-origin;
  let lightDistance=length(toLight);
  if(lightDistance<=1e-7){return 0.0;}
  let rayEpsilon=max(
    1e-7,min(lightDistance*0.001,sourceExtent*0.001)
  );
  let traceDistance=max(1e-7,lightDistance-rayEpsilon);
  let blocker=traceScene(origin,toLight/lightDistance,traceDistance);
  return select(0.0,1.0,blocker.t>=traceDistance);
}
fn primaryEmissiveTriangleIrradiance(
  origin:vec3f,normal:vec3f,triangle:Triangle,radius:f32
)->vec3f{
  if(max(triangle.emissive.x,max(triangle.emissive.y,triangle.emissive.z))<=0.0){
    return vec3f(0);
  }
  let a=triangle.a.xyz;
  let b=triangle.b.xyz;
  let c=triangle.c.xyz;
  // Radiance from an authored area source exists only in the hemisphere of
  // its geometric normal. This is independent of tessellation and scene
  // scale, and prevents back-side energy from contaminating nearby probes.
  let sourceNormal=cross(b-a,c-a);
  if(dot(sourceNormal,origin-(a+b+c)/3.0)<=0.0){return vec3f(0);}
  let minimum=min(a,min(b,c));
  let maximum=max(a,max(b,c));
  let proximityDistance=sqrt(primaryPointAabbDistanceSquared(
    origin,minimum,maximum
  ));
  if(proximityDistance>=radius){return vec3f(0);}
  // Smoothly cross-fade the analytic C(-1) estimator into c0 across a
  // geometry-scaled overlap. This is deliberately continuous; the prior
  // hard ownership boundary made a whole source appear or disappear.
  let intervalBlend=1.0-smoothstep(radius*0.72,radius,proximityDistance);
  let parentFormFactor=primaryClippedTriangleFormFactor(
    a-origin,b-origin,c-origin,normal
  );
  if(parentFormFactor<=1e-8){return vec3f(0);}
  let centroid=(a+b+c)/3.0;
  let sourceExtent=max(length(b-a),max(length(c-b),length(a-c)));
  // A tessellated curved emitter already supplies spatial quadrature through
  // its source facets. Subdividing a facet that fits within the C(-1)
  // footprint is redundant and makes cost depend on display tessellation.
  // The 2.5-interval bound is world-scale invariant; genuinely broad area
  // sources (including the Cornell ceiling panel) still take the full
  // partial-coverage path.
  if(sourceExtent<=radius*2.5){
    let visibility=primaryEmissiveVisibility(origin,centroid,sourceExtent);
    return triangle.emissive.xyz*(parentFormFactor*intervalBlend*visibility);
  }
  let samples=array<vec3f,7>(
    mix(a,centroid,0.06),mix(b,centroid,0.06),mix(c,centroid,0.06),
    (a+b)*0.5,(b+c)*0.5,(c+a)*0.5,centroid
  );
  var visibleSamples=0.0;
  for(var sampleIndex=0u;sampleIndex<7u;sampleIndex++){
    visibleSamples+=primaryEmissiveVisibility(
      origin,samples[sampleIndex],sourceExtent
    );
  }
  // The common unoccluded case retains the exact analytic polygon result.
  // Only a detected blocker invokes the bounded patch visibility quadrature.
  if(visibleSamples>=6.5){
    return triangle.emissive.xyz*(parentFormFactor*intervalBlend);
  }
  if(visibleSamples<=0.5){return vec3f(0);}
  let edgeB=(triangle.b.xyz-triangle.a.xyz)*0.125;
  let edgeC=(triangle.c.xyz-triangle.a.xyz)*0.125;
  var result=vec3f(0);
  // Eight-by-eight barycentric subdivision yields 64 stable integration
  // patches per source triangle. Their exact horizon-clipped form factors sum
  // to the parent polygon while independently sampled visibility represents
  // partial occlusion without temporal or screen-space noise.
  for(var i=0u;i<8u;i++){
    for(var j=0u;j<8u-i;j++){
      let p00=triangle.a.xyz+edgeB*f32(i)+edgeC*f32(j);
      let p10=p00+edgeB;
      let p01=p00+edgeC;
      result+=primaryEmissivePatchIrradiance(
        origin,normal,p00,p10,p01,triangle.emissive.xyz
      );
      if(i+j+2u<=8u){
        let p11=p10+edgeC;
        result+=primaryEmissivePatchIrradiance(
          origin,normal,p10,p11,p01,triangle.emissive.xyz
        );
      }
    }
  }
  return result*intervalBlend;
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
  if(frame.dynamicInfo.z!=0xffffffffu){
    var dynamicStack:array<u32,32>;
    var dynamicStackSize=1u;
    dynamicStack[0]=frame.dynamicInfo.z;
    loop{
      if(dynamicStackSize==0u){break;}
      dynamicStackSize-=1u;
      let node=bvhNodes[dynamicStack[dynamicStackSize]];
      if(primaryPointAabbDistanceSquared(
        origin,node.minMeta.xyz,node.maxMeta.xyz
      )>radiusSquared){continue;}
      let left=bitcast<u32>(node.minMeta.w);
      let right=bitcast<u32>(node.maxMeta.w);
      if((left&0x80000000u)!=0u){
        let firstInstance=left&0x7fffffffu;
        for(var instanceOffset=0u;instanceOffset<right;instanceOffset++){
          let instance=triangles[
            frame.dynamicInfo.y+firstInstance+instanceOffset
          ];
          if((instance.normalOct.w&1u)==0u){continue;}
          for(var triangleOffset=0u;triangleOffset<instance.normalOct.z;triangleOffset++){
            let localTriangle=triangles[instance.normalOct.y+triangleOffset];
            let worldTriangle=Triangle(
              vec4f(instancePoint(localTriangle.a.xyz,instance),0),
              vec4f(instancePoint(localTriangle.b.xyz,instance),0),
              vec4f(instancePoint(localTriangle.c.xyz,instance),0),
              vec4f(0),vec4f(instance.emissive.xyz,0),
              vec4f(0),vec4f(0),vec4u(0)
            );
            result+=primaryEmissiveTriangleIrradiance(
              origin,normal,worldTriangle,radius
            );
          }
        }
      }else{
        if(dynamicStackSize>29u){break;}
        dynamicStack[dynamicStackSize]=left;
        dynamicStack[dynamicStackSize+1u]=right;
        dynamicStackSize+=2u;
      }
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

// Decide once per frame whether the camera's connected region has any path to
// the environment. This must not inspect the G-buffer: using a background
// pixel or a visible sunlit surface made the result depend on framing, so the
// entire indirect field changed estimator when a dolly removed the final sky
// pixel. A dense, deterministic spherical Fibonacci set instead queries the
// scene BVH in world space. It is invariant to FOV and screen coverage, sees
// apertures in every direction (including an open door behind the camera), and
// remains exact for a genuinely sealed volume.
@compute @workgroup_size(64) fn classifyEnvironmentAccess(
  @builtin(global_invocation_id) gid:vec3u
){
  if(featureEnabled(64u)||frame.pointColorIntensity.w>0.0001){
    if(gid.x==0u){atomicStore(&state[8],1u);}
    return;
  }
  const ACCESS_RAY_COUNT=512u;
  if(gid.x>=ACCESS_RAY_COUNT){return;}
  if(atomicLoad(&state[8])!=0u){return;}
  let sample=f32(gid.x)+0.5;
  let y=1.0-2.0*sample/f32(ACCESS_RAY_COUNT);
  let radius=sqrt(max(0.0,1.0-y*y));
  let phi=6.28318530718*fract(sample*0.61803398875);
  let direction=vec3f(cos(phi)*radius,y,sin(phi)*radius);
  let traceEnd=frame.sceneBounds.w*1.001;
  let origin=frame.cameraPos.xyz+direction*max(0.004,frame.envBaseSpacing.w*0.006);
  let blocker=traceScene(origin,direction,traceEnd+0.001);
  if(blocker.t>=traceEnd){atomicStore(&state[8],1u);}
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
    insertProbeRaw(0u,keyFromCellSurface(cell+bits,lod,pixelSurfaceClass(normal)));
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
  setRigidSheetOverride(world,normal);
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

fn hazardWinnerIndex(probe:u32,lane:u32)->u32{
  return HAZARD_WINNER_OFFSET+probe*HAZARD_WINNER_LANES+lane;
}

fn hazardSecondIndex(probe:u32,lane:u32)->u32{
  return HAZARD_SECOND_OFFSET+probe*HAZARD_WINNER_LANES+lane;
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
  return u32(info.w+0.5);
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
    f32(lod|(surfaceClassFromKey(key)<<2u))
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
  setRigidSheetOverride(world,normal);
  let sheet=pixelSurfaceClass(normal);
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

@compute @workgroup_size(64) fn resolvePersistentC0(
  @builtin(global_invocation_id) gid:vec3u
){
  let probe=gid.x;
  if(probe>=PROBE_CAPS[0]){return;}
  let mapAddress=PERSISTENT_MAP_OFFSET+probe;
  atomicStore(&persistentIrradiance[mapAddress],EMPTY);
  let activeCount=min(atomicLoad(&state[0]),PROBE_CAPS[0]);
  if(probe>=activeCount){return;}
  // The persistent cold-revisit cache stays a static-scene extension. A
  // quiescent dynamic scene already accumulates exactly like a static one
  // (its swept hierarchy is empty), and enabling the contention-arbitrated
  // cache there made the emitter-step oracle non-repeatable and froze
  // under-converged cones into the daylight-door field.
  let persistentEnabled=featureEnabled(8u)&&!featureEnabled(128u)
    &&!featureEnabled(512u)&&!featureEnabled(16384u);
  if(!persistentEnabled){return;}
  let key=probeKeyFromInfo(probeMeta[PROBE_OFFSETS[0]+probe],0u);
  let slot=resolvePersistentProbeSlot(key);
  atomicStore(&persistentIrradiance[mapAddress],slot);
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

fn depositWeighted(
  cascade:u32,probe:u32,direction:u32,radiance:vec3f,beta:f32,weight:u32
){
  let base=accumIndex(cascade,probe,direction);
  let safe=min(max(radiance,vec3f(0)),vec3f(16));
  let scale=FIXED_SCALE*f32(weight);
  atomicAdd(&accum[base],u32(safe.r*scale+0.5));
  atomicAdd(&accum[base+1u],u32(safe.g*scale+0.5));
  atomicAdd(&accum[base+2u],u32(safe.b*scale+0.5));
  atomicAdd(&accum[base+3u],u32(clamp(beta,0.0,1.0)*scale+0.5));
  atomicAdd(&accum[base+4u],weight);
}

fn deposit(cascade:u32,probe:u32,direction:u32,radiance:vec3f,beta:f32){
  depositWeighted(cascade,probe,direction,radiance,beta,1u);
}

fn staticSurfaceCode(world:vec4f,normal:vec3f)->u32{
  // A hazard representative's priority code is a material identity. For a
  // rigid instance it must be quantized in OWNER-LOCAL space: a world-space
  // micro-cell changes for every surface point on every moved frame, which
  // reshuffles the winning anchor-ray origins and turns smooth rigid motion
  // into estimator noise. Immutable surfaces keep the world quantization.
  var canonicalPosition=world.xyz;
  var canonicalNormal=normal;
  let owner=packedDynamicOwner(world.w);
  if(owner!=EMPTY){
    let instance=triangles[frame.dynamicInfo.y+owner];
    canonicalPosition=inverseInstanceVector(world.xyz-instance.a.xyz,instance);
    canonicalNormal=normalize(
      quaternionRotate(normal,vec4f(-instance.b.xyz,instance.b.w))*instance.c.xyz
    );
  }
  let micro=max(0.001,frame.envBaseSpacing.w/128.0);
  let cell=vec3i(floor(canonicalPosition/micro));
  var code=hash32(bitcast<u32>(cell.x)*0x9e3779b9u);
  code=hash32(code^(bitcast<u32>(cell.y)*0x85ebca6bu));
  code=hash32(code^(bitcast<u32>(cell.z)*0xc2b2ae35u));
  code=hash32(code^((u32(world.w+0.5)>>2u)*0x165667b1u));
  code=hash32(code^(surfaceClass(canonicalNormal)*0x27d4eb2du));
  return code&0x00ffffffu;
}

fn mapRaySample(world:vec4f,normal:vec3f,lod:u32,stableSlot:u32){
  let probe=lookupProbe(0u,keyFromCellSurface(
    probeCell(world.xyz,0u,lod),lod,pixelSurfaceClass(normal)
  ));
  if(probe==EMPTY){return;}
  probeMeta[raySampleProbeBase()+stableSlot]=vec4f(
    f32(probe),
    // A small integer bit-cast to f32 is a subnormal. WebGPU backends may
    // flush subnormals, which makes the stable-slot validity tag unreliable.
    // Numeric f32 represents every 24-bit integer exactly.
    f32(passParams.sampleEpoch&0x00ffffffu),
    f32(staticSurfaceCode(world,normal)),
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
  setRigidSheetOverride(world,normal);
  let lods=lodSelection(world.xyz);
  let sample=sampleIndex(gid);
  mapRaySample(world,normal,u32(lods.x),sample);
  if(u32(lods.y)!=u32(lods.x)){
    mapRaySample(world,normal,u32(lods.y),samplesPerFrame()+sample);
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
  let epoch=u32(entry.y+0.5);
  return select(
    EMPTY,u32(entry.x+0.5),
    epoch==(passParams.sampleEpoch&0x00ffffffu)
  );
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

fn packedDynamicOwner(marker:f32)->u32{
  let code=u32(marker+0.5);
  if((code&0x800000u)==0u){return EMPTY;}
  return (code>>14u)&63u;
}
fn packedSurfaceIdentity(marker:f32)->u32{
  return u32(marker+0.5)>>2u;
}
fn packedClosedSurface(marker:f32)->bool{
  return (u32(marker+0.5)&1u)!=0u;
}



fn stableStaticProbeDirection(
  probe:u32,sampleLane:u32,representative:u32,normal:vec3f
)->vec3f{
  // Anchor directions are a fixed per-PROBE-KEY R2 sequence. Per-key seeds
  // decorrelate the fixed quadrature's directional bias between cells, which
  // matters for energy: a single universal direction set missed the same
  // narrow transport band everywhere at once and measurably starved
  // aperture-dominated rooms (the daylight door underlit by a third), while
  // spatially varied seeds let the four-probe gather average the band back.
  // The sheet class is excluded from the seed so a rigid rotation crossing
  // sheet boundaries cannot switch the quadrature.
  let info=probeMeta[PROBE_OFFSETS[0]+probe];
  let key=probeKeyFromInfo(info,0u)&0x1fffffffu;
  let seed=hash32(key^0x6a09e667u);
  let seed2=hash32(seed^0xbb67ae85u);
  let base=vec2f(f32(seed&65535u),f32(seed2&65535u))/65536.0;
  let g=1.324717957244746;
  let sequence=sampleLane+representative*HAZARD_WINNER_LANES;
  let uv=fract(
    base+(f32(sequence)+0.5)*vec2f(1.0/g,1.0/(g*g))
  );
  var direction=decodeEqualArea(uv);
  if(dot(direction,normal)<0.0){direction=-direction;}
  return direction;
}

fn staticDirectionNeedsCurrentSample(
  world:vec4f,direction:vec3f,lod:u32
)->bool{
  for(var cascade=0u;cascade<4u;cascade++){
    let origin=probePositionFromCell(
      probeCell(world.xyz,cascade,lod),cascade,lod
    );
    let directionIndexIn=directionIndex(direction,cascade);
    // A dynamic-scene cone uses the same fixed current-state estimator in a
    // stationary pose and in a swept frame. Otherwise the first frame after a
    // move is compared against a differently sampled warm field, producing a
    // visible estimator switch even when invalidation is geometrically exact.
    // Current-TLAS anchoring is localized to cones whose complete support can
    // see a rigid instance; immutable scenes retain the paper estimator.
    if(!dynamicConeRootClear(
      frame.dynamicInfo.x,origin,directionIndexIn,cascade,lod
    )){return true;}
    if(!dynamicConeHistoryValid(
      origin,directionIndexIn,cascade,lod
    )){return true;}
  }
  return false;
}


fn publishHazardCandidate(
  probe:u32,lane:u32,candidate:u32,second:bool
){
  if(second){
    let first=atomicLoad(&state[hazardWinnerIndex(probe,lane)]);
    if(candidate!=first){
      atomicMax(&state[hazardSecondIndex(probe,lane)],candidate);
    }
  }else{
    atomicMax(&state[hazardWinnerIndex(probe,lane)],candidate);
  }
}

fn hazardCandidate(probe:u32,lane:u32,stableSlot:u32)->u32{
  let code=u32(probeMeta[raySampleProbeBase()+stableSlot].z+0.5)
    &0x00ffffffu;
  let key=probeKeyFromInfo(probeMeta[PROBE_OFFSETS[0]+probe],0u);
  let rotation=(lane*0x009e3779u+(hash32(key)&0x00ffffffu))
    &0x00ffffffu;
  // State is cleared to zero. Keep zero as the unselected sentinel while the
  // 24-bit cyclic priority itself remains exactly representable and stable.
  return ((code+rotation)&0x00ffffffu)+1u;
}

struct StaticAnchorSelection{
  needed:u32,
  direction:vec3f,
};

fn staticAnchorForLod(
  world:vec4f,normal:vec3f,lod:u32,lane:u32,stableSlot:u32
)->StaticAnchorSelection{
  let probe=lookupProbe(0u,keyFromCellSurface(
    probeCell(world.xyz,0u,lod),lod,pixelSurfaceClass(normal)
  ));
  if(probe==EMPTY){
    return StaticAnchorSelection(0u,vec3f(0,0,1));
  }
  let winner=atomicLoad(&state[hazardWinnerIndex(probe,lane)]);
  let second=atomicLoad(&state[hazardSecondIndex(probe,lane)]);
  let candidate=hazardCandidate(probe,lane,stableSlot);
  if(winner!=candidate&&second!=candidate){
    return StaticAnchorSelection(0u,vec3f(0,0,1));
  }
  let representative=select(1u,0u,winner==candidate);
  let direction=stableStaticProbeDirection(
    probe,lane,representative,normal
  );
  return StaticAnchorSelection(
    select(0u,1u,staticDirectionNeedsCurrentSample(world,direction,lod)),
    direction
  );
}

fn selectStaticHazardRepresentative(gid:vec3u,second:bool){
  let giSize=vec2u(u32(frame.resolution.z),u32(frame.resolution.w));
  let baseSamples=max(1u,passParams.value);
  if(any(gid.xy>=giSize)||gid.z>=baseSamples*64u){return;}
  let fullSize=vec2u(u32(frame.resolution.x),u32(frame.resolution.y));
  let pixel=min(fullSize-vec2u(1),gid.xy*fullSize/giSize);
  let world=textureLoad(worldTex,vec2i(pixel),0);
  if(world.w<0.5){return;}
  let normal=gbufferNormal(vec2i(pixel));
  setRigidSheetOverride(world,normal);
  let lods=lodSelection(world.xyz);
  let lane=gid.z/baseSamples;
  let baseGid=vec3u(gid.xy,gid.z%baseSamples);
  let sample=sampleIndex(baseGid);
  let sheet=pixelSurfaceClass(normal);
  let fine=u32(lods.x);
  let fineProbe=lookupProbe(0u,keyFromCellSurface(
    probeCell(world.xyz,0u,fine),fine,sheet
  ));
  if(fineProbe!=EMPTY){
    publishHazardCandidate(
      fineProbe,lane,hazardCandidate(fineProbe,lane,sample),second
    );
  }
  let coarse=u32(lods.y);
  if(coarse!=fine){
    let coarseProbe=lookupProbe(0u,keyFromCellSurface(
      probeCell(world.xyz,0u,coarse),coarse,sheet
    ));
    if(coarseProbe!=EMPTY){
      publishHazardCandidate(
        coarseProbe,lane,hazardCandidate(
          coarseProbe,lane,samplesPerFrame()+sample
        ),second
      );
    }
  }
}


@compute @workgroup_size(8,8) fn selectStaticHazardRepresentatives(
  @builtin(global_invocation_id) gid:vec3u
){
  selectStaticHazardRepresentative(gid,false);
}

@compute @workgroup_size(8,8) fn selectSecondStaticHazardRepresentatives(
  @builtin(global_invocation_id) gid:vec3u
){
  selectStaticHazardRepresentative(gid,true);
}

// Rigid receivers use the unified world-field path; the owner-local
// material-node estimator (32768-slot barycentric lattice, 1024 rays per
// node per frame) was removed once motion-aware cache invalidation made
// the shared sparse field temporally stable on moving surfaces.
fn traceAndSplit(
  world:vec4f,normal:vec3f,lod:u32,stableSlot:u32,sampleLane:u32,
  currentStateAnchor:bool,anchorDirection:vec3f
){
  let sheet=pixelSurfaceClass(normal);
  let key0=keyFromCellSurface(probeCell(world.xyz,0u,lod),lod,sheet);
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
  // Two independent fixed-point Weyl sequences implement the paper's global
  // temporal R2 rotation. Shifting before f32 conversion preserves all 24
  // representable mantissa bits, avoids correlated low/high halves, and keeps
  // deterministic reset/replay across the complete u32 frame cycle.
  // Swap the two R2 fixed-point components relative to the spatial recurrence.
  // The same order mostly permutes an existing prefix; this decorrelated global
  // rotation fills angular gaps as temporal samples accumulate.
  let temporalX=sampleFrame*0x91e10da5u;
  let temporalY=sampleFrame*0xc13fa9a9u;
  var jitter=vec2f(
    f32(temporalX>>8u),f32(temporalY>>8u)
  )*(1.0/16777216.0);
  if(!featureEnabled(8u)){
    let temporal=hash32(u32(frame.sunDirTime.w*60.0));
    let temporal2=hash32(temporal^0x9e3779b9u);
    jitter=vec2f(f32(temporal&65535u),f32(temporal2&65535u))/65536.0;
  }
  // Immutable and rigid receivers alike retain Algorithm 3's contiguous
  // hierarchical R2 ranges and the global temporal rotation; the deposit
  // gate below keeps population-dependent rays out of swept-invalid cones.
  let stableSequenceIndex=sequenceIndex+sampleLane*samplesPerFrame();
  var uv=fract(vec2f(0.5)+f32(stableSequenceIndex+1u)*alpha+jitter);
  var direction=decodeEqualArea(uv);
  if(currentStateAnchor){
    direction=anchorDirection;
  }else if(dot(direction,normal)<0.0){direction=-direction;}

  let baseLength=frame.envBaseSpacing.w*exp2(f32(lod&3u))*1.6;
  let maxDistance=baseLength*64.0;
  // Section 7.1's C(-1) prototype is optional. When its directional interval
  // is not evaluated for every open receiver, the paper's base c0 must begin
  // at the receiving surface (t_-1 = 0). Starting c0 at delta_s0 while later
  // returning the unmodified c0 field silently dropped all nearby reflected
  // radiance and produced the stable dark patches found by the path-reference
  // gate. Near emitters are still integrated analytically below, but ordinary
  // near geometry remains part of the unbiased ray-splitting estimator.
  let cMinusOneEnd=0.0;
  let surfaceOrigin=world.xyz+normal*max(0.008,frame.envBaseSpacing.w*0.012);
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
  var radiance=select(
    frame.envBaseSpacing.xyz,directAtHit(origin+direction*hit.t,hit),didHit
  );
  // The analytic C(-1) source estimator owns a smooth, receiver-to-triangle
  // fraction of every nearby emitter -- not merely ray hits whose travel
  // distance is shorter than the probe spacing. A receiver can be close to a
  // large polygon while a sampled direction reaches its far edge much later.
  // Testing hit.t therefore double-counted most of that polygon and produced
  // the bright red fringe beside Sponza's floor emitter. Probe deposits need
  // a source footprint dilated by the four-sample tangent-plane interpolation
  // support: the maximum receiver-to-contributor span is sqrt(2) cells. A 1.5
  // cell bound keeps the receiver's exact term out of every probe it samples,
  // independent of scene scale or emitter dimensions. Reflected sun/point
  // light remains in the stochastic transport field.
  if(didHit&&hit.triangleIndex!=0xffffffffu
    &&max(hit.emissive.x,max(hit.emissive.y,hit.emissive.z))>0.0){
    let sourceProximity=sqrt(primaryPointAabbDistanceSquared(
      surfaceOrigin,hit.sourceMinimum,hit.sourceMaximum
    ));
    let nearSourceRadius=frame.envBaseSpacing.w*1.5;
    let sourceOwnership=1.0-smoothstep(
      nearSourceRadius*0.72,nearSourceRadius,sourceProximity
    );
    radiance=max(vec3f(0),radiance-hit.emissive.xyz*sourceOwnership);
  }
  atomicAdd(&state[4],1u);
  if(didHit){atomicAdd(&state[5],1u);}
  for(var cascade=0u;cascade<4u;cascade++){
    if(cascade>targetCascade){break;}
    let key=keyFromCellSurface(
      probeCell(world.xyz,cascade,lod),lod,sheet
    );
    let probe=lookupProbe(cascade,key);
    if(probe==EMPTY){continue;}
    let dir=directionIndex(direction,cascade);
    // A swept-invalid cone cannot accumulate: its value each frame is a
    // fresh estimate, so every sample it receives must be a smooth function
    // of the scene pose alone. Algorithm 3 rays are not — their rank/count
    // assignment reshuffles with the visible-pixel population every frame,
    // which measured as ~5% per-frame luminance churn on exact-key probes
    // near movers. Population-dependent rays therefore feed only cones that
    // still accumulate (where churn averages out); swept-invalid cones
    // resolve purely from the deterministic probe-keyed anchor quadrature.
    // The gate exists to keep population churn out of cones under CONTINUOUS
    // motion. On a teleport or radiometric-step frame history is rejected
    // outright anyway, and the very first fresh estimate should use every
    // available ray — anchor-only first frames measurably diverged from the
    // converged state in the round-trip immediate-closure gates.
    if(!currentStateAnchor&&frame.dynamicInfo.w!=0xffffffffu
      &&!featureEnabled(2048u)&&!featureEnabled(512u)){
      let coneOrigin=probePositionFromCell(
        probeCell(world.xyz,cascade,lod),cascade,lod
      );
      if(!dynamicConeRootClear(
        frame.dynamicInfo.w,coneOrigin,dir,cascade,lod
      )){continue;}
    }
    if(cascade<targetCascade){deposit(cascade,probe,dir,vec3f(0),1.0);}
    else {deposit(cascade,probe,dir,radiance,0.0);}
  }
}

@compute @workgroup_size(8,8) fn splitRays(@builtin(global_invocation_id) gid: vec3u) {
  let giSize=vec2u(u32(frame.resolution.z),u32(frame.resolution.w));
  let baseSamples=max(1u,passParams.value);
  if(any(gid.xy>=giSize)||gid.z>=baseSamples*64u){return;}
  let fullSize=vec2u(u32(frame.resolution.x),u32(frame.resolution.y));
  let pixel=min(fullSize-vec2u(1),gid.xy*fullSize/giSize);
  let world=textureLoad(worldTex,vec2i(pixel),0);
  if(world.w<0.5){return;}
  let normal=gbufferNormal(vec2i(pixel));
  setRigidSheetOverride(world,normal);
  let lods=lodSelection(world.xyz);
  let fine=u32(lods.x);
  let laneGroup=gid.z/baseSamples;
  let baseGid=vec3u(gid.xy,gid.z%baseSamples);
  let sample=sampleIndex(baseGid);
  let coarse=u32(lods.y);
  var fineAnchor=StaticAnchorSelection(0u,vec3f(0,0,1));
  var coarseAnchor=StaticAnchorSelection(0u,vec3f(0,0,1));
  // Rigid receivers use the identical paper path: their pixels seed the same
  // world probes, own the same Algorithm 3 ranks, and publish the same
  // hazard-anchor candidates as static surfaces. Their surface cones overlap
  // their own swept TLAS, so motion invalidates exactly the history that
  // depended on the previous pose; a stationary rigid instance accumulates
  // like any other static geometry.
  if(laneGroup>0u){
    fineAnchor=staticAnchorForLod(
      world,normal,fine,laneGroup,sample
    );
    if(coarse!=fine){
      coarseAnchor=staticAnchorForLod(
        world,normal,coarse,laneGroup,samplesPerFrame()+sample
      );
    }
    if(fineAnchor.needed==0u&&coarseAnchor.needed==0u){return;}
  }
  let baseLane=laneGroup==0u;
  if(baseLane||fineAnchor.needed!=0u){
    traceAndSplit(
      world,normal,fine,sample,gid.z,
      fineAnchor.needed!=0u,fineAnchor.direction
    );
  }
  // Each selected LOD owns its exact probe key, hazard predicate, and anchor
  // direction. Sharing a fine trigger with the coarse call (or vice versa)
  // changed the estimator while crossing the blend band.
  if(coarse!=fine&&(baseLane||coarseAnchor.needed!=0u)){
    traceAndSplit(
      world,normal,coarse,samplesPerFrame()+sample,gid.z,
      coarseAnchor.needed!=0u,coarseAnchor.direction
    );
  }
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
  var environmentTransmittance=0.0;
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
        environmentTransmittance+=max(0.0,cone.w-1.0)*weight;
        total+=weight;
      }
    }
  }
  if(total<1e-5){return vec4f(0);}
  return vec4f(value/total,1.0+environmentTransmittance/total);
}

fn mergedParent(
  cascade:u32,direction:u32,position:vec3f,lod:u32,sheet:u32
)->vec4f{
  var sum=vec3f(0);
  var environmentTransmittance=0.0;
  var valid=0.0;
  for(var child=0u;child<4u;child++){
    // Morton ordering makes the four angular children of every lower
    // direction one contiguous aligned group, matching Section 6.
    let parentDirection=direction*4u+child;
    let parent=sampleParentDirection(cascade,position,lod,sheet,parentDirection);
    if(parent.w>0.5){
      sum+=parent.xyz;
      environmentTransmittance+=max(0.0,parent.w-1.0);
      valid+=1.0;
    }
  }
  if(valid<0.5){return vec4f(0);}
  return vec4f(sum/valid,1.0+environmentTransmittance/valid);
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
  let previousFrame=(currentFrame()+3u)&3u;
  let previousProbe=lookupProbeFrame(cascade,key,previousFrame);
  // A hard analytic/source-output step rejects all history. Continuous sun or
  // point-light motion changes the endpoint radiance globally and therefore
  // uses a bounded responsive EMA. A finite moving mesh emitter or rigid
  // blocker is different: only cones overlapping its swept TLAS are stale,
  // and for CONTINUOUS rigid motion that staleness is bounded by how far the
  // object moved in a frame. Discarding the cone outright replaced a
  // converged value with a fresh estimator in a single frame — a visible pop
  // on every invalidation edge. Instead the cone's history keeps
  // participating with its effective sample count capped low, so the fresh
  // deterministic estimate takes over within a few frames, smoothly.
  // Teleports (2048) and radiometric steps (512) still reject outright:
  // their history describes a state that never smoothly connects to this one.
  var historySampleCap=65535u;
  if(featureEnabled(512u)){
    historySampleCap=0u;
  }else if(!dynamicConeHistoryValid(
    probeInfo.xyz,direction,cascade,lod
  )){
    historySampleCap=select(12u,0u,featureEnabled(2048u));
  }
  if(historyWeight()>0.0&&historySampleCap>0u
    &&previousProbe!=EMPTY&&previousProbe<PROBE_CAPS[cascade]){
    let previousBase=accumIndexFrame(cascade,previousProbe,direction,previousFrame);
    var previousSamples=atomicLoad(&accum[previousBase+4u]);
    if(previousSamples>0u){
      let previousDenominator=FIXED_SCALE*f32(previousSamples);
      let previousInterval=vec3f(
        f32(atomicLoad(&accum[previousBase])),
        f32(atomicLoad(&accum[previousBase+1u])),
        f32(atomicLoad(&accum[previousBase+2u]))
      )/previousDenominator;
      // History that carries radiance embeds direct shading at its hit point.
      // A mover crossing the light corridor to that endpoint leaves the
      // receiver cone untouched, so converged history would freeze the stale
      // shadow in place; cap it to the same graceful window instead.
      if(max(previousInterval.x,max(previousInterval.y,previousInterval.z))>1e-5
        &&historySampleCap>12u
        &&!dynamicEndpointShadingValid(probeInfo.xyz,direction,cascade,lod)){
        historySampleCap=12u;
      }
    }
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
        if(featureEnabled(8192u)&&historySampleCap==65535u){
          // The converged world-key field is view independent. While a static
          // camera moves, retain the exact-key value instead of replacing it
          // with a different screen population. New/disoccluded keys have no
          // previous probe and therefore still resolve from fresh rays.
          interval=previousInterval;
          beta=previousBeta;
          resolvedSamples=previousSamples;
        }else if(featureEnabled(16384u)&&historySampleCap==65535u){
          // Actual source motion needs a bounded response time. The UI's
          // animation checkbox is not transport state: audits and shareable
          // frozen-time poses may keep it checked while the source is fixed,
          // and those frames must retain the paper's exact running average.
          let temporalWeight=intervalHistoryWeight();
          interval=mix(interval,previousInterval,temporalWeight);
          beta=mix(beta,previousBeta,temporalWeight);
          resolvedSamples=1u;
        }else{
          // Section 5.2 accumulates rays for semi-static scenes. Preserve an
          // effective sample count so repeated exact-key probes converge as a
          // true running average instead of a path-dependent fixed EMA.
          // Fixed-point radiance is clamped to 16 and scaled by 4096. 65,535
          // samples therefore occupy at most 16*4096*65535 = 4,294,901,760,
          // still below u32 max. Use that full overflow-safe history window so
          // one dense frame cannot visibly replace a converged static cone.
          let boundedPrevious=min(previousSamples,historySampleCap);
          let totalSamples=samples+boundedPrevious;
          interval=(
            interval*f32(samples)+previousInterval*f32(boundedPrevious)
          )/f32(max(1u,totalSamples));
          beta=(
            beta*f32(samples)+previousBeta*f32(boundedPrevious)
          )/f32(max(1u,totalSamples));
          resolvedSamples=min(65535u,totalSamples);
        }
      }else{
        interval=previousInterval;
        beta=previousBeta;
        resolvedSamples=min(previousSamples,historySampleCap);
      }
    }
  }
  var persistentSlot=EMPTY;
  if(cascade==0u){
    persistentSlot=atomicLoad(&persistentIrradiance[PERSISTENT_MAP_OFFSET+probe]);
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
  // w packs validity + the fraction of this cone that still reaches the
  // environment. Keeping transmittance beside radiance lets the per-pixel
  // visibility merge remove only leaked sky energy without discarding valid
  // light from internal emitters or point sources.
  var distant=vec4f(frame.envBaseSpacing.xyz,2.0);
  if(cascade<3u){distant=mergedParent(cascade,direction,probeInfo.xyz,lod,sheet);}
  var resolvedCone=vec4f(0);
  if(!hasInterval){
    // Full trilinear support can introduce probes that own no screen ray.
    // They have no measured transmittance, so inheriting a parent environment
    // cone fabricates an escape path through sealed geometry. Keep them
    // invalid; the spatial gather renormalizes over measured same-sheet
    // neighbors, as required for sparse probes in Section 5.
    resolvedCone=vec4f(0);
  }else if(beta>0.999&&distant.w<0.5){
    resolvedCone=vec4f(0);
  }else{
    resolvedCone=vec4f(
      min(vec3f(16.0),interval+clamp(beta,0.0,1.0)*distant.xyz),
      1.0+clamp(beta,0.0,1.0)*max(0.0,distant.w-1.0)
    );
  }
  // Fixed immutable scenes retain the fully composed c0 directional field,
  // including parent contribution and validity/transmittance. This is the
  // world-space transport representation consumed by irradiance prefiltering,
  // not recursive screen history. During camera motion an existing exact key
  // is invariant; a cold key publishes its current result immediately.
  if(cascade==0u&&persistentSlot!=EMPTY){
    let persistentBase=persistentDirectionBase(persistentSlot,direction);
    let persistentReady=atomicLoad(&persistentIrradiance[persistentBase+4u]);
    if(persistentReady!=0u&&featureEnabled(8192u)){
      resolvedCone=vec4f(
        bitcast<f32>(atomicLoad(&persistentIrradiance[persistentBase])),
        bitcast<f32>(atomicLoad(&persistentIrradiance[persistentBase+1u])),
        bitcast<f32>(atomicLoad(&persistentIrradiance[persistentBase+2u])),
        bitcast<f32>(atomicLoad(&persistentIrradiance[persistentBase+3u]))
      );
    }else if(resolvedCone.w>0.5){
      atomicStore(&persistentIrradiance[persistentBase],bitcast<u32>(resolvedCone.x));
      atomicStore(&persistentIrradiance[persistentBase+1u],bitcast<u32>(resolvedCone.y));
      atomicStore(&persistentIrradiance[persistentBase+2u],bitcast<u32>(resolvedCone.z));
      atomicStore(&persistentIrradiance[persistentBase+3u],bitcast<u32>(resolvedCone.w));
      atomicStore(&persistentIrradiance[persistentBase+4u],1u);
    }else if(persistentReady!=0u){
      resolvedCone=vec4f(
        bitcast<f32>(atomicLoad(&persistentIrradiance[persistentBase])),
        bitcast<f32>(atomicLoad(&persistentIrradiance[persistentBase+1u])),
        bitcast<f32>(atomicLoad(&persistentIrradiance[persistentBase+2u])),
        bitcast<f32>(atomicLoad(&persistentIrradiance[persistentBase+3u]))
      );
    }
  }
  cones[index]=resolvedCone;
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

@compute @workgroup_size(64) fn prefilterIrradiance(
  @builtin(global_invocation_id) gid:vec3u
) {
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
  var cosineWeight=0.0;
  for(var direction=0u;direction<32u;direction++){
    let ray=directionFromIndex(direction,0u);
    let cone=cones[dataIndex(0u,probe,direction)];
    if(cone.w>0.5){
      let cosine=max(0.0,dot(normal,ray));
      result+=cone.xyz*cosine;
      cosineWeight+=cosine;
    }
  }
  // Sparse RC ignores unfilled directions during merging. Do the same during
  // the final Lambertian prefilter instead of treating missing angular bins as
  // black radiance. For complete uniform-sphere coverage cosineWeight is N/4,
  // so this normalization is identical to the paper's 4/N estimator. Alpha
  // carries continuous coverage confidence into sparse trilinear shading.
  let filtered=select(vec3f(0),result/max(cosineWeight,1e-6),cosineWeight>1e-6);
  let confidence=clamp(cosineWeight*(4.0/32.0),0.0,1.0);
  let stored=vec4f(filtered,confidence);
  irradiance[currentFrame()*IRRADIANCE_FRAME_STRIDE+probe*64u+texel]=stored;
  let tile=vec2u(probe%64u,probe/64u)*8u;
  let atlasCoordinate=vec2i(tile+vec2u(x,y)+vec2u(0u,currentFrame()*2048u));
  textureStore(irradianceAtlasStorage,atlasCoordinate,stored);
}

fn resolvePrimaryIrradianceBase(
  world:vec3f,normal:vec3f,base:vec3f,closedBackFace:bool
)->vec3f{
  let intervalEnd=frame.envBaseSpacing.w;
  let origin=world+normal*max(0.006,intervalEnd*0.012);
  let nearEmission=primaryNearEmissiveIrradiance(
    origin,normal,intervalEnd
  );
  // A back face explicitly authored as part of one closed volume needs the
  // long guard. Ordinary receiving surfaces retain the paper's C(-1) extent;
  // empty tangent-support probes are rejected at merge time instead of being
  // allowed to manufacture environment visibility.
  let visibilityGuardEnd=select(
    intervalEnd,max(intervalEnd,frame.sceneBounds.w*1.001),closedBackFace
  );
  var ambientWeight=0.0;
  var ambientVisible=0.0;
  if(!closedBackFace){return base+nearEmission;}
  // The ambient-form optimization avoids Section 7.1's expensive directional
  // gather. One deterministic visibility sample per already-filtered c0 cone
  // is sufficient because the result is a single hemispherical ratio, not 32
  // independently displayed angular sectors.
  for(var directionIndexValue=0u;directionIndexValue<32u;directionIndexValue++){
    let direction=directionFromIndex(directionIndexValue,0u);
    let cosine=max(0.0,dot(normal,direction));
    if(cosine<=0.0){continue;}
    ambientWeight+=cosine;
    let hit=traceScene(origin,direction,visibilityGuardEnd+0.001);
    var visible=hit.t>=visibilityGuardEnd;
    if(visible){ambientVisible+=cosine;}
  }
  // Section 7.1 explicitly proposes this ambient-form production path for
  // C(-1): retain the smooth directional c0 irradiance and apply the exact
  // short-interval energy ratio as one scalar. Clamping the validated energy
  // prevents a single sub-cone hit from becoming a bright angular petal.
  let visibilityCorrection=select(
    1.0,clamp(ambientVisible/ambientWeight,0.0,1.0),
    ambientWeight>1e-7
  );
  return base*visibilityCorrection+nearEmission;
}

fn resolvedPrimaryIrradiance(
  world:vec3f,normal:vec3f,closedBackFace:bool
)->vec3f{
  return resolvePrimaryIrradianceBase(
    world,normal,samplePrimaryIrradiance(world,normal).xyz,closedBackFace
  );
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
  if((u32(world.w+0.5)&2u)!=0u
    ||max(packedNormal.z,packedNormal.w)>1e-5){return;}
  let normal=gbufferNormal(vec2i(pixel));
  setRigidSheetOverride(world,normal);
  let base=(gid.y*outputSize.x+gid.x)*8u;
  if(gid.z==0u){
    // Rigid receivers consume the same sparse world field as static
    // surfaces, so the oracle evaluates the identical reconstruction for
    // every pixel; the sheet override above already keys mover pixels by
    // their owner-canonical class. (The former material-node audit texture
    // is gone — reading it mislabeled the entire door leaf as black.)
    var current=resolvedPrimaryIrradiance(
      world.xyz,normal,packedClosedSurface(world.w)
    );
    current=clamp(current,vec3f(0),vec3f(16));
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
  for(var y=-1;y<=1;y++){
    for(var x=-1;x<=1;x++){
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
  sceneBounds: vec4f,
  dynamicInfo: vec4u,
  previousViewProj: mat4x4<f32>,
  lodCamera: vec4f,
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
@group(0) @binding(9) var<storage,read_write> frameState: array<atomic<u32>>;
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
@group(0) @binding(21) var dynamicReceiverIrradiance:texture_2d<f32>;

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
fn finalFeatureEnabled(bit:u32)->bool{
  return (u32(frame.cameraPos.w+0.5)&bit)!=0u;
}

fn finalDynamicPointHistoryValid(position:vec3f,radius:f32)->bool{
  let sweptRoot=frame.dynamicInfo.w;
  if(sweptRoot==0xffffffffu){return true;}
  var stack:array<u32,32>;
  var stackSize=1u;
  stack[0]=sweptRoot;
  loop{
    if(stackSize==0u){break;}
    stackSize-=1u;
    let node=shortBvhNodes[stack[stackSize]];
    let delta=max(max(node.minMeta.xyz-position,vec3f(0)),position-node.maxMeta.xyz);
    if(dot(delta,delta)>radius*radius){continue;}
    let left=bitcast<u32>(node.minMeta.w);
    let right=bitcast<u32>(node.maxMeta.w);
    if((left&0x80000000u)!=0u){return false;}
    if(stackSize>29u){return false;}
    stack[stackSize]=left;
    stack[stackSize+1u]=right;
    stackSize+=2u;
  }
  return true;
}
fn finalDynamicPointNearCurrent(position:vec3f,radius:f32)->bool{
  let currentRoot=frame.dynamicInfo.x;
  if(currentRoot==0xffffffffu){return false;}
  var stack:array<u32,32>;
  var stackSize=1u;
  stack[0]=currentRoot;
  loop{
    if(stackSize==0u){break;}
    stackSize-=1u;
    let node=shortBvhNodes[stack[stackSize]];
    let delta=max(max(node.minMeta.xyz-position,vec3f(0)),position-node.maxMeta.xyz);
    if(dot(delta,delta)>radius*radius){continue;}
    let left=bitcast<u32>(node.minMeta.w);
    let right=bitcast<u32>(node.maxMeta.w);
    if((left&0x80000000u)!=0u){return true;}
    if(stackSize>29u){return true;}
    stack[stackSize]=left;
    stack[stackSize+1u]=right;
    stackSize+=2u;
  }
  return false;
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

// Mirror of the compute module's rigid sheet override: rigid pixels key their
// probe lookups by the owner-canonical normal so rotation cannot swap the
// interpolation support mid-tumble. 8 means "no override" (static pixel).
var<private> rigidSheetOverride:u32=8u;

fn pixelSurfaceClass(normalIn:vec3f)->u32{
  return select(
    surfaceClass(normalIn),rigidSheetOverride,rigidSheetOverride!=8u
  );
}

fn setFinalRigidSheetOverride(owner:u32,normal:vec3f){
  rigidSheetOverride=8u;
  if(owner==0xffffffffu){return;}
  let instance=shortTriangles[frame.dynamicInfo.y+owner];
  let canonicalNormal=normalize(
    shortQuaternionRotate(normal,vec4f(-instance.b.xyz,instance.b.w))
    *instance.c.xyz
  );
  rigidSheetOverride=surfaceClass(canonicalNormal);
}

fn keyFromCellSurface(cellIn:vec3i,lod:u32,surfaceClassValue:u32)->u32{
  if(any(cellIn<vec3i(-256))||any(cellIn>vec3i(255))){return EMPTY;}
  let c=cellIn+vec3i(256);
  return u32(c.x)|(u32(c.y)<<9u)|(u32(c.z)<<18u)|((lod&3u)<<27u)|((surfaceClassValue&7u)<<29u);
}
fn lookupProbeCascadeFrame(cascade:u32,key:u32,frameIndex:u32)->u32{
  let mask=HASH_SIZES[cascade]-1u;
  let start=hash32(key)&mask;
  let base=frameIndex*HASH_FRAME_STRIDE+HASH_OFFSETS[cascade];
  for(var step=0u;step<32u;step++){
    let slot=base+((start+step)&mask);
    let found=atomicLoad(&slots[slot].key);
    if(found==key){return atomicLoad(&slots[slot].index);}
    if(found==EMPTY){return EMPTY;}
  }
  return EMPTY;
}
fn lookupProbeCascade(cascade:u32,key:u32)->u32{
  return lookupProbeCascadeFrame(
    cascade,key,u32(floor(frame.controls.w))&3u
  );
}
fn lookupProbe(key:u32)->u32{return lookupProbeCascade(0u,key);}
fn loadFinalAtlasIrradianceBilinear(
  probe:u32,frameIndex:u32,localIn:vec2f
)->vec4f{
  let local=clamp(localIn,vec2f(0),vec2f(7));
  let low=vec2i(floor(local));
  let high=min(low+vec2i(1),vec2i(7));
  let fraction=local-vec2f(low);
  let tile=vec2i(
    i32(probe%64u)*8,
    i32(probe/64u)*8+i32(frameIndex)*2048
  );
  let v00=textureLoad(irradianceAtlas,tile+vec2i(low.x,low.y),0);
  let v10=textureLoad(irradianceAtlas,tile+vec2i(high.x,low.y),0);
  let v01=textureLoad(irradianceAtlas,tile+vec2i(low.x,high.y),0);
  let v11=textureLoad(irradianceAtlas,tile+high,0);
  return mix(mix(v00,v10,fraction.x),mix(v01,v11,fraction.x),fraction.y);
}
fn lodDistance(position:vec3f)->f32{
  let delta=abs(position-frame.lodCamera.xyz);
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
  let start=boundary*0.75;
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
// Directional band-limit for the current receiver, set once per fragment.
// Zero keeps the exact octahedral lookup; one converges the lookup toward the
// probe's ambient mean. It is the directional analogue of texture mip
// selection: a rigid surface only a few pixels across cannot support
// per-normal directional detail, because raster coverage re-picks the sampled
// normal every frame and the 6x6 octahedral gradient would turn that churn
// into lighting flicker on the whole object.
var<private> directionalSpread:f32=0.0;

fn tileAmbientMean(probe:u32,frameIndex:u32)->vec4f{
  // Normal-independent estimate of the probe's direction-averaged field:
  // four bilinear taps tile the 6x6 octahedral interior uniformly. Texels
  // facing into geometry legitimately carry no data; weighting by their
  // confidence keeps unpopulated directions from dragging the mean to black
  // (and their per-frame coverage churn out of the estimate).
  var value=vec3f(0);
  var confidence=0.0;
  for(var tap=0u;tap<4u;tap++){
    let coordinate=vec2f(
      select(2.5,4.5,(tap&1u)!=0u),
      select(2.5,4.5,(tap>>1u)!=0u)
    );
    let sampleValue=loadFinalAtlasIrradianceBilinear(
      probe,frameIndex,coordinate
    );
    value+=sampleValue.xyz*sampleValue.a;
    confidence+=sampleValue.a;
  }
  if(confidence<1e-4){return vec4f(0);}
  return vec4f(value/confidence,confidence*0.25);
}

fn filteredFinalAtlasSample(
  probe:u32,frameIndex:u32,octCoordinate:vec2f
)->vec4f{
  let base=loadFinalAtlasIrradianceBilinear(probe,frameIndex,octCoordinate);
  if(directionalSpread<=0.001){return base;}
  return mix(base,tileAmbientMean(probe,frameIndex),directionalSpread);
}

fn sampleIrradianceLod(position:vec3f,normal:vec3f,lod:u32)->vec4f{
  let spacing=frame.envBaseSpacing.w*exp2(f32(lod));
  let grid=position/spacing-vec3f(0.5);
  let cell=vec3i(floor(grid));
  let linearFraction=fract(grid);
  // C1 surface reconstruction. Linear weights are only C0: a rigid surface
  // moving through the probe grid crosses a cell boundary somewhere on its
  // body almost every frame, and each crossing converts the field's spatial
  // curvature into a second-difference spike of the reconstructed lighting.
  // Hermite weights have zero derivative at cell boundaries, so the same
  // sparse field is C1 along any smooth motion path — for rigid receivers,
  // for the camera, and for static geometry alike.
  let fraction=linearFraction*linearFraction*(vec3f(3.0)-2.0*linearFraction);
  let fixedBits=vec3i(floor(position/spacing))-cell;
  let absoluteNormal=abs(normal);
  var normalAxis=2u;
  if(absoluteNormal.x>=absoluteNormal.y&&absoluteNormal.x>=absoluteNormal.z){
    normalAxis=0u;
  }else if(absoluteNormal.y>=absoluteNormal.z){
    normalAxis=1u;
  }
  // Rigid receivers drop the tangent-axis machinery entirely: the dominant
  // axis and the normal-axis weight are raster-derived and flip continuously
  // on curved or tumbling instances, swapping the interpolation pattern
  // per frame. The full eight-corner trilinear needs neither — cells with no
  // probes renormalize away naturally — so the support set is a pure function
  // of the query position, and the canonical sheet key still separates the
  // instance's opposite faces.
  let rigidReceiver=rigidSheetOverride!=8u;
  var normalWeight=0.0;
  if(normalAxis==0u){normalWeight=select(1.0-fraction.x,fraction.x,fixedBits.x==1);}
  if(normalAxis==1u){normalWeight=select(1.0-fraction.y,fraction.y,fixedBits.y==1);}
  if(normalAxis==2u){normalWeight=select(1.0-fraction.z,fraction.z,fixedBits.z==1);}
  if(rigidReceiver){normalWeight=1.0;}
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
  let frameIndex=u32(floor(frame.controls.w))&3u;
  // Residency continuity is independent from radiometric source history. A
  // one-to-three-frame exact-key fallback is valid during continuous analytic
  // light motion and prevents a cold LOD from turning a lit receiver black.
  let allowHistoricalSupport=!finalFeatureEnabled(512u)
    &&finalDynamicPointHistoryValid(position,spacing*1.75);
  var value=vec3f(0);
  var total=0.0;
  let cornerCount=select(4u,8u,rigidReceiver);
  for(var corner=0u;corner<cornerCount;corner++){
    var bits=fixedBits;
    if(rigidReceiver){
      bits=vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u));
    }else if(normalAxis==0u){
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
    let key=keyFromCellSurface(cell+bits,lod,pixelSurfaceClass(normal));
    let probe=lookupProbe(key);
    var irradiance=vec4f(0);
    if(probe!=EMPTY&&probe<16384u){
      irradiance=filteredFinalAtlasSample(
        probe,frameIndex,octCoordinate
      );
    }
    if(irradiance.a<0.001&&allowHistoricalSupport){
      for(var age=1u;age<4u;age++){
        let historyFrame=(frameIndex+4u-age)&3u;
        let historyProbe=lookupProbeCascadeFrame(0u,key,historyFrame);
        if(historyProbe!=EMPTY&&historyProbe<16384u){
          let historyIrradiance=filteredFinalAtlasSample(
            historyProbe,historyFrame,octCoordinate
          );
          if(historyIrradiance.a>0.001){
            irradiance=historyIrradiance;
            break;
          }
        }
      }
    }
    if(irradiance.a>0.001){
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
  let fineLod=u32(lods.x);
  let coarseLod=u32(lods.y);
  let fine=sampleIrradianceLod(position,normal,fineLod);
  var coarse=vec4f(0);
  if(coarseLod!=fineLod){
    coarse=sampleIrradianceLod(position,normal,coarseLod);
  }
  // Search every bounded LOD for the best still-resident exact-key field.
  // This is used only while an incoming selected field is cold; it does not
  // change steady-state LOD choice or invent radiance for a disocclusion.
  var resident=vec4f(0);
  var residentDistance=5u;
  for(var candidateLod=0u;candidateLod<4u;candidateLod++){
    if(candidateLod==fineLod||candidateLod==coarseLod){continue;}
    let candidate=sampleIrradianceLod(position,normal,candidateLod);
    let distance=u32(abs(i32(candidateLod)-i32(fineLod)));
    if(candidate.w>resident.w+0.001
      ||(abs(candidate.w-resident.w)<=0.001&&distance<residentDistance)){
      resident=candidate;
      residentDistance=distance;
    }
  }
  if(coarseLod==fineLod){
    if(fine.w<0.001){return resident;}
    if(resident.w<0.001||fine.w>=0.35){return fine;}
    let readiness=smoothstep(0.02,0.35,fine.w);
    return vec4f(
      mix(resident.xyz,fine.xyz,readiness),
      mix(resident.w,fine.w,readiness)
    );
  }
  if(fine.w<0.001){
    if(coarse.w>=0.001){return coarse;}
    return resident;
  }
  if(coarse.w<0.001){
    if(resident.w<0.001||fine.w>=0.35){return fine;}
    let readiness=smoothstep(0.02,0.35,fine.w);
    return vec4f(
      mix(resident.xyz,fine.xyz,readiness),
      mix(resident.w,fine.w,readiness)
    );
  }
  return vec4f(
    mix(fine.xyz,coarse.xyz,lods.z),mix(fine.w,coarse.w,lods.z)
  );
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
  var environmentTransmittance=0.0;
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
      cell+bits,lod,pixelSurfaceClass(normal)
    ));
    if(probe!=EMPTY&&probe<16384u){
      let cone=cones[dataIndex(0u,probe,direction)];
      if(cone.w>0.5){
        value+=cone.xyz*weight;
        environmentTransmittance+=max(0.0,cone.w-1.0)*weight;
        total+=weight;
      }
    }
  }
  if(total<1e-5){return vec4f(0);}
  return vec4f(value/total,1.0+environmentTransmittance/total);
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
  return mix(fine,coarse,lods.z);
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
  let frameIndex=u32(floor(frame.controls.w))&1u;
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
  // Hardware comparison filtering plus a deterministic 3x3 tent gives a
  // smooth 6x6 effective footprint. The former 5x5 kernel could issue fifty
  // comparison samples in a cascade blend and dominated the entire Sponza
  // frame without improving temporal stability.
  for(var y=-1;y<=1;y++){
    for(var x=-1;x<=1;x++){
      let weight=(2.0-abs(f32(x)))*(2.0-abs(f32(y)));
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
fn shortQuaternionRotate(vector:vec3f,quaternion:vec4f)->vec3f{
  let doubled=2.0*cross(quaternion.xyz,vector);
  return vector+quaternion.w*doubled+cross(quaternion.xyz,doubled);
}
fn traceShortDynamicInstance(
  origin:vec3f,direction:vec3f,instanceIndex:u32,watertight:bool,best:ShortHit
)->ShortHit{
  let instance=shortTriangles[frame.dynamicInfo.y+instanceIndex];
  let inverseRotation=vec4f(-instance.b.xyz,instance.b.w);
  let localOrigin=shortQuaternionRotate(
    origin-instance.a.xyz,inverseRotation
  )/instance.c.xyz;
  let localDirection=shortQuaternionRotate(direction,inverseRotation)/instance.c.xyz;
  let inverseLocalDirection=select(
    vec3f(-1e20),vec3f(1e20),localDirection>=vec3f(0)
  )/max(vec3f(1),abs(localDirection)*1e20);
  var result=best;
  var stack:array<u32,64>;
  var stackSize=1u;
  stack[0]=instance.normalOct.x;
  loop{
    if(stackSize==0u){break;}
    stackSize-=1u;
    let node=shortBvhNodes[stack[stackSize]];
    if(shortRayBoxNear(
      localOrigin,inverseLocalDirection,node.minMeta.xyz,node.maxMeta.xyz,result.t
    )>=result.t){continue;}
    let left=bitcast<u32>(node.minMeta.w);
    let right=bitcast<u32>(node.maxMeta.w);
    if((left&0x80000000u)!=0u){
      let first=left&0x7fffffffu;
      for(var triangleOffset=0u;triangleOffset<right;triangleOffset++){
        let triangle=shortTriangles[first+triangleOffset];
        var intersection=shortTriangle(
          localOrigin,localDirection,triangle,result.t
        );
        if(watertight){
          intersection=shortTriangleWatertight(
            localOrigin,localDirection,triangle,result.t
          );
        }
        if(intersection.x<result.t){
          let barycentric=vec3f(
            1.0-intersection.y-intersection.z,intersection.y,intersection.z
          );
          let localNormal=normalize(
            shortNormal(triangle.normalOct.x)*barycentric.x
            +shortNormal(triangle.normalOct.y)*barycentric.y
            +shortNormal(triangle.normalOct.z)*barycentric.z
          );
          var worldNormal=normalize(shortQuaternionRotate(
            localNormal/instance.c.xyz,instance.b
          ));
          if(dot(worldNormal,direction)>0.0){worldNormal=-worldNormal;}
          let sourceFrontFace=dot(cross(
            triangle.b.xyz-triangle.a.xyz,
            triangle.c.xyz-triangle.a.xyz
          ),localDirection)<0.0;
          let isEmitter=(instance.normalOct.w&1u)!=0u&&sourceFrontFace;
          result=ShortHit(
            intersection.x,worldNormal,
            triangle.albedo.xyz*instance.albedo.xyz,
            select(vec3f(0),instance.emissive.xyz,isEmitter)
          );
        }
      }
    }else{
      let leftNear=shortRayBoxNear(
        localOrigin,inverseLocalDirection,shortBvhNodes[left].minMeta.xyz,
        shortBvhNodes[left].maxMeta.xyz,result.t
      );
      let rightNear=shortRayBoxNear(
        localOrigin,inverseLocalDirection,shortBvhNodes[right].minMeta.xyz,
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
fn traceShortRangeImpl(
  origin:vec3f,directionIn:vec3f,maxDistance:f32,watertight:bool,
  includeDynamic:bool
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
          let sourceFrontFace=dot(
            cross(
              triangle.b.xyz-triangle.a.xyz,
              triangle.c.xyz-triangle.a.xyz
            ),direction
          )<0.0;
          result=ShortHit(
            intersection.x,hitNormal,triangle.albedo.xyz*surface.rgb,
            select(vec3f(0),triangle.emissive.xyz,sourceFrontFace)
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
  if(includeDynamic&&frame.dynamicInfo.x!=0xffffffffu){
    var dynamicStack:array<u32,32>;
    var dynamicStackSize=1u;
    dynamicStack[0]=frame.dynamicInfo.x;
    loop{
      if(dynamicStackSize==0u){break;}
      dynamicStackSize-=1u;
      let node=shortBvhNodes[dynamicStack[dynamicStackSize]];
      if(shortRayBoxNear(
        origin,inverseDirection,node.minMeta.xyz,node.maxMeta.xyz,result.t
      )>=result.t){continue;}
      let left=bitcast<u32>(node.minMeta.w);
      let right=bitcast<u32>(node.maxMeta.w);
      if((left&0x80000000u)!=0u){
        let first=left&0x7fffffffu;
        for(var instanceOffset=0u;instanceOffset<right;instanceOffset++){
          result=traceShortDynamicInstance(
            origin,direction,first+instanceOffset,watertight,result
          );
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
          if(dynamicStackSize>29u){break;}
          if(leftNear<rightNear){
            dynamicStack[dynamicStackSize]=right;
            dynamicStack[dynamicStackSize+1u]=left;
          }else{
            dynamicStack[dynamicStackSize]=left;
            dynamicStack[dynamicStackSize+1u]=right;
          }
          dynamicStackSize+=2u;
        }else if(leftNear<result.t){
          dynamicStack[dynamicStackSize]=left;dynamicStackSize+=1u;
        }else if(rightNear<result.t){
          dynamicStack[dynamicStackSize]=right;dynamicStackSize+=1u;
        }
      }
    }
  }
  return result;
}
fn traceShortRange(origin:vec3f,direction:vec3f,maxDistance:f32)->ShortHit{
  return traceShortRangeImpl(origin,direction,maxDistance,false,true);
}
fn traceShortRangeWatertight(
  origin:vec3f,direction:vec3f,maxDistance:f32
)->ShortHit{
  return traceShortRangeImpl(origin,direction,maxDistance,true,true);
}
fn traceShortRangeWatertightStatic(
  origin:vec3f,direction:vec3f,maxDistance:f32
)->ShortHit{
  return traceShortRangeImpl(origin,direction,maxDistance,true,false);
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
  let sunCosine=max(0.0,dot(hit.normal,lightDirection));
  var sunVisibility=0.0;
  if(sunCosine>0.0){
    // C(-1) hit points are frequently off screen. Camera-frustum shadow maps
    // intentionally return visible outside their atlas, which is unsuitable
    // for indirect visibility and leaks sunlight into sealed rooms. Resolve
    // these secondary rays against the same exact, watertight scene BVH.
    if(frame.pointColorIntensity.w<=0.0001&&!finalFeatureEnabled(64u)){
      let sunOrigin=hitWorld+hit.normal*max(0.006,frame.envBaseSpacing.w*0.01);
      let sunBlocker=traceShortRangeWatertight(
        sunOrigin,lightDirection,frame.sceneBounds.w*1.001
      );
      sunVisibility=select(0.0,1.0,sunBlocker.t>=frame.sceneBounds.w*1.001);
    }else{
      sunVisibility=fastSunVisibility(hitWorld,hit.normal);
    }
  }
  let sun=frame.sunColorIntensity.xyz*frame.sunColorIntensity.w
    *sunCosine*sunVisibility;
  let toPoint=frame.pointPosRange.xyz-hitWorld;
  let distance=length(toPoint);
  let pointWindow=max(0.0,1.0-distance/frame.pointPosRange.w);
  let pointDirection=toPoint/max(distance,1e-4);
  var pointVisibility=0.0;
  if(frame.pointColorIntensity.w>0.0&&pointWindow>0.0){
    pointVisibility=pointShadowVisibility(hitWorld,hit.normal);
  }
  let point=frame.pointColorIntensity.xyz*frame.pointColorIntensity.w
    *max(0.0,dot(hit.normal,pointDirection))
    *pointWindow*pointWindow/(1.0+0.06*distance*distance)
    *pointVisibility;
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
fn emissivePatchIrradiance(
  origin:vec3f,normal:vec3f,a:vec3f,b:vec3f,c:vec3f,
  emission:vec3f
)->vec3f{
  let centroid=(a+b+c)/3.0;
  let toLight=centroid-origin;
  let lightDistance=length(toLight);
  if(lightDistance<=1e-7){return vec3f(0);}
  let patchExtent=max(length(b-a),max(length(c-b),length(a-c)));
  let formFactor=clippedTriangleFormFactor(
    a-origin,b-origin,c-origin,normal
  );
  if(formFactor<=1e-8){return vec3f(0);}
  let rayEpsilon=max(
    1e-7,min(lightDistance*0.001,patchExtent*0.001)
  );
  let traceDistance=max(1e-7,lightDistance-rayEpsilon);
  let blocker=traceShortRangeWatertight(
    origin,toLight/lightDistance,traceDistance
  );
  if(blocker.t<traceDistance){return vec3f(0);}
  return emission*formFactor;
}
fn emissiveVisibility(origin:vec3f,lightPoint:vec3f,sourceExtent:f32)->f32{
  let toLight=lightPoint-origin;
  let lightDistance=length(toLight);
  if(lightDistance<=1e-7){return 0.0;}
  let rayEpsilon=max(
    1e-7,min(lightDistance*0.001,sourceExtent*0.001)
  );
  let traceDistance=max(1e-7,lightDistance-rayEpsilon);
  let blocker=traceShortRangeWatertight(
    origin,toLight/lightDistance,traceDistance
  );
  return select(0.0,1.0,blocker.t>=traceDistance);
}
fn emissiveTriangleIrradiance(
  origin:vec3f,normal:vec3f,triangle:Triangle,radius:f32
)->vec3f{
  if(max(triangle.emissive.x,max(triangle.emissive.y,triangle.emissive.z))<=0.0){
    return vec3f(0);
  }
  let a=triangle.a.xyz;
  let b=triangle.b.xyz;
  let c=triangle.c.xyz;
  let sourceNormal=cross(b-a,c-a);
  if(dot(sourceNormal,origin-(a+b+c)/3.0)<=0.0){return vec3f(0);}
  let minimum=min(a,min(b,c));
  let maximum=max(a,max(b,c));
  let proximityDistance=sqrt(pointAabbDistanceSquared(origin,minimum,maximum));
  if(proximityDistance>=radius){return vec3f(0);}
  let intervalBlend=1.0-smoothstep(radius*0.72,radius,proximityDistance);
  let parentFormFactor=clippedTriangleFormFactor(
    a-origin,b-origin,c-origin,normal
  );
  if(parentFormFactor<=1e-8){return vec3f(0);}
  let centroid=(a+b+c)/3.0;
  let sourceExtent=max(length(b-a),max(length(c-b),length(a-c)));
  // Match the primary/reference implementation: facets contained by the
  // local interval use one stable world-space visibility sample, while broad
  // polygons retain the 8x8 partial-coverage integration below.
  if(sourceExtent<=radius*2.5){
    let visibility=emissiveVisibility(origin,centroid,sourceExtent);
    return triangle.emissive.xyz*(parentFormFactor*intervalBlend*visibility);
  }
  let samples=array<vec3f,7>(
    mix(a,centroid,0.06),mix(b,centroid,0.06),mix(c,centroid,0.06),
    (a+b)*0.5,(b+c)*0.5,(c+a)*0.5,centroid
  );
  var visibleSamples=0.0;
  for(var sampleIndex=0u;sampleIndex<7u;sampleIndex++){
    visibleSamples+=emissiveVisibility(
      origin,samples[sampleIndex],sourceExtent
    );
  }
  if(visibleSamples>=6.5){
    return triangle.emissive.xyz*(parentFormFactor*intervalBlend);
  }
  if(visibleSamples<=0.5){return vec3f(0);}
  let edgeB=(triangle.b.xyz-triangle.a.xyz)*0.125;
  let edgeC=(triangle.c.xyz-triangle.a.xyz)*0.125;
  var result=vec3f(0);
  for(var i=0u;i<8u;i++){
    for(var j=0u;j<8u-i;j++){
      let p00=triangle.a.xyz+edgeB*f32(i)+edgeC*f32(j);
      let p10=p00+edgeB;
      let p01=p00+edgeC;
      result+=emissivePatchIrradiance(
        origin,normal,p00,p10,p01,triangle.emissive.xyz
      );
      if(i+j+2u<=8u){
        let p11=p10+edgeC;
        result+=emissivePatchIrradiance(
          origin,normal,p10,p11,p01,triangle.emissive.xyz
        );
      }
    }
  }
  return result*intervalBlend;
}
fn shortInstancePoint(localPoint:vec3f,instance:Triangle)->vec3f{
  return instance.a.xyz+shortQuaternionRotate(
    localPoint*instance.c.xyz,instance.b
  );
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
  if(frame.dynamicInfo.z!=0xffffffffu&&!finalFeatureEnabled(4096u)){
    var dynamicStack:array<u32,32>;
    var dynamicStackSize=1u;
    dynamicStack[0]=frame.dynamicInfo.z;
    loop{
      if(dynamicStackSize==0u){break;}
      dynamicStackSize-=1u;
      let node=shortBvhNodes[dynamicStack[dynamicStackSize]];
      if(pointAabbDistanceSquared(
        origin,node.minMeta.xyz,node.maxMeta.xyz
      )>radiusSquared){continue;}
      let left=bitcast<u32>(node.minMeta.w);
      let right=bitcast<u32>(node.maxMeta.w);
      if((left&0x80000000u)!=0u){
        let firstInstance=left&0x7fffffffu;
        for(var instanceOffset=0u;instanceOffset<right;instanceOffset++){
          let instance=shortTriangles[
            frame.dynamicInfo.y+firstInstance+instanceOffset
          ];
          if((instance.normalOct.w&1u)==0u){continue;}
          // The TLAS leaf identifies an emissive instance; traverse that
          // asset's reusable BLAS in local space instead of brute-forcing all
          // of its triangles for every receiver pixel. Radius/minScale is a
          // conservative enclosing sphere under non-uniform scale, so this
          // cannot cull a contributing world-space triangle.
          let inverseRotation=vec4f(-instance.b.xyz,instance.b.w);
          let localOrigin=shortQuaternionRotate(
            origin-instance.a.xyz,inverseRotation
          )/max(instance.c.xyz,vec3f(1e-6));
          let minimumScale=max(1e-6,min(
            abs(instance.c.x),min(abs(instance.c.y),abs(instance.c.z))
          ));
          let localRadius=radius/minimumScale;
          let localRadiusSquared=localRadius*localRadius;
          var blasStack:array<u32,64>;
          var blasStackSize=1u;
          blasStack[0]=instance.normalOct.x;
          loop{
            if(blasStackSize==0u){break;}
            blasStackSize-=1u;
            let blasNode=shortBvhNodes[blasStack[blasStackSize]];
            if(pointAabbDistanceSquared(
              localOrigin,blasNode.minMeta.xyz,blasNode.maxMeta.xyz
            )>localRadiusSquared){continue;}
            let blasLeft=bitcast<u32>(blasNode.minMeta.w);
            let blasRight=bitcast<u32>(blasNode.maxMeta.w);
            if((blasLeft&0x80000000u)!=0u){
              let firstTriangle=blasLeft&0x7fffffffu;
              for(var triangleOffset=0u;triangleOffset<blasRight;triangleOffset++){
                let localTriangle=shortTriangles[firstTriangle+triangleOffset];
                let worldTriangle=Triangle(
                  vec4f(shortInstancePoint(localTriangle.a.xyz,instance),0),
                  vec4f(shortInstancePoint(localTriangle.b.xyz,instance),0),
                  vec4f(shortInstancePoint(localTriangle.c.xyz,instance),0),
                  vec4f(0),vec4f(instance.emissive.xyz,0),
                  vec4f(0),vec4f(0),vec4u(0)
                );
                result+=emissiveTriangleIrradiance(
                  origin,normal,worldTriangle,radius
                );
              }
            }else{
              if(blasStackSize>61u){break;}
              blasStack[blasStackSize]=blasLeft;
              blasStack[blasStackSize+1u]=blasRight;
              blasStackSize+=2u;
            }
          }
        }
      }else{
        if(dynamicStackSize>29u){break;}
        dynamicStack[dynamicStackSize]=left;
        dynamicStack[dynamicStackSize+1u]=right;
        dynamicStackSize+=2u;
      }
    }
  }
  return result;
}

// A 14-point Lebedev rule (the six axes plus the eight cube corners) is a
// deterministic, rotation-balanced quadrature for the ambient C(-1) term.
// It avoids both screen-space visibility changes and the directional fans of
// returning one sparse ray as outgoing radiance.  The final normalization
// below makes the rule energy preserving for every receiver normal.
const C_MINUS_DIRECTIONS=array<vec3f,14>(
  vec3f( 1.0, 0.0, 0.0),vec3f(-1.0, 0.0, 0.0),
  vec3f( 0.0, 1.0, 0.0),vec3f( 0.0,-1.0, 0.0),
  vec3f( 0.0, 0.0, 1.0),vec3f( 0.0, 0.0,-1.0),
  vec3f( 0.577350269, 0.577350269, 0.577350269),
  vec3f( 0.577350269, 0.577350269,-0.577350269),
  vec3f( 0.577350269,-0.577350269, 0.577350269),
  vec3f( 0.577350269,-0.577350269,-0.577350269),
  vec3f(-0.577350269, 0.577350269, 0.577350269),
  vec3f(-0.577350269, 0.577350269,-0.577350269),
  vec3f(-0.577350269,-0.577350269, 0.577350269),
  vec3f(-0.577350269,-0.577350269,-0.577350269)
);

fn cMinusOneIrradiance(
  world:vec3f,normal:vec3f,baseIrradiance:vec3f,
  closedBackFace:bool
)->vec3f{
  // Section 7.1 defines C(-1) as the per-pixel interval from the surface to
  // t_-1 (normally delta_s0), followed by the directional c0 interval.  Do
  // that merge independently for every direction.  The former six-axis
  // enclosure classifier and "any miss" fallback switched an entire pixel
  // between unrelated estimators; box gaps therefore produced the large
  // black/white wedges reported in Cornell, and open rooms skipped C(-1)
  // altogether.  A miss now continues into the matching world-space c0 cone.
  let intervalEnd=frame.envBaseSpacing.w;
  let origin=world+normal*max(0.006,intervalEnd*0.012);
  let nearEmission=nearEmissiveIrradiance(origin,normal,intervalEnd);
  // Run the same directional estimator at every pixel. In particular, do
  // not key C(-1) activation to neighboring screen pixels: that would make a
  // receiver change estimators as silhouettes move through the G-buffer.
  var ambientWeight=0.0;
  var ambientVisible=0.0;
  // Open receivers retain the smooth paper cascade field. The exact ambient
  // visibility resolve is reserved for topology-classified enclosures and
  // explicitly closed-mesh back faces, where it prevents real environment
  // leakage rather than adding a sparse high-frequency AO term.
  if(!closedBackFace){
    return baseIrradiance+nearEmission;
  }
  let visibilityGuardEnd=max(intervalEnd,frame.sceneBounds.w*1.001);
  for(var directionIndexValue=0u;directionIndexValue<14u;directionIndexValue++){
    let direction=C_MINUS_DIRECTIONS[directionIndexValue];
    let cosine=max(0.0,dot(normal,direction));
    if(cosine<=0.0){continue;}
    ambientWeight+=cosine;
    let shortHit=traceShortRangeWatertight(origin,direction,intervalEnd+0.001);
    // A c0 direction represents a finite solid-angle cone, not an infinitesimal
    // binary ray. Filter the blocker distance across the complete C(-1)
    // interval as a cone-footprint estimate. This removes hard AO contours as
    // either the receiver or blocker crosses t_-1, without screen coordinates,
    // random noise, or a scene-dependent threshold.
    var visibility=smoothstep(0.0,intervalEnd,shortHit.t);
    if(visibility>0.0){
      let farHit=traceShortRangeWatertight(
        origin,direction,visibilityGuardEnd+0.001
      );
      visibility*=select(0.0,1.0,farHit.t>=visibilityGuardEnd);
    }
    ambientVisible+=cosine*visibility;
  }
  // The paper calls out this ambient-term form as the practical C(-1)
  // optimization. One energy correction preserves the smooth directional c0
  // result while exact short rays recover fine occlusion. It cannot create
  // bright fan sectors because validated energy is bounded by its cone.
  let visibilityCorrection=select(
    1.0,clamp(ambientVisible/ambientWeight,0.0,1.0),
    ambientWeight>1e-7
  );
  return baseIrradiance*visibilityCorrection+nearEmission;
}

@vertex fn fullscreenVS(@builtin(vertex_index) index:u32)->@builtin(position) vec4f{
  let uv=vec2f(f32((index<<1u)&2u),f32(index&2u));
  return vec4f(uv*2.0-1.0,0.0,1.0);
}
// (owner-guarded 5x5 filtering lives in the temporal pass; the unused
// final-pass variant was removed with the material-node estimator.)
fn finalPackedOwner(marker:f32)->u32{
  let code=u32(marker+0.5);
  if((code&0x800000u)==0u){return 0xffffffffu;}
  return (code>>14u)&63u;
}
fn finalClosedSurface(marker:f32)->bool{
  return (u32(marker+0.5)&1u)!=0u;
}
fn exactPrimarySunVisibility(world:vec3f,normal:vec3f,direction:vec3f)->f32{
  // Shadow-map depth bias and PCF can classify a thin jamb or moving door as
  // visible even when the current scene BVH is blocked. Primary direct light
  // uses the same watertight static+dynamic traversal as GI endpoints; twice
  // the scene radius reaches the opposite side from every receiver.
  let maxDistance=max(1.0,frame.sceneBounds.w*2.05);
  let origin=world+normal*max(0.006,frame.envBaseSpacing.w*0.01);
  let blocker=traceShortRangeWatertight(origin,direction,maxDistance);
  return select(0.0,1.0,blocker.t>=maxDistance);
}
struct CurrentLightingOut{
  // Linear irradiance is the only temporally reconstructed quantity.
  @location(0) irradiance:vec4f,
  // Direct/emission remains current. Debug modes use this as a display-ready
  // bypass so temporal reconstruction cannot alter diagnostic truth.
  @location(1) directOrDebug:vec4f,
};
@fragment fn finalFS(@builtin(position) position:vec4f)->CurrentLightingOut{
  let pixel=vec2i(position.xy);
  let world=textureLoad(worldTex,pixel,0);
  let uv=position.xy/frame.resolution.xy;
  if(world.w<0.5){
    let sky=mix(frame.envBaseSpacing.xyz*0.35,frame.envBaseSpacing.xyz*1.5,clamp(1.0-uv.y,0.0,1.0));
    return CurrentLightingOut(
      vec4f(0),vec4f(displayEncode(sky*frame.controls.y),1.0)
    );
  }
  let albedoData=textureLoad(albedoTex,pixel,0);
  let albedo=albedoData.xyz;
  let normal=gbufferNormal(pixel);
  let mode=u32(frame.controls.z+0.5);
  let dynamicOwner=finalPackedOwner(world.w);
  // Rigid receivers reconstruct from the same sparse world field as static
  // surfaces. Motion is handled where it belongs: cones whose support
  // overlaps the swept-change TLAS reject history and re-resolve from the
  // deterministic anchored current estimate, so a moving surface reads a
  // smooth current-state field and a stationary one reads the converged
  // accumulation — with no separate receiver estimator to switch to.
  setFinalRigidSheetOverride(dynamicOwner,normal);
  directionalSpread=0.0;
  var sample=vec4f(0);
  if(dynamicOwner!=0xffffffffu){
    // Band-limit the reconstruction to the instance's PROJECTED size. On an
    // instance a few pixels across, rasterization re-picks the sampled
    // surface point and normal every frame, so neither the per-point spatial
    // gather nor the per-normal octahedral lookup is a stable function of the
    // object's pose. As the projection shrinks, blend the per-pixel
    // directional sample toward the ambient (direction-averaged) field
    // evaluated at the instance center: the same sparse field, read at the
    // scale it can actually support — the reconstruction analogue of mip
    // selection, not a separate estimator. The limit derives only from the
    // rigid transform and the camera, so it is C1 along any smooth motion.
    let instance=shortTriangles[frame.dynamicInfo.y+dynamicOwner];
    let objectRadius=2.0*max(instance.c.x,max(instance.c.y,instance.c.z));
    let viewDistance=max(0.05,length(world.xyz-frame.cameraPos.xyz));
    let projectedRadiusPixels=objectRadius/viewDistance*frame.lodCamera.w;
    directionalSpread=1.0-smoothstep(6.0,26.0,projectedRadiusPixels);
    sample=sampleIrradiance(world.xyz,normal);
  }else{
    sample=sampleIrradiance(world.xyz,normal);
  }
  if(mode==3u){
    return CurrentLightingOut(vec4f(0),vec4f(normal*0.5+0.5,1.0));
  }
  if(mode==4u){
    // Coverage is a validity predicate. Fractional cosine weight is estimator
    // confidence and naturally changes as directional bins accumulate;
    // displaying that weight as coverage falsely suggested validity toggles.
    let covered=select(0.0,1.0,sample.w>0.001);
    return CurrentLightingOut(
      vec4f(0),
      vec4f(mix(vec3f(0.35,0.025,0.015),vec3f(0.05,1.0,0.55),covered),1.0)
    );
  }
  if(mode==5u){return CurrentLightingOut(vec4f(0),vec4f(albedo,1.0));}
  // Test-only mode 7 isolates the pre-C(-1) cascade gather and deliberately
  // returns before the short-interval resolve so its timing is independently
  // measurable.
  if(mode==7u){
    return CurrentLightingOut(
      vec4f(0),vec4f(displayEncode(sample.xyz*frame.controls.y),1.0)
    );
  }
  let resolvedIrradiance=cMinusOneIrradiance(
    world.xyz,normal,sample.xyz,finalClosedSurface(world.w)
  );
  let L=normalize(-frame.sunDirTime.xyz);
  let sunCosine=max(0.0,dot(normal,L));
  let sunVisibility=select(
    0.0,exactPrimarySunVisibility(world.xyz,normal,L),sunCosine>0.0
  );
  let sun=albedo*frame.sunColorIntensity.xyz*frame.sunColorIntensity.w
    *sunCosine*sunVisibility;
  let toPoint=frame.pointPosRange.xyz-world.xyz;
  let distance=length(toPoint);
  let pointWindow=max(0.0,1.0-distance/frame.pointPosRange.w);
  let point=albedo*frame.pointColorIntensity.xyz*frame.pointColorIntensity.w*max(0.0,dot(normal,toPoint/max(distance,1e-4)))*pointWindow*pointWindow/(1.0+0.06*distance*distance)*pointShadowVisibility(world.xyz,normal);
  let emissive=surfaceEmission(pixel);
  let direct=sun+point+emissive;
  // Negative confidence is a conservative local transport hazard. The
  // temporal pass then uses the deterministic current estimate immediately
  // rather than retaining radiance beside a swept mover.
  // A mover necessarily overlaps its own swept hierarchy. Exact owner-local
  // reprojection in the next pass is the correct validity test for that
  // receiver; the local swept guard applies only to immutable receivers.
  let safeHistory=finalPackedOwner(world.w)!=0xffffffffu
    ||finalDynamicPointHistoryValid(
      world.xyz,max(0.02,frame.envBaseSpacing.w*2.25)
    );
  let signedConfidence=select(-sample.w,sample.w,safeHistory);
  return CurrentLightingOut(
    vec4f(max(resolvedIrradiance,vec3f(0)),signedConfidence),
    vec4f(max(direct,vec3f(0)),1.0)
  );
}
`;

// Motion-aware reconstruction is an explicitly separate dynamic-geometry
// extension. The paper's world-space radiance-cascade history remains the
// estimator; this pass stabilizes only its linear indirect irradiance and
// never filters current direct light or visible emission.
export const temporalShader = /* wgsl */`
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
  sceneBounds: vec4f,
  dynamicInfo: vec4u,
  previousViewProj: mat4x4<f32>,
  lodCamera: vec4f,
};
struct Triangle {
  a:vec4f, b:vec4f, c:vec4f, albedo:vec4f, emissive:vec4f,
  uvAB:vec4f, uvCMaterial:vec4f, normalOct:vec4u,
};
@group(0) @binding(0) var<uniform> frame:FrameUniforms;
@group(0) @binding(1) var currentIrradiance:texture_2d<f32>;
@group(0) @binding(2) var currentDirect:texture_2d<f32>;
@group(0) @binding(3) var currentWorld:texture_2d<f32>;
@group(0) @binding(4) var currentNormal:texture_2d<f32>;
@group(0) @binding(5) var previousIrradiance:texture_2d<f32>;
@group(0) @binding(6) var previousWorld:texture_2d<f32>;
@group(0) @binding(7) var previousNormal:texture_2d<f32>;
@group(0) @binding(8) var currentAlbedo:texture_2d<f32>;
@group(0) @binding(9) var<storage,read> instanceRecords:array<Triangle>;

fn feature(bit:u32)->bool{return (u32(frame.cameraPos.w+0.5)&bit)!=0u;}
fn decodeNormal(texture:texture_2d<f32>,pixel:vec2i)->vec3f{
  let oct=textureLoad(texture,pixel,0).xy;
  var normal=vec3f(oct,1.0-abs(oct.x)-abs(oct.y));
  if(normal.z<0.0){
    let old=normal.xy;
    normal.x=(1.0-abs(old.y))*select(-1.0,1.0,old.x>=0.0);
    normal.y=(1.0-abs(old.x))*select(-1.0,1.0,old.y>=0.0);
  }
  return normalize(normal);
}
fn rotate(vector:vec3f,q:vec4f)->vec3f{
  let doubled=2.0*cross(q.xyz,vector);
  return vector+q.w*doubled+cross(q.xyz,doubled);
}
fn owner(marker:f32)->u32{
  let code=u32(marker+0.5);
  if((code&0x800000u)==0u){return 0xffffffffu;}
  return (code>>14u)&63u;
}
fn sameSide(a:f32,b:f32)->bool{
  return (u32(a+0.5)&1u)==(u32(b+0.5)&1u);
}
fn samePrimitive(a:f32,b:f32)->bool{
  return (u32(a+0.5)>>2u)==(u32(b+0.5)>>2u);
}
fn compatibleTemporalSurface(a:f32,b:f32)->bool{
  if(!sameSide(a,b)){return false;}
  let ownerA=owner(a);
  let ownerB=owner(b);
  if(ownerA!=0xffffffffu||ownerB!=0xffffffffu){
    // A rigid owner is the topology identity. Tight expected-world and normal
    // tests below provide the material-point identity across tessellation
    // edges, where requiring the same raster primitive would reject almost all
    // history on a moving sphere or detailed production mesh.
    return ownerA!=0xffffffffu&&ownerA==ownerB;
  }
  return samePrimitive(a,b);
}
fn compatibleTemporalHistorySurface(a:f32,b:f32)->bool{
  if(!compatibleTemporalSurface(a,b)){return false;}
  // Owner identity permits surface-aware spatial filtering, but temporal
  // bilinear taps must represent the same material triangle. Otherwise a
  // small rotating sphere/torus can pull last frame's irradiance from an
  // adjacent face and turn subpixel raster motion into a bright flash.
  return samePrimitive(a,b);
}
fn expectedPrevious(world:vec3f,normal:vec3f,marker:f32)->array<vec3f,2>{
  let id=owner(marker);
  if(id==0xffffffffu){return array<vec3f,2>(world,normal);}
  let current=instanceRecords[frame.dynamicInfo.y+id];
  let previous=instanceRecords[frame.dynamicInfo.y+64u+id];
  let local=rotate(world-current.a.xyz,vec4f(-current.b.xyz,current.b.w))
    /max(current.c.xyz,vec3f(1e-6));
  let localNormal=rotate(normal,vec4f(-current.b.xyz,current.b.w))*current.c.xyz;
  let oldWorld=previous.a.xyz+rotate(local*previous.c.xyz,previous.b);
  let oldNormal=normalize(rotate(
    localNormal/max(previous.c.xyz,vec3f(1e-6)),previous.b
  ));
  return array<vec3f,2>(oldWorld,oldNormal);
}
fn sameCurrentSurface(
  centerWorld:vec4f,centerNormal:vec3f,pixel:vec2i,normalFloor:f32
)->bool{
  let sampleWorld=textureLoad(currentWorld,pixel,0);
  if(sampleWorld.w<0.5
    ||!compatibleTemporalSurface(sampleWorld.w,centerWorld.w)){return false;}
  let sampleNormal=decodeNormal(currentNormal,pixel);
  if(dot(sampleNormal,centerNormal)<normalFloor){return false;}
  let delta=sampleWorld.xyz-centerWorld.xyz;
  let radius=max(0.02,frame.envBaseSpacing.w*1.4);
  // The plane guard is meaningful only at directional-detail scales. When the
  // receiver's footprint approaches the probe spacing, its whole body is one
  // material patch at this resolution and curvature must not reject support.
  let planeLimit=mix(
    max(0.006,frame.envBaseSpacing.w*0.04),
    radius,
    clamp(1.0-(normalFloor-0.55)/0.41,0.0,1.0)
  );
  return dot(delta,delta)<=radius*radius
    &&abs(dot(delta,centerNormal))<=planeLimit;
}
fn validPreviousTap(
  pixel:vec2i,expectedWorld:vec3f,expectedNormal:vec3f,currentMarker:f32,
  worldTolerance:f32
)->bool{
  let dimensions=vec2i(textureDimensions(previousWorld));
  if(any(pixel<vec2i(0))||any(pixel>=dimensions)){return false;}
  let oldWorld=textureLoad(previousWorld,pixel,0);
  if(oldWorld.w<0.5
    ||!compatibleTemporalHistorySurface(oldWorld.w,currentMarker)){return false;}
  let delta=oldWorld.xyz-expectedWorld;
  if(dot(delta,delta)>worldTolerance*worldTolerance){return false;}
  let normalThreshold=select(0.98,0.96,owner(currentMarker)!=0xffffffffu);
  return dot(decodeNormal(previousNormal,pixel),expectedNormal)>=normalThreshold;
}
fn displayEncode(color:vec3f)->vec3f{
  let mapped=max(color,vec3f(0));
  let filmic=(mapped*(2.51*mapped+0.03))/(mapped*(2.43*mapped+0.59)+0.14);
  return pow(clamp(filmic,vec3f(0),vec3f(1)),vec3f(1.0/2.2));
}
@vertex fn temporalVS(@builtin(vertex_index) index:u32)->@builtin(position) vec4f{
  let uv=vec2f(f32((index<<1u)&2u),f32(index&2u));
  return vec4f(uv*2.0-1.0,0.0,1.0);
}
struct TemporalOut{
  @location(0) composite:vec4f,
  @location(1) irradiance:vec4f,
};
@fragment fn temporalFS(@builtin(position) position:vec4f)->TemporalOut{
  let pixel=vec2i(position.xy);
  let dimensions=vec2i(textureDimensions(currentWorld));
  let world=textureLoad(currentWorld,pixel,0);
  let directOrDebug=textureLoad(currentDirect,pixel,0).xyz;
  if(world.w<0.5){return TemporalOut(vec4f(directOrDebug,1),vec4f(0));}
  let mode=u32(frame.controls.z+0.5);
  let currentSample=textureLoad(currentIrradiance,pixel,0);
  let normal=decodeNormal(currentNormal,pixel);
  let currentOwner=owner(world.w);
  let dynamicReceiver=currentOwner!=0xffffffffu;
  // Static receivers preserve the paper-path cross reconstruction. Rigid
  // receivers need a wider surface-bilateral footprint because the exact
  // material point moves continuously while raster pixel centers change
  // discretely; indirect irradiance is low-frequency, but edges remain guarded
  // by sameCurrentSurface's owner, side, normal, world, and plane tests.
  let dynamicSpatial=currentOwner!=0xffffffffu;
  // Footprint-adaptive support: when a rigid receiver's projected material
  // footprint approaches the probe spacing, its whole body is one material
  // patch at this sampling rate, so curvature (normal) differences between
  // taps are sub-resolution detail and must not shrink the filter support.
  var spatialNormalFloor=0.96;
  var smallProjection=false;
  if(dynamicSpatial){
    // Pose-smooth band limit by the instance's projected size (see finalFS):
    // raster-neighbor footprints are population-dependent and flicker on
    // instances a few pixels across.
    let instance=instanceRecords[frame.dynamicInfo.y+currentOwner];
    let objectRadius=2.0*max(instance.c.x,max(instance.c.y,instance.c.z));
    let viewDistance=max(0.05,length(world.xyz-frame.cameraPos.xyz));
    let projectedRadiusPixels=objectRadius/viewDistance*frame.lodCamera.w;
    spatialNormalFloor=mix(
      0.96,0.55,
      1.0-smoothstep(6.0,26.0,projectedRadiusPixels)
    );
    // Below the sampling limit the whole instance is one displayed feature;
    // averaging every one of its pixels is the only stable estimate raster
    // can express.
    smallProjection=projectedRadiusPixels<16.0;
  }
  let centerWeight=select(3.0,1.0,dynamicSpatial);
  var currentSum=max(currentSample.xyz,vec3f(0))*centerWeight;
  var currentWeight=centerWeight;
  var neighborhoodMin=max(currentSample.xyz,vec3f(0));
  var neighborhoodMax=neighborhoodMin;
  let tapRadius=select(2,3,smallProjection);
  for(var y=-3;y<=3;y++){
    for(var x=-3;x<=3;x++){
      if(x==0&&y==0){continue;}
      if(abs(x)>tapRadius||abs(y)>tapRadius){continue;}
      if(!dynamicSpatial&&(abs(x)>1||abs(y)>1||abs(x)+abs(y)>1)){continue;}
      let tap=clamp(pixel+vec2i(x,y),vec2i(0),dimensions-vec2i(1));
      if(!sameCurrentSurface(world,normal,tap,spatialNormalFloor)){continue;}
      let value=max(textureLoad(currentIrradiance,tap,0).xyz,vec3f(0));
      neighborhoodMin=min(neighborhoodMin,value);
      neighborhoodMax=max(neighborhoodMax,value);
      let weight=exp(select(-0.5,select(-0.32,-0.18,smallProjection),dynamicSpatial)*f32(x*x+y*y));
      currentSum+=value*weight;
      currentWeight+=weight;
    }
  }
  // Rigid receivers read the shared sparse field, whose per-pixel gather has
  // per-frame variance where motion rejects history. The wider owner-guarded
  // 5x5 world-bilateral loop above is a pure current-frame filter: owner,
  // side, normal, world, and plane tests keep it from crossing silhouettes,
  // so it removes sparse-ray variance without any temporal state.
  let current=currentSum/currentWeight;
  var history=vec3f(0);
  var historyWeight=0.0;
  // Negative confidence is the explicit swept-transport hazard produced by
  // finalFS. Source changes are globally influential, so neither condition may
  // reuse old screen irradiance even when the visible surface reprojects.
  // Every immutable receiver uses only the authoritative paper-path world-
  // cone result: recursive screen history could hide a camera-dependent field
  // defect or trail a blocker. A rigid
  // receiver is different—its exact owner-local transform provides material-
  // point reprojection. Reuse is still rejected by owner, side, expected
  // world, normal, teleport, source-step, and neighborhood tests below.
  let expected=expectedPrevious(world.xyz,normal,world.w);
  // A discontinuity belongs to the receiver that made it.  The CPU flag is a
  // conservative scene-wide signal (one large mover can raise it for every
  // owner), so using it here discarded exact reprojection on unrelated rigid
  // objects.  Measure this material point against its own previous transform;
  // ordinary rigid motion remains valid while a true owner jump is rejected.
  let ownerDiscontinuity=dynamicReceiver&&feature(2048u)
    &&length(world.xyz-expected[0])>max(0.025,frame.envBaseSpacing.w*0.25);
  // A dynamic result must be a pure function of the current scene. Previous
  // screen lighting is never consulted: this prevents lag, ghost trails, and
  // stale light when a hidden material point becomes visible again.
  let temporalValid=false;
  let baseTolerance=max(0.002,frame.envBaseSpacing.w*0.02);
  var projectedFootprint=0.0;
  var historyWorldTolerance=baseTolerance;
  if(currentOwner!=0xffffffffu){
    let footprintX=textureLoad(currentWorld,clamp(
      pixel+vec2i(1,0),vec2i(0),dimensions-vec2i(1)
    ),0);
    let footprintY=textureLoad(currentWorld,clamp(
      pixel+vec2i(0,1),vec2i(0),dimensions-vec2i(1)
    ),0);
    projectedFootprint=max(
      select(0.0,length(footprintX.xyz-world.xyz),owner(footprintX.w)==currentOwner),
      select(0.0,length(footprintY.xyz-world.xyz),owner(footprintY.w)==currentOwner)
    );
    // A subpixel material point is surrounded by four previous pixel-center
    // samples, so a fixed sub-pixel world epsilon rejects valid history as
    // soon as a curved mover rotates. Derivatives provide the projected
    // material footprint; owner, side, and 0.98 normal tests still prevent
    // this adaptive tolerance from crossing a silhouette or thin wall.
    historyWorldTolerance=max(
      baseTolerance,
        min(frame.envBaseSpacing.w,projectedFootprint*2.25)
    );
  }
  if(temporalValid){
    let clip=frame.previousViewProj*vec4f(expected[0],1);
    if(clip.w>1e-6){
      let ndc=clip.xy/clip.w;
      let previousPixel=vec2f(
        (ndc.x*0.5+0.5)*f32(dimensions.x),
        (0.5-ndc.y*0.5)*f32(dimensions.y)
      )-vec2f(0.5);
      if(dynamicReceiver){
        // Reconstruct a rigid material point from a small world-bilateral
        // footprint. Four bilinear pixel centers are discontinuous when a
        // subpixel rotating primitive gains or loses one covered tap; the
        // object-local/world/normal/primitive gates make this wider gather a
        // stable approximation of the same low-frequency material irradiance.
        let center=vec2i(floor(previousPixel+vec2f(0.5)));
        for(var oy=-2;oy<=2;oy++){
          for(var ox=-2;ox<=2;ox++){
            let tap=center+vec2i(ox,oy);
            if(!validPreviousTap(
              tap,expected[0],expected[1],world.w,historyWorldTolerance
            )){continue;}
            let oldWorld=textureLoad(previousWorld,tap,0).xyz;
            let worldRatio=dot(oldWorld-expected[0],oldWorld-expected[0])
              /max(1e-8,historyWorldTolerance*historyWorldTolerance);
            let screenDelta=vec2f(tap)+vec2f(0.5)-previousPixel;
            let weight=exp(-0.32*dot(screenDelta,screenDelta)-1.25*worldRatio);
            history+=max(textureLoad(previousIrradiance,tap,0).xyz,vec3f(0))*weight;
            historyWeight+=weight;
          }
        }
      }else{
        let base=vec2i(floor(previousPixel));
        let fraction=fract(previousPixel);
        for(var tapIndex=0u;tapIndex<4u;tapIndex++){
          let offset=vec2i(i32(tapIndex&1u),i32((tapIndex>>1u)&1u));
          let tap=base+offset;
          if(!validPreviousTap(
            tap,expected[0],expected[1],world.w,historyWorldTolerance
          )){continue;}
          let weight=select(1.0-fraction.x,fraction.x,offset.x==1)
            *select(1.0-fraction.y,fraction.y,offset.y==1);
          history+=max(textureLoad(previousIrradiance,tap,0).xyz,vec3f(0))*weight;
          historyWeight+=weight;
        }
      }
    }
  }
  var resolved=current;
  var appliedHistoryBlend=0.0;
  let requiredHistoryWeight=select(0.999,0.45,currentOwner!=0xffffffffu);
  if(historyWeight>requiredHistoryWeight){
    history/=historyWeight;
    let span=neighborhoodMax-neighborhoodMin;
    let clipExpansion=select(0.25,2.0,dynamicReceiver);
    let clipped=clamp(
      history,neighborhoodMin-span*clipExpansion,neighborhoodMax+span*clipExpansion
    );
    let currentLength=max(0.02,length(current));
    let residual=length(clipped-current)/currentLength;
    var blend=0.95*(1.0-smoothstep(0.25,1.0,residual));
    if(dynamicReceiver){
      // Exact owner-local correspondence is materially stronger than a
      // screen-space residual. A sparse-current outlier must not disable the
      // very filter intended to remove it. Current-neighborhood clipping above
      // bounds the reused value, and every topology/source/disocclusion test
      // has already run, so keep a stable EMA floor for rigid material points.
      // Continuously changing sources use a more responsive floor.
      let sourceMoving=feature(16384u)||feature(4096u);
      blend=max(blend,select(0.68,0.90,!sourceMoving));
    }
    // Actual continuously moving sources need a stable estimator, not a UI
    // checkbox-dependent cap. Topology, owner, disocclusion, and neighborhood
    // clipping have already validated this sample, so retain a responsive but
    // high-history floor that suppresses per-frame ray-population noise.
    if(feature(16384u)||feature(4096u)){blend=max(blend,0.96);}
    resolved=mix(current,max(clipped,vec3f(0)),blend);
    appliedHistoryBlend=blend;
  }
  // A sealed current neighborhood is authoritative and exactly cancels any
  // bright history. This is both a leak guard and the closed-door invariant.
  if(max(neighborhoodMax.x,max(neighborhoodMax.y,neighborhoodMax.z))<1e-7){
    resolved=vec3f(0);
    appliedHistoryBlend=0.0;
  }
  let albedo=textureLoad(currentAlbedo,pixel,0).xyz;
  var color=directOrDebug+albedo*resolved*frame.controls.x;
  if(mode==1u){color=albedo*resolved*frame.controls.x;}
  if(mode==2u){color=directOrDebug;}
  if(mode==6u){color=resolved;}
  if(mode<=2u||mode==6u){color=displayEncode(color*frame.controls.y);}
  if(mode==3u||mode==4u||mode==5u||mode==7u){color=directOrDebug;}
  // Alpha is audit-only metadata; history consumes RGB exclusively. Keeping
  // the actually applied blend makes temporal acceptance measurable without
  // shader atomics or a production hot-path readback.
  return TemporalOut(vec4f(color,1),vec4f(resolved,appliedHistoryBlend));
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
