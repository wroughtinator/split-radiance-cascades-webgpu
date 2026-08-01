import {
  buildBVH,
  buildEmissiveBVHFromPacked,
  cross3,
  normalize3,
  sub3,
  TAU,
} from "./math.js?v=2026-07-31-daylight-door5";

const C = {
  chalk: [0.72, 0.75, 0.72],
  white: [0.82, 0.84, 0.8],
  red: [0.72, 0.075, 0.045],
  green: [0.055, 0.5, 0.18],
  blue: [0.055, 0.2, 0.68],
  yellow: [0.9, 0.56, 0.08],
  orange: [0.9, 0.2, 0.035],
  cyan: [0.04, 0.65, 0.72],
  violet: [0.42, 0.08, 0.7],
  stone: [0.43, 0.39, 0.32],
  sand: [0.58, 0.42, 0.24],
  concrete: [0.32, 0.34, 0.35],
  dark: [0.055, 0.065, 0.07],
  metal: [0.24, 0.28, 0.3],
  leaf: [0.035, 0.25, 0.07],
  bark: [0.22, 0.095, 0.035],
};

export class Geometry {
  constructor() {
    this.vertices = [];
    this.triangles = [];
    this.boundsMin = [Infinity, Infinity, Infinity];
    this.boundsMax = [-Infinity, -Infinity, -Infinity];
  }

  vertex(p, n, albedo, emissive, closedSurface = false) {
    // materialCutoff.y is otherwise zero for procedural opaque geometry.
    // Its negative sign is stable topology metadata consumed by the raster
    // pass; it is never inferred from the camera-facing state.
    this.vertices.push(...p, ...n, ...albedo, ...emissive, 0, 0, -1, closedSurface ? -1 : 0);
    for (let i = 0; i < 3; i++) {
      this.boundsMin[i] = Math.min(this.boundsMin[i], p[i]);
      this.boundsMax[i] = Math.max(this.boundsMax[i], p[i]);
    }
  }

  triangle(a, b, c, albedo = C.white, emissive = [0, 0, 0], normals, closedSurface = false) {
    const face = normalize3(cross3(sub3(b, a), sub3(c, a)));
    const ns = normals || [face, face, face];
    this.vertex(a, ns[0], albedo, emissive, closedSurface);
    this.vertex(b, ns[1], albedo, emissive, closedSurface);
    this.vertex(c, ns[2], albedo, emissive, closedSurface);
    this.triangles.push({ a, b, c, albedo, emissive, normals: ns });
  }

  quad(a, b, c, d, color = C.white, emissive = [0, 0, 0], closedSurface = false) {
    this.triangle(a, b, c, color, emissive, undefined, closedSurface);
    this.triangle(a, c, d, color, emissive, undefined, closedSurface);
  }

  box(center, size, color = C.white, emissive = [0, 0, 0]) {
    const [x, y, z] = center, [sx, sy, sz] = size.map((v) => v * 0.5);
    const p = [
      [x-sx,y-sy,z-sz],[x+sx,y-sy,z-sz],[x+sx,y+sy,z-sz],[x-sx,y+sy,z-sz],
      [x-sx,y-sy,z+sz],[x+sx,y-sy,z+sz],[x+sx,y+sy,z+sz],[x-sx,y+sy,z+sz],
    ];
    this.quad(p[1],p[0],p[3],p[2],color,emissive,true);
    this.quad(p[4],p[5],p[6],p[7],color,emissive,true);
    this.quad(p[0],p[4],p[7],p[3],color,emissive,true);
    this.quad(p[5],p[1],p[2],p[6],color,emissive,true);
    this.quad(p[3],p[7],p[6],p[2],color,emissive,true);
    this.quad(p[0],p[1],p[5],p[4],color,emissive,true);
  }

  boxRotatedY(center, size, angle, color = C.white, emissive = [0, 0, 0]) {
    const [sx, sy, sz] = size.map((v) => v * 0.5);
    const cosine = Math.cos(angle);
    const sine = Math.sin(angle);
    const point = (x, y, z) => [
      center[0] + x * cosine - z * sine,
      center[1] + y,
      center[2] + x * sine + z * cosine,
    ];
    const p = [
      point(-sx,-sy,-sz), point(sx,-sy,-sz), point(sx,sy,-sz), point(-sx,sy,-sz),
      point(-sx,-sy,sz), point(sx,-sy,sz), point(sx,sy,sz), point(-sx,sy,sz),
    ];
    this.quad(p[1],p[0],p[3],p[2],color,emissive,true);
    this.quad(p[4],p[5],p[6],p[7],color,emissive,true);
    this.quad(p[0],p[4],p[7],p[3],color,emissive,true);
    this.quad(p[5],p[1],p[2],p[6],color,emissive,true);
    this.quad(p[3],p[7],p[6],p[2],color,emissive,true);
    this.quad(p[0],p[1],p[5],p[4],color,emissive,true);
  }

  sphere(center, radius, color = C.white, emissive = [0,0,0], rings = 8, segments = 14) {
    for (let v=0;v<rings;v++) {
      const t0=v/rings*Math.PI, t1=(v+1)/rings*Math.PI;
      for (let u=0;u<segments;u++) {
        const p0=u/segments*TAU,p1=(u+1)/segments*TAU;
        const n00=[Math.sin(t0)*Math.cos(p0),Math.cos(t0),Math.sin(t0)*Math.sin(p0)];
        const n10=[Math.sin(t0)*Math.cos(p1),Math.cos(t0),Math.sin(t0)*Math.sin(p1)];
        const n01=[Math.sin(t1)*Math.cos(p0),Math.cos(t1),Math.sin(t1)*Math.sin(p0)];
        const n11=[Math.sin(t1)*Math.cos(p1),Math.cos(t1),Math.sin(t1)*Math.sin(p1)];
        const pt=(n)=>[center[0]+n[0]*radius,center[1]+n[1]*radius,center[2]+n[2]*radius];
        // Parameter-space +phi x +theta points outward. The previous order
        // used +theta x +phi, so the authored smooth normals were outward but
        // every geometric face (and raster front-facing classification) was
        // inward. That incorrectly marked sphere exteriors as closed-volume
        // backfaces and activated the interior visibility guard.
        if(v>0)this.triangle(pt(n00),pt(n10),pt(n01),color,emissive,[n00,n10,n01],true);
        if(v<rings-1)this.triangle(pt(n10),pt(n11),pt(n01),color,emissive,[n10,n11,n01],true);
      }
    }
  }

  cylinder(center, radius, height, color = C.white, segments = 14, axis = 1, emissive = [0,0,0]) {
    const axes = axis===1 ? [0,1,2] : axis===0 ? [1,0,2] : [0,2,1];
    const point=(h,a)=>{
      const p=[...center]; p[axes[1]]+=h; p[axes[0]]+=Math.cos(a)*radius; p[axes[2]]+=Math.sin(a)*radius; return p;
    };
    const normal=(a)=>{const n=[0,0,0];n[axes[0]]=Math.cos(a);n[axes[2]]=Math.sin(a);return n;};
    const top=[...center],bottom=[...center];top[axes[1]]+=height/2;bottom[axes[1]]-=height/2;
    for(let i=0;i<segments;i++){
      const a0=i/segments*TAU,a1=(i+1)/segments*TAU;
      const p00=point(-height/2,a0),p10=point(-height/2,a1),p01=point(height/2,a0),p11=point(height/2,a1);
      this.triangle(p00,p01,p10,color,emissive,[normal(a0),normal(a0),normal(a1)],true);
      this.triangle(p10,p01,p11,color,emissive,[normal(a1),normal(a0),normal(a1)],true);
      this.triangle(top,p11,p01,color,emissive,undefined,true);
      this.triangle(bottom,p00,p10,color,emissive,undefined,true);
    }
  }

  cone(center, radius, height, color = C.white, segments = 12) {
    const apex=[center[0],center[1]+height/2,center[2]],baseY=center[1]-height/2;
    for(let i=0;i<segments;i++){
      const a=i/segments*TAU,b=(i+1)/segments*TAU;
      const p=[center[0]+Math.cos(a)*radius,baseY,center[2]+Math.sin(a)*radius];
      const q=[center[0]+Math.cos(b)*radius,baseY,center[2]+Math.sin(b)*radius];
      this.triangle(p,apex,q,color,[0,0,0],undefined,true);
      this.triangle([center[0],baseY,center[2]],p,q,color,[0,0,0],undefined,true);
    }
  }

  torus(center, major, minor, color = C.white, majorSegments = 18, minorSegments = 8, rotation = 0, emissive=[0,0,0]) {
    const sample=(u,v)=>{
      const cu=Math.cos(u),su=Math.sin(u),cv=Math.cos(v),sv=Math.sin(v);
      let p=[(major+minor*cv)*cu,minor*sv,(major+minor*cv)*su];
      let n=[cv*cu,sv,cv*su];
      if(rotation){const c=Math.cos(rotation),s=Math.sin(rotation);p=[p[0],p[1]*c-p[2]*s,p[1]*s+p[2]*c];n=[n[0],n[1]*c-n[2]*s,n[1]*s+n[2]*c];}
      return {p:[center[0]+p[0],center[1]+p[1],center[2]+p[2]],n};
    };
    for(let i=0;i<majorSegments;i++)for(let j=0;j<minorSegments;j++){
      const a=sample(i/majorSegments*TAU,j/minorSegments*TAU);
      const b=sample((i+1)/majorSegments*TAU,j/minorSegments*TAU);
      const c=sample((i+1)/majorSegments*TAU,(j+1)/minorSegments*TAU);
      const d=sample(i/majorSegments*TAU,(j+1)/minorSegments*TAU);
      this.triangle(a.p,d.p,b.p,color,emissive,[a.n,d.n,b.n],true);
      this.triangle(b.p,d.p,c.p,color,emissive,[b.n,d.n,c.n],true);
    }
  }

  terrain(size, resolution, heightFn, colorFn = () => C.stone, center = [0,0,0]) {
    const step=size/resolution;
    const sample=(ix,iz)=>{
      const x=(ix/resolution-0.5)*size+center[0],z=(iz/resolution-0.5)*size+center[2];
      return [x,center[1]+heightFn(x,z),z];
    };
    // A height field is a smooth surface, even though the ray-tracing mesh is
    // triangulated.  Central-difference vertex normals prevent the lighting
    // from exposing every triangle as a contour band while preserving the
    // exact triangle geometry used by the software BVH.
    const normal=(point)=>normalize3([
      heightFn(point[0]-step,point[2])-heightFn(point[0]+step,point[2]),
      2*step,
      heightFn(point[0],point[2]-step)-heightFn(point[0],point[2]+step),
    ]);
    for(let z=0;z<resolution;z++)for(let x=0;x<resolution;x++){
      const a=sample(x,z),b=sample(x+1,z),c=sample(x+1,z+1),d=sample(x,z+1);
      const col=colorFn((a[1]+b[1]+c[1]+d[1])*0.25,x,z);
      const na=normal(a),nb=normal(b),nc=normal(c),nd=normal(d);
      this.triangle(a,d,b,col,[0,0,0],[na,nd,nb]);
      this.triangle(b,d,c,col,[0,0,0],[nb,nd,nc]);
    }
  }

  finish() {
    const bvh = buildBVH(this.triangles);
    return {
      vertices: new Float32Array(this.vertices),
      vertexCount: this.vertices.length / 16,
      ...bvh,
      emissiveGeometry: buildEmissiveBVHFromPacked(bvh.triangles),
      boundsMin: this.boundsMin,
      boundsMax: this.boundsMax,
    };
  }
}

function room(g, half=[8,5,8], back=true) {
  const [x,y,z]=half;
  g.box([0,-0.2,0],[x*2,0.4,z*2],C.chalk);
  g.box([-x, y*0.5,0],[0.35,y,z*2],C.red);
  g.box([x, y*0.5,0],[0.35,y,z*2],C.green);
  if(back)g.box([0,y*0.5,-z],[x*2,y,0.35],C.white);
  g.box([0,y,0],[x*2,0.3,z*2],C.white);
}

function addArch(g, x, z, scale=1, color=C.stone) {
  g.box([x-1.25*scale,1.8*scale,z],[0.55*scale,3.6*scale,0.7*scale],color);
  g.box([x+1.25*scale,1.8*scale,z],[0.55*scale,3.6*scale,0.7*scale],color);
  for(let i=0;i<9;i++){
    const a=Math.PI*i/8;
    g.box([x+Math.cos(a)*1.25*scale,3.55*scale+Math.sin(a)*1.25*scale,z],[0.55*scale,0.55*scale,0.7*scale],color);
  }
}

function deterministic(i) {
  const x=Math.sin(i*91.345+17.23)*43758.5453;
  return x-Math.floor(x);
}

let sponzaGeometryPromise;

async function loadPackedSponzaGeometry() {
  if (!sponzaGeometryPromise) {
    sponzaGeometryPromise = (async () => {
      const response = await fetch("/models/sponza.rcb");
      if (!response.ok) throw new Error(`Sponza geometry request failed (${response.status}).`);
      if (typeof DecompressionStream === "undefined") {
        throw new Error("This browser does not expose the gzip decompressor required by the Sponza scene.");
      }
      const stream = response.body.pipeThrough(new DecompressionStream("gzip"));
      const packed = await new Response(stream).arrayBuffer();
      const header = new Uint32Array(packed, 0, 8);
      if (header[0] !== 0x31424352 || header[1] !== 3) {
        throw new Error("The Sponza geometry package is invalid or uses an unsupported version.");
      }
      const [, , vertexFloats, nodeFloats, triangleFloats, vertexCount, nodeCount, triangleCount] = header;
      const bounds = new Float32Array(packed, 32, 6);
      let byteOffset = 64;
      const vertices = new Float32Array(packed, byteOffset, vertexFloats);
      byteOffset += vertexFloats * 4;
      const nodes = new Float32Array(packed, byteOffset, nodeFloats);
      byteOffset += nodeFloats * 4;
      const triangles = new Float32Array(packed, byteOffset, triangleFloats);
      return {
        vertices,
        nodes,
        triangles,
        vertexCount,
        nodeCount,
        triangleCount,
        emissiveGeometry: buildEmissiveBVHFromPacked(triangles),
        boundsMin: [...bounds.slice(0, 3)],
        boundsMax: [...bounds.slice(3, 6)],
      };
    })().catch((error) => {
      sponzaGeometryPromise = undefined;
      throw error;
    });
  }
  return sponzaGeometryPromise;
}

export const SCENE_INFO = [
  {name:"Color bleed laboratory",short:"Lab",description:"Near-field red/green transfer, hard occluders, emissive geometry, and moving dual lights."},
  {name:"Sponza atrium (paper scene)",short:"Sponza",description:"The official 262k-triangle Crytek Sponza geometry, recreated neutral/cyan paper palette, and red area emitter, with exact software-BVH traversal."},
  {name:"Concave canyon heightfield",short:"Canyon",description:"A 72×72 terrain mesh with nested craters, ravines, overhangs, and a moving low sun."},
  {name:"Dense lantern forest",short:"Forest",description:"Thousands of thin triangles, deep occlusion, high-frequency foliage, and colored moving lanterns."},
  {name:"Multi-level atrium",short:"Atrium",description:"Stairs, balconies, skylight transfer, curved sculptures, and cross-floor indirect illumination."},
  {name:"Industrial pipe maze",short:"Pipes",description:"Dense curved pipework, narrow gaps, emissive furnaces, and complex self-occlusion."},
  {name:"Sun temple",short:"Temple",description:"Columns, layered portals, sharp-to-soft moving sunlight, and warm/cool material transfer."},
  {name:"Orbital sculpture field",short:"Orbit",description:"Freestanding high-curvature meshes test open-sky misses, distance intervals, and moving lights."},
  {name:"Daylight door room",short:"Door",description:"A sealed unlit room whose hinged door opens to moving sun and sky, with curved normal-detail sculptures inside."},
  {name:"Megacity stress grid",short:"Stress",description:"Large terrain, 150 structures, complex monuments, maximum probe pressure, and profiling load."},
  {name:"Cornell box reference",short:"Cornell",description:"Canonical red/green Cornell enclosure, two occluding boxes, ceiling area emitter, and an animated comparison light."},
  {name:"Grand concave heightmap",short:"Heightmap",description:"A 128×128, 32k-triangle terrain with nested bowls, a winding ravine, cliff shelves, moving sun, and orbiting fill light."},
  {name:"Universal visibility laboratory",short:"Visibility",description:"A partially occluded area emitter and explicitly open two-sided sheets stress smooth visibility and topology-invariant C(-1) reconstruction."},
];

function buildScene0(g) {
  room(g);
  g.box([-2.2,1.25,0.5],[2.2,2.5,2.2],C.white);
  g.box([2.1,0.8,-2.1],[2.6,1.6,2.6],C.blue);
  g.sphere([0,1.05,2.4],1.05,C.yellow,[0,0,0],10,18);
  g.torus([0,3.2,-2.6],1.25,0.25,C.cyan,20,10,Math.PI/2);
  g.box([0,4.75,0],[3.3,0.08,3.3],[0.8,0.8,0.8],[5.2,4.6,3.5]);
  return {camera:[6.5,3.7,13],target:[0,1.8,0],env:[0.02,0.025,0.04],sun:2.2};
}

function buildScene1(g) {
  g.box([0,-0.25,0],[30,0.5,18],C.stone);
  for(const side of [-1,1])for(let i=-4;i<=4;i++){
    addArch(g,i*3.1,side*6.5,0.92, i%2?C.sand:C.stone);
    g.cylinder([i*3.1,2.15,side*5.7],0.34,4.3,C.chalk,16);
  }
  g.box([0,6.3,-7.3],[30,0.45,2.0],C.sand);
  g.box([0,6.3,7.3],[30,0.45,2.0],C.sand);
  for(let i=0;i<9;i++)g.sphere([-12+i*3,1.15,0],0.65,i%3===0?C.red:C.chalk,[0,0,0],7,12);
  g.torus([0,2.8,0],2.3,0.42,C.yellow,24,10,Math.PI/2);
  return {
    camera:[20,8.5,18],target:[0,2.2,0],env:[0.055,0.08,0.12],sun:3.3,
    paperPalette:true,sunHorizontal:0.28,sunHeight:-0.96,
    pointColor:[1.0,0.12,0.06],sunColor:[1.0,0.98,0.92],pointIntensity:0,
  };
}

function buildScene2(g) {
  const h=(x,z)=>{
    const r=Math.hypot(x+4,z-2),r2=Math.hypot(x-7,z+7);
    return 2.4*Math.sin(x*0.19)*Math.cos(z*0.16)-6*Math.exp(-r*r/38)-3.7*Math.exp(-r2*r2/19)+0.012*(x*x+z*z);
  };
  g.terrain(44,72,h,(height)=>height<0?[0.31,0.18,0.075]:height>5?[0.54,0.49,0.38]:C.sand);
  g.box([-4,-0.7,2],[7,0.65,3.2],C.dark);
  for(let i=0;i<7;i++)g.torus([-10+i*3.2,2.0+h(-10+i*3.2,-8),-8],1.0,0.22,i%2?C.orange:C.stone,16,8,Math.PI/2);
  g.box([9,3.2,-4],[8,0.45,4],C.stone);
  return {camera:[31,18,30],target:[0,0,0],env:[0.08,0.11,0.16],sun:4.2};
}

function buildScene3(g) {
  g.terrain(34,32,(x,z)=>0.35*Math.sin(x*0.4)+0.28*Math.cos(z*0.5),()=>[0.08,0.14,0.055]);
  for(let i=0;i<70;i++){
    const x=(deterministic(i*3)-0.5)*31,z=(deterministic(i*3+1)-0.5)*31;
    const h=2.5+deterministic(i*3+2)*3;
    g.cylinder([x,h*0.48,z],0.13+deterministic(i+99)*0.12,h,C.bark,9);
    g.cone([x,h+0.4,z],1.0+deterministic(i+7)*0.7,2.9+deterministic(i+12)*1.8,i%8===0?C.yellow:C.leaf,10);
  }
  for(let i=0;i<8;i++)g.sphere([-12+i*3.4,2.0+Math.sin(i),-2+Math.cos(i)*4],0.28,[0.8,0.45,0.08],[5,1.6,0.15],6,10);
  return {camera:[25,11,26],target:[0,3,0],env:[0.035,0.05,0.075],sun:1.65,exposure:1.12};
}

function buildScene4(g) {
  g.box([0,-0.3,0],[28,0.6,22],C.concrete);
  for(const y of [2.8,6.0]) {
    g.box([-11,y,0],[1.2,0.4,20],C.white); g.box([11,y,0],[1.2,0.4,20],C.white);
    g.box([0,y,-9],[23,0.4,1.2],C.white);
  }
  for(const x of [-9,-5,-1,3,7])g.cylinder([x,3.1,-7.8],0.35,6.2,C.stone,16);
  for(let i=0;i<10;i++)g.box([-7+i*0.8,0.25+i*0.28,3],[0.8,0.56,5.5],i%2?C.stone:C.chalk);
  g.torus([3.5,2.7,2],2.1,0.48,C.violet,24,12,Math.PI/2);
  g.sphere([-3.8,1.6,2.3],1.55,C.cyan,[0,0,0],12,20);
  g.box([0,8.4,0],[8,0.12,6],[0.7,0.75,0.8],[3.6,4.1,5.3]);
  return {camera:[25,12,25],target:[0,3,0],env:[0.025,0.04,0.065],sun:2.8};
}

function buildScene5(g) {
  g.box([0,-0.25,0],[34,0.5,24],C.concrete);
  g.box([0,5,-11.5],[34,10,0.7],C.dark);
  for(let i=0;i<15;i++){
    const x=-14+(i%5)*7,z=-8+Math.floor(i/5)*7,y=1.2+(i%3)*1.6;
    g.cylinder([x,y,z],0.38+(i%2)*0.18,5.5,C.metal,14,i%3);
    g.torus([x,y+1.4,z],1.0+(i%3)*0.2,0.24,i%4===0?C.orange:C.metal,18,8,i%2?Math.PI/2:0);
  }
  for(let i=0;i<7;i++)g.box([-13+i*4.3,1.2,-10.7],[2.8,2.4,0.5],C.dark,[4.5,0.35+0.15*i,0.05]);
  return {camera:[29,13,30],target:[0,2.2,0],env:[0.012,0.018,0.022],sun:1.7};
}

function buildScene6(g) {
  g.box([0,-0.25,0],[32,0.5,32],C.sand);
  for(const z of [-10,-4,4,10])for(const x of [-11,-5,5,11]){
    g.cylinder([x,3.8,z],0.65,7.6,C.stone,18);
    g.box([x,7.75,z],[2.1,0.5,2.1],C.sand);
  }
  for(let z=-8;z<=8;z+=4)addArch(g,0,z,1.15,z%8?C.stone:C.chalk);
  g.box([0,0.7,-13],[9,1.4,5],C.red);
  g.sphere([0,2.7,-13],1.75,C.yellow,[0,0,0],12,20);
  g.box([0,8.2,0],[6,0.1,10],[0.8,0.75,0.55],[4.2,3.6,2.0]);
  return {camera:[31,16,33],target:[0,3,0],env:[0.07,0.08,0.12],sun:4.8};
}

function buildScene7(g) {
  g.box([0,-3.2,0],[38,0.45,38],C.dark);
  for(let i=0;i<22;i++){
    const a=i/22*TAU,r=5+(i%5)*2.4,y=-0.5+Math.sin(i*1.7)*3.2;
    const p=[Math.cos(a)*r,y,Math.sin(a)*r];
    if(i%3===0)g.torus(p,1.4+(i%4)*0.2,0.3,i%2?C.cyan:C.violet,20,10,a);
    else if(i%3===1)g.sphere(p,1.0+(i%4)*0.3,i%2?C.orange:C.blue,[0,0,0],10,18);
    else {g.cylinder(p,0.65,3.3,C.metal,16,i%3);g.torus(p,1.2,0.18,C.yellow,16,8,Math.PI/2);}
  }
  // A mesh light is a radiometric source, not a silhouette tessellation
  // benchmark. Smooth vertex normals keep the orb visually round while a
  // bounded 6x12 source mesh prevents exact local-visibility cost from being
  // multiplied by hundreds of nearly redundant micro-triangles.
  g.sphere([0,1,0],2.2,C.white,[1.8,0.45,3.5],6,12);
  return {camera:[30,14,32],target:[0,0,0],env:[0.025,0.035,0.09],sun:2.0};
}

function daylightDoorOpenAmount(seconds) {
  // Eight-second cycle with deliberate closed/open holds so both states are
  // easy to inspect. The cubic ramps have zero velocity at either end.
  const phase=((seconds%8)+8)%8;
  const smooth=(value)=>value*value*(3-2*value);
  if(phase<1.5)return 0;
  if(phase<3.0)return smooth((phase-1.5)/1.5);
  if(phase<5.5)return 1;
  if(phase<7.0)return smooth((7.0-phase)/1.5);
  return 0;
}

function buildScene8(g, seconds = 0) {
  const wall=[0.39,0.42,0.43];
  const plaster=[0.68,0.66,0.59];
  const roomWidth=16,roomDepth=14,roomHeight=5.2;
  const front=roomDepth*0.5;
  const doorHalfWidth=1.35,doorHeight=4.35;

  // Six opaque shells make a genuinely sealed room. The front is split only
  // around the authored doorway; the slightly oversized leaf overlaps the
  // jamb and floor in its closed pose to avoid numerical pinholes.
  g.box([0,-0.2,0],[roomWidth,0.4,roomDepth],plaster);
  g.box([0,roomHeight+0.2,0],[roomWidth,0.4,roomDepth],wall);
  g.box([0,roomHeight*0.5,-roomDepth*0.5],[roomWidth,roomHeight,0.4],wall);
  g.box([-roomWidth*0.5,roomHeight*0.5,0],[0.4,roomHeight,roomDepth],wall);
  g.box([roomWidth*0.5,roomHeight*0.5,0],[0.4,roomHeight,roomDepth],wall);
  const sideWidth=(roomWidth*0.5-doorHalfWidth);
  g.box([-(doorHalfWidth+sideWidth*0.5),doorHeight*0.5,front],[sideWidth,doorHeight,0.4],wall);
  g.box([doorHalfWidth+sideWidth*0.5,doorHeight*0.5,front],[sideWidth,doorHeight,0.4],wall);
  g.box([0,doorHeight+(roomHeight-doorHeight)*0.5,front],[roomWidth,roomHeight-doorHeight,0.4],wall);

  const openness=daylightDoorOpenAmount(seconds);
  const angle=openness*Math.PI*0.49;
  // The leaf overlaps all four sides of the structural opening, like a real
  // rebated exterior door. Merely matching the aperture edge leaves a
  // zero-width geometric seam through which exact floating-point rays can
  // travel even though raster coverage looks closed.
  const doorOverlap=0.18;
  const hinge=[-doorHalfWidth-doorOverlap,doorHeight*0.5-0.02,front-0.27];
  const leafWidth=doorHalfWidth*2+doorOverlap*2;
  const center=[
    hinge[0]+Math.cos(angle)*leafWidth*0.5,
    hinge[1],
    hinge[2]+Math.sin(angle)*leafWidth*0.5,
  ];
  g.boxRotatedY(center,[leafWidth,doorHeight+0.32,0.28],angle,[0.18,0.21,0.22]);

  // Sunlit exterior ground and a pair of chunky portals make the changing
  // daylight direction legible through the open doorway without adding any
  // emissive or point-light source.
  g.box([0,-0.18,15],[24,0.36,16],[0.30,0.34,0.27]);
  g.box([-5.2,2.2,13.5],[1.2,4.4,1.2],C.stone);
  g.box([5.0,1.5,17],[1.5,3.0,1.5],C.sand);

  // Normal-rich interior exhibit: smooth curved normals, thin silhouettes,
  // convex caps, and rotated planar forms under daylight only.
  g.boxRotatedY([-3.1,0.38,3.4],[3.15,0.55,1.65],0.16,C.dark);
  g.torus([-3.1,2.25,3.4],1.45,0.34,C.cyan,24,12,Math.PI/2);
  g.sphere([1.35,1.05,3.65],1.05,C.orange,[0,0,0],10,18);
  g.cylinder([5.3,1.65,-0.2],0.72,3.3,C.blue,18);
  g.cone([-5.0,1.15,0.2],1.05,2.3,C.yellow,18);
  g.torus([2.55,2.55,1.25],0.9,0.2,C.violet,20,10,0);
  for(let i=0;i<7;i++){
    const x=-5.7+i*1.9;
    g.cylinder([x,3.7,-6.45],0.16,2.15,i%2?C.chalk:C.metal,10,0);
  }

  return {
    camera:[5.8,3.2,-3.2],target:[0,1.9,4.5],
    env:[0.18,0.235,0.33],sun:3.0,pointIntensity:0,exposure:0.95,
    // Keep the animated source high enough that an open aperture produces a
    // compact daylight pool rather than a several-metre grazing needle. This
    // is authored lighting (the universal GI path is unchanged), and the
    // lower filmic peak retains detail instead of clipping the pool to white.
    sunHorizontal:0.65,sunHeight:-0.88,dynamicGeometry:true,
  };
}

function buildScene9(g) {
  g.terrain(60,56,(x,z)=>0.7*Math.sin(x*0.17)*Math.cos(z*0.13)-2.8*Math.exp(-((x-8)**2+(z+6)**2)/70),()=>C.concrete);
  for(let i=0;i<150;i++){
    const x=(deterministic(i*4)-0.5)*54,z=(deterministic(i*4+1)-0.5)*54;
    const h=1.5+deterministic(i*4+2)*9,w=1.1+deterministic(i*4+3)*2.2;
    const color=i%7===0?C.red:i%5===0?C.blue:C.dark;
    g.box([x,h/2,z],[w,h,w*(0.75+deterministic(i+55)*0.7)],color);
  }
  for(let i=0;i<12;i++){
    const a=i/12*TAU;
    g.torus([Math.cos(a)*11,5+Math.sin(i)*2,Math.sin(a)*11],2.0,0.38,i%2?C.orange:C.cyan,20,10,a);
  }
  // Keep source integration bounded independently of the surrounding stress
  // geometry. The interpolated normals preserve the smooth visible orb; the
  // 6x12 radiometric mesh is already finer than the local cascade's angular
  // sampling and avoids oversampling the same emitter hundreds of times.
  g.sphere([0,8,0],3.0,C.white,[2.5,0.25,0.08],6,12);
  return {camera:[50,30,53],target:[0,3,0],env:[0.02,0.035,0.06],sun:3.6};
}

function buildScene10(g) {
  // Cornell's front is intentionally open. The walls use the original
  // proportions and diffuse color arrangement so this doubles as a compact
  // color-bleed/reference scene rather than another arbitrary room.
  const left=-2.78,right=2.78,floor=0,ceiling=5.49,back=-5.59,front=0.15;
  g.quad([left,floor,front],[right,floor,front],[right,floor,back],[left,floor,back],C.white);
  g.quad([left,ceiling,back],[right,ceiling,back],[right,ceiling,front],[left,ceiling,front],C.white);
  g.quad([left,floor,back],[right,floor,back],[right,ceiling,back],[left,ceiling,back],C.white);
  g.quad([left,floor,front],[left,floor,back],[left,ceiling,back],[left,ceiling,front],C.red);
  g.quad([right,floor,back],[right,floor,front],[right,ceiling,front],[right,ceiling,back],C.green);
  // Match the canonical Cornell layout: the short block turns clockwise and
  // the tall block counter-clockwise instead of presenting axis-aligned faces.
  g.boxRotatedY([-1.05,0.82,-3.8],[1.65,1.64,1.65],-0.30,C.white);
  g.boxRotatedY([1.0,1.55,-2.05],[1.75,3.1,1.75],0.28,C.white);
  g.quad(
    [-0.65,ceiling-0.025,-3.9],[0.65,ceiling-0.025,-3.9],
    [0.65,ceiling-0.025,-2.85],[-0.65,ceiling-0.025,-2.85],
    [0.9,0.88,0.72],[8.5,7.4,5.8],
  );
  return {
    camera:[0,2.65,8.6],target:[0,2.5,-2.75],
    env:[0.0015,0.0015,0.0015],sun:0.05,pointIntensity:3.5,
    pointOrbit:1.45,pointBaseHeight:0.0,pointHeight:0.75,
    pointColor:[1.0,0.82,0.62],exposure:1.15,
  };
}

function buildScene11(g) {
  const height=(x,z)=>{
    const bowl=Math.hypot(x+12,z-7);
    const crater=Math.hypot(x-16,z+13);
    const ravineDistance=Math.abs(z-7*Math.sin(x*0.075)-2*Math.sin(x*0.22));
    // Broad, differentiable shelves retain the concave terrace stress case
    // without quantizing the terrain into conspicuous zebra contours.
    const terraceSignal=Math.sin(x*0.055)+Math.cos(z*0.061);
    const terracing=1+Math.tanh(1.45*terraceSignal);
    const raw=2.8*Math.sin(x*0.075)*Math.cos(z*0.068)
      -10.5*Math.exp(-bowl*bowl/155)
      -7.0*Math.exp(-crater*crater/88)
      -5.8*Math.exp(-ravineDistance*ravineDistance/5.5)
      +0.7*terracing+0.0022*(x*x+z*z);
    // Blend the outer 18% to a coherent rim so a ravine cannot cut an
    // accidental single-triangle notch into the finite height-field boundary.
    const edge=Math.max(Math.abs(x),Math.abs(z))/46;
    const edgeLinear=Math.max(0,Math.min(1,(edge-0.82)/0.18));
    const edgeBlend=edgeLinear*edgeLinear*(3-2*edgeLinear);
    return raw*(1-edgeBlend)+4*edgeBlend;
  };
  g.terrain(
    92,128,height,
    ()=>[0.43,0.31,0.16],
  );
  // Shelves and arches create concavities a single-valued height field cannot,
  // while the dense ground remains a genuine heightmap.
  for(let i=0;i<9;i++){
    const x=-30+i*7.5,z=-15+5*Math.sin(i*0.9),y=height(x,z)+2.4;
    g.box([x,y,z],[7.8,0.55,5.2],i%2?C.stone:C.sand);
    g.torus([x,y-0.8,z],1.55,0.28,C.dark,18,8,Math.PI/2);
  }
  for(let i=0;i<18;i++){
    const a=i/18*TAU,r=18+4*Math.sin(i*1.7);
    const x=Math.cos(a)*r,z=Math.sin(a)*r,y=height(x,z)+1.2;
    g.sphere([x,y,z],0.65+(i%3)*0.18,i%2?C.orange:C.cyan,[0,0,0],8,14);
  }
  return {
    camera:[67,38,72],target:[0,-1,0],
    env:[0.065,0.085,0.13],sun:2.15,pointIntensity:9.0,exposure:0.88,
    pointOrbit:28,pointBaseHeight:9,pointHeight:6,
    pointColor:[0.12,0.45,1.0],sunHeight:-0.52,sunHorizontal:0.86,
  };
}

function buildScene12(g) {
  // The floor receives a broad area source through a thin, off-center
  // occluder. Camera motion sweeps continuously across partial visibility.
  g.quad([-7,0,5],[7,0,5],[7,0,-7],[-7,0,-7],C.white);
  g.quad([-7,0,-7],[7,0,-7],[7,6,-7],[-7,6,-7],C.chalk);
  g.quad(
    [-2.8,5.1,0.5],[2.8,5.1,0.5],
    [2.8,5.1,4.5],[-2.8,5.1,4.5],
    [0.9,0.9,0.9],[6.8,5.7,4.4],
  );
  g.box([0.45,2.35,2.2],[0.22,3.5,2.8],C.dark);
  g.box([-3.7,0.65,-2.7],[1.4,1.3,1.4],C.red);
  g.box([3.4,1.05,-3.4],[1.6,2.1,1.6],C.green);

  // A second emitter sits inside one automatically derived base interval of
  // a vertical receiver. The narrow intervening bar covers only part of its
  // solid angle, forcing the production C(-1) patch-visibility path.
  g.quad(
    [-2.9,1.0,-4.0],[2.9,1.0,-4.0],
    [2.9,5.2,-4.0],[-2.9,5.2,-4.0],C.white,
  );
  g.quad(
    [-1.8,2.0,-3.62],[-1.8,4.55,-3.62],
    [1.8,4.55,-3.62],[1.8,2.0,-3.62],
    [0.9,0.9,0.9],[0.55,0.48,0.38],
  );
  g.box([0.35,3.25,-3.81],[0.16,1.85,0.12],C.dark);

  // These sheets are intentionally open and visible from their authored
  // back side. They must never acquire closed-volume suppression merely
  // because the rasterizer reports a back-facing primitive.
  g.quad(
    [-4.8,0.25,0.8],[-4.8,3.6,0.8],[-4.8,3.6,-1.4],[-4.8,0.25,-1.4],
    C.blue,
  );
  g.quad(
    [4.8,0.25,-1.4],[4.8,3.6,-1.4],[4.8,3.6,0.8],[4.8,0.25,0.8],
    C.yellow,
  );
  return {
    camera:[7.8,5.0,10.5],target:[0,1.4,-1.8],
    env:[0.004,0.005,0.008],sun:0.0,pointIntensity:0,exposure:1.0,
  };
}

const BUILDERS=[
  buildScene0,buildScene1,buildScene2,buildScene3,buildScene4,buildScene5,
  buildScene6,buildScene7,buildScene8,buildScene9,buildScene10,buildScene11,
  buildScene12,
];

// Universal, asset-driven base resolution. The scene diagonal establishes
// world scale while a very shallow triangle-density term preserves small
// modeled features without turning dense assets into a probe-capacity spike.
// No scene identity or hand-authored GI spacing participates in this choice.
export function automaticBaseSpacing(radius, triangleCount) {
  const density = Math.pow(Math.max(1, triangleCount), 0.12);
  return Math.max(0.22, radius * 0.077 / density);
}

export function dynamicSceneKey(index, seconds) {
  if(index!==8)return null;
  // Quantizing more finely than the visible angular motion prevents needless
  // rebuilds during the two hold phases while keeping the hinge animation
  // visually continuous.
  return Math.round(daylightDoorOpenAmount(seconds)*480)/480;
}

export function createDynamicSceneGeometry(index, seconds) {
  if(index!==8)return null;
  const g=new Geometry();
  BUILDERS[index](g,seconds);
  const geometry=g.finish();
  geometry.dynamicKey=dynamicSceneKey(index,seconds);
  return geometry;
}

export function createScene(index) {
  if (index === 1 && typeof window !== "undefined") {
    return loadPackedSponzaGeometry().then((geometry) => {
      const radius = Math.hypot(...geometry.boundsMax.map(
        (value, axis) => (value - geometry.boundsMin[axis]) * 0.5,
      ));
      return {
        id: index,
        ...SCENE_INFO[index],
        camera: [-8.0, 8.0, -0.5],
        target: [5.0, 2.0, -0.5],
        env: [0.55, 0.65, 0.82],
        sun: 1.25,
        exposure: 1.0,
        paperPalette: true,
        sunHorizontal: 0.28,
        sunHeight: -0.96,
        pointColor: [1.0, 0.12, 0.06],
        sunColor: [1.0, 0.98, 0.92],
        pointIntensity: 0,
        geometry,
        radius,
        baseSpacing: automaticBaseSpacing(radius, geometry.triangleCount),
      };
    });
  }
  const g=new Geometry();
  const settings=BUILDERS[index](g,0);
  const geometry=g.finish();
  const radius=Math.hypot(...geometry.boundsMax.map((v,i)=>(v-geometry.boundsMin[i])*0.5));
  return {
    id:index,
    ...SCENE_INFO[index],
    ...settings,
    geometry,
    radius,
    baseSpacing:automaticBaseSpacing(radius,geometry.triangleCount),
  };
}
