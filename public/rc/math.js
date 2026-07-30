export const TAU = Math.PI * 2;
export const CASCADE_DIRECTIONS = [32, 128, 512, 2048];
export const CASCADE_MAX_PROBES = [2048, 1024, 512, 256];
export const CASCADE_HASH_SIZE = [4096, 4096, 2048, 2048];

export function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

export function vec3(x = 0, y = 0, z = 0) {
  return [x, y, z];
}

export function add3(a, b) {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

export function sub3(a, b) {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

export function mul3(a, s) {
  return [a[0] * s, a[1] * s, a[2] * s];
}

export function dot3(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

export function cross3(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

export function normalize3(v) {
  const s = 1 / Math.max(1e-12, Math.hypot(v[0], v[1], v[2]));
  return [v[0] * s, v[1] * s, v[2] * s];
}

export function mat4Identity() {
  return new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
}

export function mat4Multiply(a, b) {
  const out = new Float32Array(16);
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      out[c * 4 + r] =
        a[r] * b[c * 4] +
        a[4 + r] * b[c * 4 + 1] +
        a[8 + r] * b[c * 4 + 2] +
        a[12 + r] * b[c * 4 + 3];
    }
  }
  return out;
}

export function mat4Perspective(fovy, aspect, near, far) {
  const f = 1 / Math.tan(fovy / 2);
  const nf = 1 / (near - far);
  return new Float32Array([
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, far * nf, -1,
    0, 0, far * near * nf, 0,
  ]);
}

export function mat4Ortho(left, right, bottom, top, near, far) {
  return new Float32Array([
    2 / (right - left), 0, 0, 0,
    0, 2 / (top - bottom), 0, 0,
    0, 0, 1 / (near - far), 0,
    (left + right) / (left - right),
    (top + bottom) / (bottom - top),
    near / (near - far),
    1,
  ]);
}

export function mat4LookAt(eye, center, up = [0, 1, 0]) {
  const z = normalize3(sub3(eye, center));
  const x = normalize3(cross3(up, z));
  const y = cross3(z, x);
  return new Float32Array([
    x[0], y[0], z[0], 0,
    x[1], y[1], z[1], 0,
    x[2], y[2], z[2], 0,
    -dot3(x, eye), -dot3(y, eye), -dot3(z, eye), 1,
  ]);
}

export function mat4Inverse(m) {
  const out = new Float32Array(16);
  const a00=m[0],a01=m[1],a02=m[2],a03=m[3],a10=m[4],a11=m[5],a12=m[6],a13=m[7];
  const a20=m[8],a21=m[9],a22=m[10],a23=m[11],a30=m[12],a31=m[13],a32=m[14],a33=m[15];
  const b00=a00*a11-a01*a10,b01=a00*a12-a02*a10,b02=a00*a13-a03*a10,b03=a01*a12-a02*a11;
  const b04=a01*a13-a03*a11,b05=a02*a13-a03*a12,b06=a20*a31-a21*a30,b07=a20*a32-a22*a30;
  const b08=a20*a33-a23*a30,b09=a21*a32-a22*a31,b10=a21*a33-a23*a31,b11=a22*a33-a23*a32;
  let det=b00*b11-b01*b10+b02*b09+b03*b08-b04*b07+b05*b06;
  if (!det) return mat4Identity();
  det=1/det;
  out[0]=(a11*b11-a12*b10+a13*b09)*det; out[1]=(a02*b10-a01*b11-a03*b09)*det;
  out[2]=(a31*b05-a32*b04+a33*b03)*det; out[3]=(a22*b04-a21*b05-a23*b03)*det;
  out[4]=(a12*b08-a10*b11-a13*b07)*det; out[5]=(a00*b11-a02*b08+a03*b07)*det;
  out[6]=(a32*b02-a30*b05-a33*b01)*det; out[7]=(a20*b05-a22*b02+a23*b01)*det;
  out[8]=(a10*b10-a11*b08+a13*b06)*det; out[9]=(a01*b08-a00*b10-a03*b06)*det;
  out[10]=(a30*b04-a31*b02+a33*b00)*det; out[11]=(a21*b02-a20*b04-a23*b00)*det;
  out[12]=(a11*b07-a10*b09-a12*b06)*det; out[13]=(a00*b09-a01*b07+a02*b06)*det;
  out[14]=(a31*b01-a30*b03-a32*b00)*det; out[15]=(a20*b03-a21*b01+a22*b00)*det;
  return out;
}

export function r2(index, jitter = [0, 0]) {
  const g = 1.324717957244746;
  return [
    (0.5 + index / g + jitter[0]) % 1,
    (0.5 + index / (g * g) + jitter[1]) % 1,
  ];
}

export function decodeEqualArea([u, v]) {
  const phi = u * TAU;
  const z = v * 2 - 1;
  const r = Math.sqrt(Math.max(0, 1 - z * z));
  return [r * Math.cos(phi), r * Math.sin(phi), z];
}

export function encodeEqualArea([x, y, z]) {
  return [((Math.atan2(y, x) / TAU) % 1 + 1) % 1, z * 0.5 + 0.5];
}

export function nearestProbe(position, spacing) {
  return position.map((v) => spacing * (Math.floor(v / spacing) + 0.5));
}

export function packProbeKey(position, spacing, lod = 0) {
  const p = position.map((v) => clamp(Math.floor(v / spacing), -256, 255) + 256);
  return ((p[0] | (p[1] << 9) | (p[2] << 18) | ((lod & 7) << 27)) >>> 0);
}

function expandBits(v) {
  v = (v * 0x00010001) & 0xff0000ff;
  v = (v * 0x00000101) & 0x0f00f00f;
  v = (v * 0x00000011) & 0xc30c30c3;
  v = (v * 0x00000005) & 0x49249249;
  return v >>> 0;
}

function morton3(x, y, z) {
  return (expandBits(x) | (expandBits(y) << 1) | (expandBits(z) << 2)) >>> 0;
}

export function buildBVH(triangles, maxLeaf = 4) {
  if (!triangles.length) {
    return { nodes: new Float32Array(16), triangles: new Float32Array(28), nodeCount: 1 };
  }
  const bounds = { min: [Infinity,Infinity,Infinity], max: [-Infinity,-Infinity,-Infinity] };
  const refs = triangles.map((t, index) => {
    const c = [(t.a[0]+t.b[0]+t.c[0])/3,(t.a[1]+t.b[1]+t.c[1])/3,(t.a[2]+t.b[2]+t.c[2])/3];
    for (let k=0;k<3;k++) { bounds.min[k]=Math.min(bounds.min[k],c[k]); bounds.max[k]=Math.max(bounds.max[k],c[k]); }
    return { t, index, c, code: 0 };
  });
  const extent = bounds.max.map((v,k)=>Math.max(1e-6,v-bounds.min[k]));
  for (const r of refs) {
    const q=r.c.map((v,k)=>clamp(Math.floor((v-bounds.min[k])/extent[k]*1023),0,1023));
    r.code=morton3(q[0],q[1],q[2]);
  }
  refs.sort((a,b)=>a.code-b.code);
  const nodes=[];
  const ordered=[];
  const build=(start,end)=>{
    const nodeIndex=nodes.length;
    const node={min:[Infinity,Infinity,Infinity],max:[-Infinity,-Infinity,-Infinity],left:0,right:0,leaf:false};
    nodes.push(node);
    for(let i=start;i<end;i++){
      for(const p of [refs[i].t.a,refs[i].t.b,refs[i].t.c]){
        for(let k=0;k<3;k++){node.min[k]=Math.min(node.min[k],p[k]);node.max[k]=Math.max(node.max[k],p[k]);}
      }
    }
    if(end-start<=maxLeaf){
      node.leaf=true; node.left=ordered.length; node.right=end-start;
      for(let i=start;i<end;i++) ordered.push(refs[i].t);
    } else {
      const mid=(start+end)>>1;
      node.left=build(start,mid); node.right=build(mid,end);
    }
    return nodeIndex;
  };
  build(0,refs.length);
  const nodeData=new ArrayBuffer(nodes.length*32);
  const nf=new Float32Array(nodeData), nu=new Uint32Array(nodeData);
  nodes.forEach((n,i)=>{
    const o=i*8;
    nf.set(n.min,o); nf.set(n.max,o+4);
    nu[o+3]=n.leaf?(0x80000000|n.left)>>>0:n.left>>>0;
    nu[o+7]=n.right>>>0;
  });
  const triData=new Float32Array(ordered.length*28);
  ordered.forEach((t,i)=>{
    const o=i*28;
    const uvs=t.uvs||[[0,0],[0,0],[0,0]];
    triData.set([
      ...t.a,0,...t.b,0,...t.c,0,...t.albedo,0,...t.emissive,0,
      ...uvs[0],...uvs[1],...uvs[2],t.material??-1,t.alphaCutoff??0,
    ],o);
  });
  return {nodes:new Float32Array(nodeData),triangles:triData,nodeCount:nodes.length,triangleCount:ordered.length};
}

export function intersectTriangle(origin, direction, triangle) {
  const e1=sub3(triangle.b,triangle.a),e2=sub3(triangle.c,triangle.a);
  const p=cross3(direction,e2),det=dot3(e1,p);
  if(Math.abs(det)<1e-8)return Infinity;
  const inv=1/det,tv=sub3(origin,triangle.a),u=dot3(tv,p)*inv;
  if(u<0||u>1)return Infinity;
  const q=cross3(tv,e1),v=dot3(direction,q)*inv;
  if(v<0||u+v>1)return Infinity;
  const t=dot3(e2,q)*inv;
  return t>1e-5?t:Infinity;
}
