import assert from "node:assert/strict";
import test from "node:test";
import { computeShader, finalShader, shaderConstants } from "../public/rc/shaders.js";

test("world-space sampling and probe history are double-buffered", () => {
  assert.equal(shaderConstants.hashFrames, 2);
  assert.equal(shaderConstants.irradianceFrames, 2);
  assert.match(computeShader, /stableCell=vec3i\(floor\(world\.xyz\/stableScale\)\)/);
  assert.match(computeShader, /lookupProbeFrame\(0u,key,previousFrame\)/);
  assert.match(computeShader, /filtered=mix\(filtered,history,blend\)/);
  assert.match(finalShader, /frameIndex\*IRRADIANCE_FRAME_STRIDE/);
});
