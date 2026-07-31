import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const engine = await readFile(new URL("../public/rc/engine.js", import.meta.url), "utf8");

test("production audit gates continuous motion in baseline and multibounce modes", () => {
  assert.match(engine, /compareReprojectedFrames/);
  assert.match(engine, /matchedPixelRatio >= 0\.35/);
  assert.match(engine, /trimmedRmseByteDelta/);
  assert.match(engine, /maximum\("p95Absolute"\) <= 0\.0045/);
  assert.match(engine, /p95ByteDeltaMax/);
  assert.match(engine, /multibounceContinuousMotion/);
  assert.match(engine, /multibounceRepeatability/);
  assert.match(engine, /report\.multibounceContinuousMotion\.passed/);
});

test("path-reference gate classifies dark spots and bright leaks", () => {
  assert.match(engine, /severeUnderlitRatio/);
  assert.match(engine, /severeOverlitRatio/);
  assert.match(engine, /report\.p99Absolute <= 0\.5/);
  assert.match(engine, /report\.severeUnderlitRatio <= 0\.04/);
  assert.match(engine, /lowFrequencyScaleInvariantNrmse/);
  assert.match(engine, /report\.lowFrequencyScaleInvariantNrmse <= 0\.44/);
  assert.match(engine, /renderReferenceComparison/);
});

test("sampling epochs replay deterministically without stale ray-map tags", () => {
  assert.match(engine, /this\.sampleFrameIndex = 0/);
  assert.match(engine, /this\.sampleEpoch = 1/);
  assert.match(engine, /const sampleTag =/);
  assert.match(engine, /\(\(this\.sampleEpoch & 0xffff\) << 16\)/);
  assert.match(engine, /this\.sampleFrameIndex = \(this\.sampleFrameIndex \+ 1\) & 0xffff/);
});
