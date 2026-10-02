import { test } from "node:test";
import assert from "node:assert/strict";

import {
  computeHomography,
  expandQuad,
  orderQuad,
  warpQuadToStrip,
  REC_INPUT_HEIGHT,
  REC_INPUT_MAX_WIDTH,
} from "../src/core/homography.js";

test("computeHomography maps the four correspondences exactly", () => {
  const src = [0, 0, 100, 0, 100, 50, 0, 50];
  const dst = [10, 20, 200, 30, 190, 120, 5, 110];
  const h = computeHomography(src, dst);
  assert.ok(h);

  for (let i = 0; i < 4; i += 1) {
    const x = src[i * 2];
    const y = src[i * 2 + 1];
    const denom = h[6] * x + h[7] * y + h[8];
    const px = (h[0] * x + h[1] * y + h[2]) / denom;
    const py = (h[3] * x + h[4] * y + h[5]) / denom;
    assert.ok(Math.abs(px - dst[i * 2]) < 1e-6, `x${i}: ${px} vs ${dst[i * 2]}`);
    assert.ok(Math.abs(py - dst[i * 2 + 1]) < 1e-6, `y${i}: ${py} vs ${dst[i * 2 + 1]}`);
  }
});

test("computeHomography returns null for degenerate correspondences", () => {
  const collinear = [0, 0, 10, 0, 20, 0, 30, 0];
  assert.equal(computeHomography([0, 0, 1, 0, 1, 1, 0, 1], collinear), null);
  assert.equal(computeHomography(collinear, [0, 0, 1, 0, 1, 1, 0, 1]), null);
});

test("orderQuad puts the long left→right edge first", () => {
  const canonical = [10, 5, 170, 5, 170, 45, 10, 45];
  // Rotated corner order (starts at BR) and reversed direction.
  const startsAtShortEdge = [170, 5, 170, 45, 10, 45, 10, 5];
  const flipped180 = [170, 45, 10, 45, 10, 5, 170, 5];

  assert.deepEqual(orderQuad(canonical), canonical);
  assert.deepEqual(orderQuad(startsAtShortEdge), canonical);
  assert.deepEqual(orderQuad(flipped180), canonical);
});

test("warpQuadToStrip round-trips an axis-aligned gradient", () => {
  // 200×50 RGBA image with a horizontal red gradient r = x (0..199).
  const imgW = 200;
  const imgH = 50;
  const data = new Uint8ClampedArray(imgW * imgH * 4);
  for (let y = 0; y < imgH; y += 1) {
    for (let x = 0; x < imgW; x += 1) {
      const i = (y * imgW + x) * 4;
      data[i] = x; // R encodes source column
      data[i + 3] = 255;
    }
  }
  const image = { data, width: imgW, height: imgH };

  // Quad = plain rectangle 160×40 → aspect 4 → target width 192 columns.
  const quad = [10, 5, 170, 5, 170, 45, 10, 45];
  const out = new Float32Array(3 * REC_INPUT_HEIGHT * REC_INPUT_MAX_WIDTH);
  const result = warpQuadToStrip(image, quad, out);

  assert.ok(result);
  assert.equal(result.width, Math.round((REC_INPUT_HEIGHT * 160) / 40));

  const plane = REC_INPUT_HEIGHT * REC_INPUT_MAX_WIDTH;
  const midRow = REC_INPUT_HEIGHT >> 1;
  // Probe a few columns: normalized value should decode back to the source x.
  for (const col of [0, 50, 100, result.width - 1]) {
    const normalized = out[midRow * REC_INPUT_MAX_WIDTH + col];
    const decodedX = (normalized + 1) * 127.5;
    const expectedX = 10 + ((col + 0.5) * 160) / result.width - 0.5;
    assert.ok(
      Math.abs(decodedX - expectedX) < 1.5,
      `col ${col}: decoded ${decodedX}, expected ~${expectedX}`
    );
    // G and B channels of the gradient are zero.
    assert.ok(Math.abs(out[plane + midRow * REC_INPUT_MAX_WIDTH + col] + 1) < 0.02);
  }

  // Padding beyond targetW stays zero.
  assert.equal(out[midRow * REC_INPUT_MAX_WIDTH + result.width + 5], 0);
});

test("warpQuadToStrip rejects degenerate quads", () => {
  const image = { data: new Uint8ClampedArray(16), width: 2, height: 2 };
  const out = new Float32Array(3 * REC_INPUT_HEIGHT * REC_INPUT_MAX_WIDTH);
  assert.equal(warpQuadToStrip(image, [0, 0, 0, 0, 0, 0, 0, 0], out), null);
});

test("expandQuad grows an axis-aligned quad by the given per-side fractions", () => {
  const quad = [100, 50, 300, 50, 300, 70, 100, 70]; // 200×20
  const grown = expandQuad(quad, 0.05, 0.1);
  assert.deepEqual(grown.map((v) => Math.round(v * 1000) / 1000), [90, 48, 310, 48, 310, 72, 90, 72]);
  assert.deepEqual(expandQuad(quad, 0, 0), quad);
});

test("expandQuad keeps a rotated quad's shape (edge lengths scale, angle preserved)", () => {
  const c = Math.cos(0.3), s = Math.sin(0.3);
  const rot = (x, y) => [x * c - y * s, x * s + y * c];
  const quad = [...rot(0, 0), ...rot(200, 0), ...rot(200, 20), ...rot(0, 20)];
  const g = expandQuad(quad, 0.05, 0.1);
  const len = (i, j) => Math.hypot(g[j] - g[i], g[j + 1] - g[i + 1]);
  assert.ok(Math.abs(len(0, 2) - 220) < 1e-6);
  assert.ok(Math.abs(len(0, 6) - 24) < 1e-6);
  assert.ok(Math.abs(Math.atan2(g[3] - g[1], g[2] - g[0]) - 0.3) < 1e-9);
});
