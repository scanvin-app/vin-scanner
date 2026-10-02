import { test } from "node:test";
import assert from "node:assert/strict";

import { createCtcDecoder } from "../src/core/ctc.js";

const DICT = "A\nB\nC";

test("decodes probability rows with dedup, blanks and per-char confidence", () => {
  const decoder = createCtcDecoder(DICT);
  assert.equal(decoder.numClasses, 4); // blank + A B C

  // Rows: [blank, A, B, C], already probabilities (first row sums to 1).
  const rows = [
    [0.9, 0.05, 0.03, 0.02], // blank
    [0.1, 0.8, 0.05, 0.05], // A (0.8)
    [0.2, 0.7, 0.05, 0.05], // A duplicate (run keeps max 0.8)
    [0.97, 0.01, 0.01, 0.01], // blank
    [0.2, 0.1, 0.1, 0.6], // C (0.6)
    [0.05, 0.02, 0.03, 0.9], // C duplicate (run max → 0.9)
  ];
  const preds = new Float32Array(rows.flat());

  const read = decoder.decode(preds, rows.length, 4);
  assert.equal(read.text, "AC");
  assert.equal(read.charConfs.length, 2);
  assert.ok(Math.abs(read.charConfs[0] - 0.8) < 1e-6);
  assert.ok(Math.abs(read.charConfs[1] - 0.9) < 1e-6);
  assert.ok(Math.abs(read.meanConf - 0.85) < 1e-6);
});

test("blank-separated repeats stay distinct characters", () => {
  const decoder = createCtcDecoder(DICT);
  const rows = [
    [0.1, 0.8, 0.05, 0.05], // A
    [0.9, 0.05, 0.03, 0.02], // blank
    [0.1, 0.8, 0.05, 0.05], // A again → "AA"
  ];
  const read = decoder.decode(new Float32Array(rows.flat()), rows.length, 4);
  assert.equal(read.text, "AA");
});

test("applies softmax to raw logits", () => {
  const decoder = createCtcDecoder(DICT);
  // Single timestep of logits (sums ≠ 1): argmax A with prob e^5/(e^5+3).
  const preds = new Float32Array([0, 5, 0, 0]);
  const read = decoder.decode(preds, 1, 4);
  assert.equal(read.text, "A");
  const expected = Math.exp(5) / (Math.exp(5) + 3);
  assert.ok(Math.abs(read.charConfs[0] - expected) < 1e-4);
});

test("empty output yields empty text with zero confidence", () => {
  const decoder = createCtcDecoder(DICT);
  const rows = [
    [0.9, 0.05, 0.03, 0.02],
    [0.9, 0.05, 0.03, 0.02],
  ];
  const read = decoder.decode(new Float32Array(rows.flat()), rows.length, 4);
  assert.equal(read.text, "");
  assert.equal(read.meanConf, 0);
});
