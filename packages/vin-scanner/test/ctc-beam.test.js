import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

import { createBeamDecoder } from "../src/core/ctc-beam.js";
import { createNgramScorer } from "../src/core/ngram.js";
import { checkDigitOk } from "../src/core/vin.js";

const ALPHABET = "0123456789ABCDEFGHJKLMNPRSTUVWXYZ";
const dictText = [...ALPHABET].join("\n");
const CLASSES = ALPHABET.length + 1; // + CTC blank at index 0

const model = JSON.parse(
  await readFile(new URL("../models/vin_ngram.json", import.meta.url), "utf8")
);
const scorer = createNgramScorer(model, checkDigitOk);
const decoder = createBeamDecoder(dictText, scorer);

const classOf = (ch) => ALPHABET.indexOf(ch) + 1;

/**
 * Build a probability lattice: per char two timesteps — an emission row and
 * a blank row. `noise` maps position → { wrongChar, wrongProb } putting the
 * wrong char on top at that position.
 */
function lattice(vin, noise = {}) {
  const rows = [];
  for (let i = 0; i < vin.length; i += 1) {
    const emit = new Float32Array(CLASSES).fill(1e-6);
    const n = noise[i];
    if (n) {
      emit[classOf(n.wrongChar)] = n.wrongProb;
      emit[classOf(vin[i])] = 1 - n.wrongProb - 1e-4;
    } else {
      emit[classOf(vin[i])] = 0.98;
      emit[0] = 0.02 - 1e-4;
    }
    const blank = new Float32Array(CLASSES).fill(1e-6);
    blank[0] = 0.99;
    rows.push(emit, blank);
  }
  const preds = new Float32Array(rows.length * CLASSES);
  rows.forEach((row, t) => preds.set(row, t * CLASSES));
  return { preds, timeSteps: rows.length };
}

const GOOD = "VSSZZZ5FZN6000123";

test("beam decodes a clean lattice exactly", () => {
  const { preds, timeSteps } = lattice(GOOD);
  const read = decoder.decode(preds, timeSteps, CLASSES);
  assert.equal(read.text, GOOD);
  assert.ok(read.meanConf > 0.9);
});

test("shallow fusion flips a structurally impossible top-1 char", () => {
  // Position 4: OCR slightly prefers '2' over the true 'Z' (0.55 vs 0.45) —
  // pure CTC would take '2'; the n-gram knows VSSZ→Z and flips it.
  const { preds, timeSteps } = lattice(GOOD, { 3: { wrongChar: "2", wrongProb: 0.55 } });
  const read = decoder.decode(preds, timeSteps, CLASSES);
  assert.equal(read.text, GOOD);
});

test("beam keeps a confident char the LM merely dislikes", () => {
  // Serial position with an overwhelming emission — LM must not override it.
  const { preds, timeSteps } = lattice(GOOD, { 15: { wrongChar: "8", wrongProb: 0.999 } });
  const read = decoder.decode(preds, timeSteps, CLASSES);
  assert.equal(read.text, GOOD.slice(0, 15) + "8" + GOOD.slice(16));
});
