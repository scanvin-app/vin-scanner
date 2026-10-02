import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

import { createNgramScorer } from "../src/core/ngram.js";
import { checkDigitOk } from "../src/core/vin.js";

const model = JSON.parse(
  await readFile(new URL("../models/vin_ngram.json", import.meta.url), "utf8")
);
const scorer = createNgramScorer(model, checkDigitOk);

const GOOD = "VSSZZZ5FZN6000123"; // synthetic EU-style VIN (VW-group WMI/VDS, made-up serial)

function readFor(text, { corruptAt, altChar, altConf = 0.3, conf = 0.55 } = {}) {
  const chars = [...text];
  const charConfs = chars.map(() => 0.97);
  const alts = chars.map(() => null);
  if (corruptAt != null) {
    charConfs[corruptAt] = conf;
    alts[corruptAt] = { char: altChar, conf: altConf };
  }
  return {
    text: chars.join(""),
    charConfs,
    meanConf: charConfs.reduce((a, b) => a + b, 0) / charConfs.length,
    alts,
  };
}

test("scoreVin ranks a real VIN above an OCR-corrupted one", () => {
  const corrupted = GOOD.slice(0, 3) + "2" + GOOD.slice(4); // Z→2 at pos 4
  assert.ok(scorer.scoreVin(GOOD) > scorer.scoreVin(corrupted));
});

test("rescoreRead repairs a structural misread when the alt is the true char", () => {
  const corrupted = GOOD.slice(0, 3) + "2" + GOOD.slice(4);
  const read = readFor(corrupted, { corruptAt: 3, altChar: "Z" });
  const out = scorer.rescoreRead(read);
  assert.equal(out.text, GOOD);
  assert.equal(out.swaps, 1);
});

test("rescoreRead leaves a clean confident read untouched", () => {
  const read = readFor(GOOD);
  const out = scorer.rescoreRead(read);
  assert.equal(out.text, GOOD);
  assert.equal(out.swaps, 0);
});

test("rescoreRead does not touch reads of wrong length", () => {
  const read = readFor(GOOD.slice(0, 16));
  const out = scorer.rescoreRead(read);
  assert.equal(out.text, GOOD.slice(0, 16));
  assert.equal(out.swaps, 0);
});

test("rescoreRead deletes a trailing garbage char from an 18-char read", () => {
  // Typical stamped-VIN failure: a neighbouring character glued to the end.
  const read = readFor("WBA8H71060K9123456");
  const out = scorer.rescoreRead(read);
  assert.equal(out.text, "WBA8H71060K912345");
  assert.ok(out.swaps >= 1);
});

test("rescoreRead deletes an inserted char inside the structural part", () => {
  const read = readFor("WVBA8H71060K912345"); // stray V at position 2
  const out = scorer.rescoreRead(read);
  assert.equal(out.text, "WBA8H71060K912345");
});

test("rescoreRead restores a structural char dropped from a 16-char read", () => {
  const read = readFor("WB8H71060K912345"); // missing A of the WBA prefix
  const out = scorer.rescoreRead(read);
  assert.equal(out.text, "WBA8H71060K912345");
});

test("rescoreRead refuses to guess a missing serial digit", () => {
  const read = readFor("WBA8H71060K91234"); // serial truncated — any digit fits
  const out = scorer.rescoreRead(read);
  assert.equal(out.text, "WBA8H71060K91234");
  assert.equal(out.swaps, 0);
});

test("rescoreRead keeps a confident primary even with a plausible alt", () => {
  // High primary confidence: the OCR term should dominate a mild n-gram preference.
  const read = readFor(GOOD, { corruptAt: 15, altChar: "S", altConf: 0.02, conf: 0.98 });
  read.charConfs[15] = 0.98;
  const out = scorer.rescoreRead(read);
  assert.equal(out.text, GOOD);
});
