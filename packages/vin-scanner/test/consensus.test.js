import { test } from "node:test";
import assert from "node:assert/strict";

import { createConsensusTracker } from "../src/core/consensus.js";

// Reference VIN with a valid ISO 3779 check digit ("X" at position 9).
const VALID_VIN = "1M8GDM9AXKP042788";
// Shape-valid VIN that fails the checksum and has no known WMI.
const EU_LIKE_VIN = "XX9ZZZ1KZ8W123456";

const read = (text, meanConf, charConfs) => ({
  text,
  meanConf,
  charConfs: charConfs ?? Array(17).fill(meanConf),
});

test("tier A: two identical high-confidence reads with valid check digit", () => {
  const tracker = createConsensusTracker();
  assert.equal(tracker.add(read(VALID_VIN, 0.92), 0), null);
  const result = tracker.add(read(VALID_VIN, 0.9), 200);
  assert.ok(result);
  assert.equal(result.tier, "A");
  assert.equal(result.vin, VALID_VIN);
  assert.equal(result.checkDigitOk, true);
});

test("tier B: three identical reads without external signal", () => {
  const tracker = createConsensusTracker();
  assert.equal(tracker.add(read(EU_LIKE_VIN, 0.7), 0), null);
  assert.equal(tracker.add(read(EU_LIKE_VIN, 0.7), 200), null);
  const result = tracker.add(read(EU_LIKE_VIN, 0.7), 400);
  assert.ok(result);
  assert.equal(result.tier, "B");
  assert.equal(result.vin, EU_LIKE_VIN);
});

test("tier C: per-position majority resolves single-char flips", () => {
  // Confidence 0.5 keeps tiers A/B out of reach (tierBMinConf = 0.6).
  const flip = (vin, pos, char) => vin.slice(0, pos) + char + vin.slice(pos + 1);
  const v1 = flip(EU_LIKE_VIN, 5, "S"); // 1 → S style flip
  const v2 = flip(EU_LIKE_VIN, 10, "8");
  const v3 = flip(EU_LIKE_VIN, 12, "B");

  const tracker = createConsensusTracker();
  const sequence = [v1, v2, EU_LIKE_VIN, v3, EU_LIKE_VIN];
  let result = null;
  sequence.forEach((text, i) => {
    result = tracker.add(read(text, 0.5), i * 200);
  });

  assert.ok(result, "expected consensus after 5 reads");
  assert.equal(result.tier, "C");
  assert.equal(result.vin, EU_LIKE_VIN);
});

test("tier C does not fire on a contested position", () => {
  const flip = (vin, pos, char) => vin.slice(0, pos) + char + vin.slice(pos + 1);
  const contested = flip(EU_LIKE_VIN, 5, "S");

  const tracker = createConsensusTracker();
  // 6 reads, position 5 split 3/3 — margin 1.5× can never hold.
  const sequence = [EU_LIKE_VIN, contested, EU_LIKE_VIN, contested, EU_LIKE_VIN, contested];
  let result = null;
  sequence.forEach((text, i) => {
    result = tracker.add(read(text, 0.5), i * 200);
  });
  assert.equal(result, null);
});

test("non-VIN reads and low confidence are ignored", () => {
  const tracker = createConsensusTracker();
  assert.equal(tracker.add(read("GARBAGE", 0.9), 0), null);
  assert.equal(tracker.add(read("1M8GDM9AXKP04278", 0.9), 100), null); // 16 chars
  assert.equal(tracker.add(read(VALID_VIN, 0.2), 200), null); // below minMeanConf
  assert.equal(tracker.size, 0);
});

test("long gaps between reads reset accumulated state", () => {
  const tracker = createConsensusTracker();
  assert.equal(tracker.add(read(EU_LIKE_VIN, 0.7), 0), null);
  assert.equal(tracker.add(read(EU_LIKE_VIN, 0.7), 100), null);
  // 3 s gap — state resets, this read starts a fresh window.
  assert.equal(tracker.add(read(EU_LIKE_VIN, 0.7), 3100), null);
  assert.equal(tracker.size, 1);
});

test("systematic low-conf flip cannot win tier C over a high-conf minority", () => {
  // Regression from the registration-document benchmark clip: glare read
  // 0→8 at position 10 in 3 of 5 frames, but always at LOWER per-char
  // confidence than the clean 0 reads. Tier C must hold (contested
  // position) instead of accepting the repeated wrong string; the correct
  // string is then settled by tier B once it repeats 3×.
  const flip = (vin, pos, char) => vin.slice(0, pos) + char + vin.slice(pos + 1);
  const WRONG = flip(EU_LIKE_VIN, 10, "8");
  const OTHER = flip(EU_LIKE_VIN, 3, "B");

  const withPosConf = (text, meanConf, pos10Conf) => {
    const charConfs = Array(17).fill(meanConf);
    charConfs[10] = pos10Conf;
    return { text, meanConf, charConfs };
  };

  const tracker = createConsensusTracker();
  // meanConf 0.5 keeps tier B out of reach for the wrong string.
  assert.equal(tracker.add(withPosConf(WRONG, 0.5, 0.7), 0), null);
  assert.equal(tracker.add(withPosConf(WRONG, 0.5, 0.7), 200), null);
  assert.equal(tracker.add(withPosConf(EU_LIKE_VIN, 0.5, 0.95), 400), null);
  assert.equal(tracker.add(withPosConf(WRONG, 0.5, 0.7), 600), null);
  // 5th read — tier C evaluates, position 10 is contested → no acceptance.
  assert.equal(tracker.add(withPosConf(OTHER, 0.5, 0.95), 800), null);

  // Correct string repeats with decent confidence → tier B settles it.
  assert.equal(tracker.add(withPosConf(EU_LIKE_VIN, 0.7, 0.95), 1000), null);
  const result = tracker.add(withPosConf(EU_LIKE_VIN, 0.7, 0.95), 1200);
  assert.ok(result);
  assert.equal(result.tier, "B");
  assert.equal(result.vin, EU_LIKE_VIN);
});

test("garbage variety never converges", () => {
  const tracker = createConsensusTracker();
  const junk = [
    "AAAAAAAAAAAAAAAA1",
    "BBBBBBBBBBBBBBBB2",
    "CCCCCCCCCCCCCCCC3",
    "DDDDDDDDDDDDDDDD4",
    "EEEEEEEEEEEEEEEE5",
    "FFFFFFFFFFFFFFFF6",
  ];
  let result = null;
  junk.forEach((text, i) => {
    result = tracker.add(read(text, 0.6), i * 200);
  });
  assert.equal(result, null);
});
