import { test } from "node:test";
import assert from "node:assert/strict";

import { isVinShape, computeCheckDigit, checkDigitOk, wmiKnown } from "../src/core/vin.js";

test("isVinShape accepts legal 17-char VINs", () => {
  assert.equal(isVinShape("1M8GDM9AXKP042788"), true);
  assert.equal(isVinShape("WVWZZZ1KZ8W123456"), true);
});

test("isVinShape rejects wrong length and illegal chars", () => {
  assert.equal(isVinShape("1M8GDM9AXKP04278"), false); // 16 chars
  assert.equal(isVinShape("1M8GDM9AXKP0427889"), false); // 18 chars
  assert.equal(isVinShape("IM8GDM9AXKP042788"), false); // I illegal
  assert.equal(isVinShape("OM8GDM9AXKP042788"), false); // O illegal
  assert.equal(isVinShape("QM8GDM9AXKP042788"), false); // Q illegal
  assert.equal(isVinShape(""), false);
  assert.equal(isVinShape(null), false);
});

test("computeCheckDigit matches ISO 3779 reference vectors", () => {
  // Classic reference VIN with check digit X.
  assert.equal(computeCheckDigit("1M8GDM9AXKP042788"), "X");
  // All-ones VIN: weighted sum 89, 89 % 11 = 1.
  assert.equal(computeCheckDigit("11111111111111111"), "1");
  assert.equal(computeCheckDigit("not a vin"), null);
});

test("checkDigitOk verifies position 9", () => {
  assert.equal(checkDigitOk("1M8GDM9AXKP042788"), true);
  assert.equal(checkDigitOk("11111111111111111"), true);
  assert.equal(checkDigitOk("1M8GDM9A1KP042788"), false); // tampered check digit
});

test("wmiKnown recognizes known manufacturer prefixes", () => {
  assert.equal(wmiKnown("JMZGJ627601234567"), true); // Mazda
  assert.equal(wmiKnown("WVWZZZ1KZ8W123456"), true); // Volkswagen
  assert.equal(wmiKnown("00XZZZZZZZZZZZZZZ"), false);
});
