/**
 * VIN validation primitives — pure JS.
 *
 * Deliberately minimal: charset shape check, ISO 3779 check digit and a
 * known-WMI lookup. The check digit and WMI are BONUS signals for the
 * temporal consensus (they accelerate acceptance) — never requirements,
 * because European VINs legitimately fail the North-American checksum.
 */

import { WMI_CODES } from "./wmi-prefixes.js";

export const VIN_LENGTH = 17;
export const VIN_PATTERN = /^[A-HJ-NPR-Z0-9]{17}$/;

const VIN_WEIGHTS = [8, 7, 6, 5, 4, 3, 2, 10, 0, 9, 8, 7, 6, 5, 4, 3, 2];

const TRANSLITERATION = {
  A: 1, B: 2, C: 3, D: 4, E: 5, F: 6, G: 7, H: 8,
  J: 1, K: 2, L: 3, M: 4, N: 5, P: 7, R: 9,
  S: 2, T: 3, U: 4, V: 5, W: 6, X: 7, Y: 8, Z: 9,
};

const WMI_SET = new Set(WMI_CODES);
const WMI_2CHAR_SET = new Set(WMI_CODES.map((code) => code.slice(0, 2)));

/**
 * @param {string} text
 * @returns {boolean} true when text has the exact VIN shape (17 legal chars).
 */
export function isVinShape(text) {
  return typeof text === "string" && VIN_PATTERN.test(text);
}

/**
 * Compute the ISO 3779 check digit for a shape-valid VIN.
 * @param {string} vin
 * @returns {string|null} "0"–"9" or "X", null for invalid input.
 */
export function computeCheckDigit(vin) {
  if (!isVinShape(vin)) return null;

  let sum = 0;
  for (let i = 0; i < VIN_LENGTH; i += 1) {
    const char = vin[i];
    const digit = Number.parseInt(char, 10);
    const value = Number.isNaN(digit) ? TRANSLITERATION[char] : digit;
    if (!Number.isFinite(value)) return null;
    sum += value * VIN_WEIGHTS[i];
  }

  const remainder = sum % 11;
  return remainder === 10 ? "X" : String(remainder);
}

/**
 * @param {string} vin
 * @returns {boolean} true when position 9 matches the ISO 3779 check digit.
 */
export function checkDigitOk(vin) {
  const expected = computeCheckDigit(vin);
  return expected !== null && vin[8] === expected;
}

/**
 * @param {string} vin
 * @returns {boolean} true when the VIN starts with a known WMI
 *   (exact 3-char code or a known 2-char manufacturer prefix).
 */
export function wmiKnown(vin) {
  if (typeof vin !== "string" || vin.length < 3) return false;
  return WMI_SET.has(vin.slice(0, 3)) || WMI_2CHAR_SET.has(vin.slice(0, 2));
}
