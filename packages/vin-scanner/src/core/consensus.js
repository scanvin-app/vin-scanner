/**
 * Temporal consensus over per-frame OCR reads — pure JS.
 *
 * Replaces the legacy single-frame heuristics (soft substitutions, line
 * stitching, length correction, 0–100 scoring): instead of repairing one
 * noisy read, we accumulate cheap reads across video frames and accept a
 * VIN only when independent frames agree.
 *
 * Acceptance tiers (first match wins, evaluated after every counted read):
 *  A — fast: the same string seen twice with high confidence AND an
 *      external signal (ISO 3779 check digit or known WMI).
 *  B — standard: the same string seen `tierBCount` times with decent
 *      average confidence.
 *  C — per-position majority: when no exact string repeats (single
 *      character flips like 5↔S across frames), vote per position
 *      weighted by per-character confidence.
 */

import { VIN_LENGTH, isVinShape, checkDigitOk, wmiKnown } from "./vin.js";

export const DEFAULT_CONSENSUS_CONFIG = Object.freeze({
  /** Reads below this mean confidence are ignored entirely. */
  minMeanConf: 0.35,
  /** Gap between counted reads (ms) after which accumulated state resets. */
  resetGapMs: 2000,
  /** Cap on retained reads (memory bound; oldest dropped). */
  maxReads: 24,

  tierACount: 2,
  tierAMinConf: 0.85,
  /**
   * Tier A's external signal. The ISO 3779 check digit covers all 17
   * positions; a known WMI vouches only for the first 3 and proved too weak
   * (benchmark: a repeated 0→8 flip in the serial passed tier A via WMI).
   */
  tierAAllowWmi: false,

  tierBCount: 3,
  tierBMinConf: 0.6,

  tierCMinReads: 5,
  /** Fewer reads suffice for tier C when the majority string passes the checksum. */
  tierCMinReadsChecksum: 4,
  /** Winning char weight must exceed runner-up by this factor at every position. */
  tierCMargin: 1.5,
  /** Winning char must appear in at least this fraction of reads at its position. */
  tierCPresence: 0.5,
});

/**
 * @typedef {{ text: string, charConfs: number[], meanConf: number }} Read
 * @typedef {{ vin: string, tier: "A"|"B"|"C", reads: number, confidence: number,
 *             checkDigitOk: boolean, wmiKnown: boolean }} Acceptance
 */

/**
 * @param {Partial<typeof DEFAULT_CONSENSUS_CONFIG>} [overrides]
 */
export function createConsensusTracker(overrides = {}) {
  const config = { ...DEFAULT_CONSENSUS_CONFIG, ...overrides };

  /** @type {Array<Read & { ts: number }>} */
  let reads = [];
  /** @type {Map<string, { count: number, confSum: number }>} */
  let exact = new Map();
  /** @type {Array<Map<string, { weight: number, count: number }>>} */
  let pos = [];
  let lastCountedTs = -Infinity;

  function reset() {
    reads = [];
    exact = new Map();
    pos = Array.from({ length: VIN_LENGTH }, () => new Map());
    lastCountedTs = -Infinity;
  }
  reset();

  /**
   * Feed one OCR read. Returns an acceptance when consensus is reached.
   *
   * @param {Read} read
   * @param {number} ts  Monotonic timestamp in ms.
   * @returns {Acceptance|null}
   */
  function add(read, ts) {
    const text = typeof read?.text === "string" ? read.text.trim().toUpperCase() : "";
    if (!isVinShape(text) || !(read.meanConf >= config.minMeanConf)) {
      return null;
    }

    if (ts - lastCountedTs > config.resetGapMs) {
      reset();
    }
    lastCountedTs = ts;

    reads.push({ text, charConfs: read.charConfs ?? [], meanConf: read.meanConf, ts });
    if (reads.length > config.maxReads) {
      const dropped = reads.shift();
      removeFromTallies(dropped);
    }
    addToTallies({ text, charConfs: read.charConfs ?? [], meanConf: read.meanConf });

    return evaluate();
  }

  function addToTallies(read) {
    const entry = exact.get(read.text) ?? { count: 0, confSum: 0 };
    entry.count += 1;
    entry.confSum += read.meanConf;
    exact.set(read.text, entry);

    for (let i = 0; i < VIN_LENGTH; i += 1) {
      const char = read.text[i];
      const conf = Number.isFinite(read.charConfs[i]) ? read.charConfs[i] : read.meanConf;
      const cell = pos[i].get(char) ?? { weight: 0, count: 0 };
      cell.weight += conf;
      cell.count += 1;
      pos[i].set(char, cell);
    }
  }

  function removeFromTallies(read) {
    const entry = exact.get(read.text);
    if (entry) {
      entry.count -= 1;
      entry.confSum -= read.meanConf;
      if (entry.count <= 0) exact.delete(read.text);
    }
    for (let i = 0; i < VIN_LENGTH; i += 1) {
      const char = read.text[i];
      const conf = Number.isFinite(read.charConfs[i]) ? read.charConfs[i] : read.meanConf;
      const cell = pos[i].get(char);
      if (cell) {
        cell.weight -= conf;
        cell.count -= 1;
        if (cell.count <= 0) pos[i].delete(char);
      }
    }
  }

  /** @returns {Acceptance|null} */
  function evaluate() {
    // Tiers A and B — exact string repetition.
    for (const [text, { count, confSum }] of exact) {
      const meanConf = confSum / count;
      const external = checkDigitOk(text) || (config.tierAAllowWmi && wmiKnown(text));

      if (count >= config.tierACount && external && meanConf >= config.tierAMinConf) {
        return accepted(text, "A", meanConf);
      }
      if (count >= config.tierBCount && meanConf >= config.tierBMinConf) {
        return accepted(text, "B", meanConf);
      }
    }

    // Tier C — per-position weighted majority.
    if (reads.length >= config.tierCMinReadsChecksum) {
      const majority = majorityString();
      if (majority) {
        const enough =
          reads.length >= config.tierCMinReads ||
          (reads.length >= config.tierCMinReadsChecksum && checkDigitOk(majority.text));
        if (enough && isVinShape(majority.text)) {
          return accepted(majority.text, "C", majority.confidence);
        }
      }
    }

    return null;
  }

  /** @returns {{ text: string, confidence: number }|null} */
  function majorityString() {
    let text = "";
    let confSum = 0;

    for (let i = 0; i < VIN_LENGTH; i += 1) {
      let top = null;
      let runnerUp = null;
      for (const [char, cell] of pos[i]) {
        if (!top || cell.weight > top.weight) {
          if (top && (!runnerUp || top.weight > runnerUp.weight)) runnerUp = top;
          top = { char, ...cell };
        } else if (!runnerUp || cell.weight > runnerUp.weight) {
          runnerUp = { char, ...cell };
        }
      }
      if (!top) return null;
      if (runnerUp) {
        if (top.weight < config.tierCMargin * runnerUp.weight) return null;
        // Contested position: the runner-up is read with higher per-char
        // confidence than the weight winner — the signature of a systematic
        // misread (e.g. glare turning 0 into 8 frame after frame, always at
        // lower confidence than the clean reads). Repetition count must not
        // decide here; hold tier C and let tier B (exact repeats) settle it.
        if (runnerUp.weight / runnerUp.count > top.weight / top.count) {
          return null;
        }
      }
      if (top.count / reads.length < config.tierCPresence) return null;
      text += top.char;
      confSum += top.weight / top.count;
    }

    return { text, confidence: confSum / VIN_LENGTH };
  }

  function accepted(text, tier, confidence) {
    return {
      vin: text,
      tier,
      reads: reads.length,
      confidence,
      checkDigitOk: checkDigitOk(text),
      wmiKnown: wmiKnown(text),
    };
  }

  return {
    add,
    reset,
    get config() {
      return config;
    },
    /** Number of currently accumulated (counted) reads. */
    get size() {
      return reads.length;
    },
  };
}
