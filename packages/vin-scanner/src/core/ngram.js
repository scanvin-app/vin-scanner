/**
 * Positional n-gram VIN likelihood scorer + OCR-candidate rescorer — pure JS.
 *
 * Consumes the quantized model produced by ngram/train.mjs (trained on the
 * Danish DMR + USA registry corpora): interpolated positional trigram for
 * positions 1-11 (WMI + descriptor + year + plant — the structural part),
 * per-position unigram for the serial (12-17).
 *
 * Two roles:
 *  - scoreVin(text): mean log2 P per char — a soft plausibility signal.
 *  - rescoreRead(read): given a CTC read with per-char top-2 alternates,
 *    enumerate substitution variants at the most contested positions and
 *    pick the one maximizing OCR log-prob + n-gram log-prob (+ a check-digit
 *    bonus for North-American VINs). This is a RERANKER: it repairs single
 *    misreads (5↔S, 8↔B…) before temporal consensus, never rejects.
 */

export const DEFAULT_NGRAM_CONFIG = Object.freeze({
  /** Weight of the n-gram log2 term vs the OCR log2 term. */
  ngramWeight: 1.0,
  /** Max contested positions considered per read (variants = 2^n). */
  maxSwapPositions: 5,
  /** Ignore alternates below this softmax probability. */
  minAltConf: 0.02,
  /** Bits granted when a variant passes ISO 3779 (NA VINs only, 1st char 1-5). */
  checkDigitBonus: 4,
  /**
   * Length repair (18→17 deletion, 16→17 insertion) is accepted only when
   * the best candidate beats the runner-up by this many bits. Insertions in
   * the near-random serial tail can't clear it (all digits score alike) —
   * exactly the cases where repair would be a blind guess.
   */
  lengthRepairMargin: 3,
  /** Confidence assigned to a character synthesized by insertion repair. */
  insertedCharConf: 0.3,
});

/**
 * @param {object} model  Parsed vin-ngram-model.json.
 * @param {(vin: string) => boolean} [checkDigitOk]  Optional ISO 3779 validator.
 * @param {Partial<typeof DEFAULT_NGRAM_CONFIG>} [overrides]
 */
export function createNgramScorer(model, checkDigitOk, overrides = {}) {
  const config = { ...DEFAULT_NGRAM_CONFIG, ...overrides };
  const { alphabet, structLen, lambda, quantScale, uni, bi, tri } = model;
  const A = alphabet.length;
  const floor = lambda.floor ?? 0.01 / A;
  const deq = (q) => (q === undefined ? 0 : 2 ** (-q / quantScale));

  /**
   * log2 P(char c at position p | two preceding chars). The incremental
   * primitive — used both for whole-string likelihood and for shallow
   * fusion inside the CTC beam decoder. Positions beyond the VIN are
   * impossible (-Infinity) so a fused beam can never grow past 17.
   *
   * @param {string} c
   * @param {number} p  0-based position.
   * @param {string} [c1]  Char at p-2.
   * @param {string} [c2]  Char at p-1.
   */
  function charLogProb(c, p, c1, c2) {
    if (p >= uni.length) return -Infinity;
    const pUni = deq(uni[p]?.[c]);
    if (p >= structLen) {
      return Math.log2(0.99 * pUni + 0.01 / A);
    }
    const pTri = p >= 2 ? deq(tri[p][c1 + c2]?.[c]) : 0;
    const pBi = p >= 1 ? deq(bi[p][c2]?.[c]) : 0;
    return Math.log2(lambda.tri * pTri + lambda.bi * pBi + lambda.uni * pUni + floor);
  }

  /**
   * @param {string} text  17 legal VIN chars.
   * @returns {number} total log2 likelihood (struct interpolated + serial unigram).
   */
  function logLikelihood(text) {
    let sum = 0;
    for (let p = 0; p < text.length; p += 1) {
      sum += charLogProb(text[p], p, text[p - 2], text[p - 1]);
    }
    return sum;
  }

  /**
   * @param {string} text
   * @returns {number} mean log2 P per char — comparable against a fixed threshold.
   */
  function scoreVin(text) {
    return logLikelihood(text) / text.length;
  }

  const log2Conf = (conf) => Math.log2(Math.max(conf, 1e-6));

  /**
   * Rerank a CTC read against its per-char alternates.
   *
   * @param {{ text: string, charConfs: number[], meanConf: number,
   *           alts?: Array<{char: string, conf: number}|null> }} read
   * @returns {typeof read & { ngramScore: number, swaps: number }} the best
   *   variant (the input read itself when nothing beats it).
   */
  function rescoreRead(read) {
    if (read.text.length === 18 || read.text.length === 19) {
      const repaired = repairByDeletion(read);
      if (!repaired) return { ...read, ngramScore: scoreVin(read.text), swaps: 0 };
      read = repaired;
    } else if (read.text.length === 16) {
      const repaired = repairByInsertion(read);
      if (!repaired) return { ...read, ngramScore: scoreVin(read.text), swaps: 0 };
      read = repaired;
    }

    const { text, charConfs, alts } = read;
    if (!alts || text.length !== 17) {
      return { ...read, ngramScore: scoreVin(text), swaps: read.swaps ?? 0 };
    }

    // Contested positions, most ambiguous first.
    const spots = [];
    for (let i = 0; i < text.length; i += 1) {
      const alt = alts[i];
      if (!alt || !alt.char || alt.conf < config.minAltConf) continue;
      spots.push({ i, ratio: alt.conf / Math.max(charConfs[i] ?? 1, 1e-6) });
    }
    spots.sort((a, b) => b.ratio - a.ratio);
    const chosen = spots.slice(0, config.maxSwapPositions);

    let baseOcr = 0;
    for (let i = 0; i < text.length; i += 1) baseOcr += log2Conf(charConfs[i] ?? read.meanConf);

    const evaluate = (candidate, ocrLog) => {
      let s = ocrLog + config.ngramWeight * logLikelihood(candidate);
      if (candidate[0] >= "1" && candidate[0] <= "5" && checkDigitOk?.(candidate)) {
        s += config.checkDigitBonus;
      }
      return s;
    };

    let best = { text, ocrLog: baseOcr, score: evaluate(text, baseOcr), mask: 0 };
    const combos = 1 << chosen.length;
    for (let mask = 1; mask < combos; mask += 1) {
      const chars = [...text];
      let ocrLog = baseOcr;
      for (let b = 0; b < chosen.length; b += 1) {
        if (!(mask & (1 << b))) continue;
        const { i } = chosen[b];
        chars[i] = alts[i].char;
        ocrLog += log2Conf(alts[i].conf) - log2Conf(charConfs[i] ?? read.meanConf);
      }
      const candidate = chars.join("");
      const score = evaluate(candidate, ocrLog);
      if (score > best.score) best = { text: candidate, ocrLog, score, mask };
    }

    if (best.mask === 0) {
      return { ...read, ngramScore: scoreVin(text), swaps: read.swaps ?? 0 };
    }

    const outConfs = [...charConfs];
    let swaps = read.swaps ?? 0;
    for (let b = 0; b < chosen.length; b += 1) {
      if (!(best.mask & (1 << b))) continue;
      outConfs[chosen[b].i] = alts[chosen[b].i].conf;
      swaps += 1;
    }
    return {
      text: best.text,
      charConfs: outConfs,
      meanConf: outConfs.reduce((a, c) => a + c, 0) / outConfs.length,
      alts,
      ngramScore: scoreVin(best.text),
      swaps,
    };
  }

  /**
   * 18/19-char read → best 17-char variant by deleting 1-2 chars. All true
   * characters are present, the n-gram only chooses which extras to drop
   * (a shifted structural part scores catastrophically, so the correct
   * deletion wins by a wide margin). null when ambiguous.
   */
  function repairByDeletion(read) {
    const { text, charConfs, alts } = read;
    const candidates = [];
    const pushCandidate = (drop) => {
      let t = "";
      const confs = [];
      const a = [];
      let ocrLog = 0;
      for (let i = 0; i < text.length; i += 1) {
        if (drop.includes(i)) continue;
        t += text[i];
        confs.push(charConfs[i]);
        a.push(alts?.[i] ?? null);
        ocrLog += log2Conf(charConfs[i] ?? read.meanConf);
      }
      candidates.push({
        text: t,
        charConfs: confs,
        alts: a,
        score: ocrLog + config.ngramWeight * logLikelihood(t),
        edgeOnly: drop.every((i) => i === 0 || i === text.length - 1),
      });
    };
    if (text.length === 18) {
      for (let i = 0; i < 18; i += 1) pushCandidate([i]);
    } else {
      for (let i = 0; i < 19; i += 1) for (let j = i + 1; j < 19; j += 1) pushCandidate([i, j]);
    }
    return pickWithMargin(candidates, read);
  }

  /**
   * 16-char read → best 17-char variant by inserting one char. Reliable only
   * when the gap is in the structural part (positions 1-11) where the n-gram
   * pins the missing char; in the serial every digit scores alike and the
   * margin check rejects the repair.
   */
  function repairByInsertion(read) {
    const { text, charConfs, alts } = read;
    const candidates = [];
    for (let i = 0; i <= 16; i += 1) {
      for (const c of alphabet) {
        const t = text.slice(0, i) + c + text.slice(i);
        let ocrLog = log2Conf(config.insertedCharConf);
        for (let k = 0; k < 16; k += 1) ocrLog += log2Conf(charConfs[k] ?? read.meanConf);
        candidates.push({
          text: t,
          charConfs: [...charConfs.slice(0, i), config.insertedCharConf, ...charConfs.slice(i)],
          alts: alts ? [...alts.slice(0, i), null, ...alts.slice(i)] : null,
          score: ocrLog + config.ngramWeight * logLikelihood(t),
        });
      }
    }
    return pickWithMargin(candidates, read);
  }

  function pickWithMargin(candidates, read) {
    let best = null;
    let runnerUp = null;
    for (const c of candidates) {
      if (!best || c.score > best.score) {
        if (best && (!runnerUp || best.score > runnerUp.score)) runnerUp = best;
        best = c;
      } else if ((!runnerUp || c.score > runnerUp.score) && c.text !== best.text) {
        runnerUp = c;
      }
    }
    if (!best) return null;
    if (runnerUp && runnerUp.text !== best.text && best.score - runnerUp.score < config.lengthRepairMargin) {
      // Ambiguous by likelihood alone (ties happen inside the serial, where
      // every digit scores alike). OCR insertions come from neighbouring
      // stamped characters at the strip edges — when exactly one contender
      // is a pure edge deletion, that asymmetry settles the tie.
      const contenders = candidates.filter((c) => best.score - c.score < config.lengthRepairMargin);
      const edge = contenders.filter((c) => c.edgeOnly);
      if (edge.length !== 1) return null;
      best = edge[0];
    }
    return {
      text: best.text,
      charConfs: best.charConfs,
      meanConf: best.charConfs.reduce((a, c) => a + c, 0) / best.charConfs.length,
      alts: best.alts,
      swaps: (read.swaps ?? 0) + 1,
    };
  }

  return { scoreVin, logLikelihood, charLogProb, rescoreRead, config };
}
