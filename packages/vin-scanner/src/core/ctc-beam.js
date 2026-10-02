/**
 * CTC prefix beam search with n-gram shallow fusion — pure JS.
 *
 * Drop-in alternative to the greedy decoder in ctc.js (same decode()
 * signature and RecRead result). Instead of taking the argmax at every
 * timestep and repairing mistakes post-hoc, it keeps `beamWidth` candidate
 * prefixes over the full lattice (including blank probabilities, so
 * insertions/deletions resolve naturally) and adds the positional VIN
 * n-gram as a language model each time a prefix grows by a character:
 *
 *   score(prefix) = logP_ctc(prefix) + lmWeight · logP_ngram(prefix)
 *
 * The n-gram returns -Infinity past position 17, so beams can never grow
 * longer than a VIN. At the end the best 17-char prefix wins unless a
 * shorter prefix beats it by more than `fullLengthMargin` nats — the guard
 * against hallucinating two chars that were never in the frame.
 */

/**
 * @param {string} dictText  Dictionary file contents, one character per line.
 * @param {import("./ngram.js").createNgramScorer extends (...a: any) => infer R ? R : never} ngram
 * @param {{ beamWidth?: number, topK?: number, lmWeight?: number,
 *           fullLengthMargin?: number }} [opts]
 * @returns {{ decode: (preds: Float32Array, timeSteps: number, numClasses: number)
 *               => import("./ctc.js").RecRead, numClasses: number }}
 */
export function createBeamDecoder(dictText, ngram, opts = {}) {
  const chars = String(dictText)
    .split("\n")
    .map((line) => line.replace(/\r$/, ""))
    .filter((line) => line.length > 0);
  const characters = ["", ...chars]; // index 0 = CTC blank
  const numClasses = characters.length;

  const beamWidth = opts.beamWidth ?? 16;
  const topK = opts.topK ?? 6;
  const lmWeight = opts.lmWeight ?? 0.8;
  const fullLengthMargin = opts.fullLengthMargin ?? 10;
  const LN2 = Math.LN2;

  const NEG_INF = -Infinity;
  const logSum = (a, b) => {
    if (a === NEG_INF) return b;
    if (b === NEG_INF) return a;
    const m = Math.max(a, b);
    return m + Math.log(Math.exp(a - m) + Math.exp(b - m));
  };

  /**
   * @typedef {{ text: string, pb: number, pnb: number, lm: number,
   *             confs: number[] }} Beam
   *   pb/pnb: log CTC probability of the prefix ending in blank / non-blank.
   *   lm: accumulated weighted LM log-prob (nats).
   *   confs: per-char best emission probability (for the consensus tracker).
   */

  function decode(preds, timeSteps, classes) {
    const logRow = new Float64Array(classes);
    const order = new Int32Array(classes);
    // Rows may be probabilities (sum≈1) or raw logits — same check as ctc.js.
    let rowSum = 0;
    for (let c = 0; c < Math.min(classes, preds.length); c += 1) rowSum += preds[c];
    const isProb = Math.abs(rowSum - 1.0) < 0.01;

    /** @type {Map<string, Beam>} */
    let beams = new Map();
    beams.set("", { text: "", pb: 0, pnb: NEG_INF, lm: 0, confs: [] });

    for (let t = 0; t < timeSteps; t += 1) {
      const off = t * classes;

      if (isProb) {
        for (let c = 0; c < classes; c += 1) logRow[c] = Math.log(Math.max(preds[off + c], 1e-12));
      } else {
        // Log-softmax of the logit row.
        let max = -Infinity;
        for (let c = 0; c < classes; c += 1) if (preds[off + c] > max) max = preds[off + c];
        let sumExp = 0;
        for (let c = 0; c < classes; c += 1) sumExp += Math.exp(preds[off + c] - max);
        const logZ = max + Math.log(sumExp);
        for (let c = 0; c < classes; c += 1) logRow[c] = preds[off + c] - logZ;
      }

      // Blank + topK non-blank classes.
      for (let c = 0; c < classes; c += 1) order[c] = c;
      order.subarray(1).sort((a, b) => logRow[b] - logRow[a]);
      const kEnd = Math.min(1 + topK, classes);

      /** @type {Map<string, Beam>} */
      const next = new Map();
      const merge = (text, pb, pnb, lm, confs) => {
        const hit = next.get(text);
        if (!hit) {
          next.set(text, { text, pb, pnb, lm, confs });
          return;
        }
        hit.pb = logSum(hit.pb, pb);
        hit.pnb = logSum(hit.pnb, pnb);
        // Same text ⇒ same LM score; keep the stronger conf estimate.
        for (let i = 0; i < hit.confs.length; i += 1) {
          if (confs[i] > hit.confs[i]) hit.confs[i] = confs[i];
        }
      };

      for (const beam of beams.values()) {
        const total = logSum(beam.pb, beam.pnb);
        const last = beam.text.length > 0 ? beam.text[beam.text.length - 1] : null;

        // Blank: prefix unchanged, now ends in blank.
        merge(beam.text, total + logRow[0], NEG_INF, beam.lm, beam.confs);

        for (let k = 1; k < kEnd; k += 1) {
          const ci = order[k];
          if (ci === 0) continue;
          const ch = characters[ci];
          const lp = logRow[ci];
          const emitProb = Math.exp(lp);

          if (ch === last) {
            // Continue the same run: prefix unchanged, ends non-blank.
            const confs = [...beam.confs];
            if (emitProb > confs[confs.length - 1]) confs[confs.length - 1] = emitProb;
            merge(beam.text, NEG_INF, beam.pnb + lp, beam.lm, confs);
            // New occurrence of the same char needs a blank gap: from pb only.
            if (beam.pb !== NEG_INF) {
              const lm2 = charLm(beam, ch);
              if (lm2 !== NEG_INF) {
                merge(beam.text + ch, NEG_INF, beam.pb + lp, beam.lm + lm2, [...beam.confs, emitProb]);
              }
            }
          } else {
            const lm2 = charLm(beam, ch);
            if (lm2 !== NEG_INF) {
              merge(beam.text + ch, NEG_INF, total + lp, beam.lm + lm2, [...beam.confs, emitProb]);
            }
          }
        }
      }

      // Prune to beamWidth by combined CTC+LM score.
      const ranked = [...next.values()].sort(
        (a, b) => logSum(b.pb, b.pnb) + b.lm - (logSum(a.pb, a.pnb) + a.lm)
      );
      beams = new Map();
      for (let i = 0; i < Math.min(beamWidth, ranked.length); i += 1) {
        beams.set(ranked[i].text, ranked[i]);
      }
    }

    // Prefer the best full-length (17) prefix unless a shorter one wins by a
    // clear margin — then the frame genuinely doesn't contain a whole VIN.
    let bestAll = null;
    let best17 = null;
    for (const beam of beams.values()) {
      const score = logSum(beam.pb, beam.pnb) + beam.lm;
      if (!bestAll || score > bestAll.score) bestAll = { beam, score };
      if (beam.text.length === 17 && (!best17 || score > best17.score)) best17 = { beam, score };
    }
    const pick =
      best17 && bestAll && best17.score >= bestAll.score - fullLengthMargin ? best17 : bestAll;

    if (!pick) return { text: "", charConfs: [], meanConf: 0, alts: [] };
    const { text, confs } = pick.beam;
    const meanConf =
      confs.length > 0 ? confs.reduce((a, b) => a + b, 0) / confs.length : 0;
    return { text, charConfs: [...confs], meanConf, alts: confs.map(() => null) };
  }

  /** Weighted LM contribution (nats) for appending ch to a beam's text. */
  function charLm(beam, ch) {
    const p = beam.text.length;
    const lp2 = ngram.charLogProb(ch, p, beam.text[p - 2], beam.text[p - 1]);
    return lp2 === NEG_INF ? NEG_INF : lmWeight * lp2 * LN2;
  }

  return { decode, numClasses };
}
