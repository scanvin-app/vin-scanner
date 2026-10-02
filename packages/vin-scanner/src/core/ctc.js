/**
 * Greedy CTC decoder with per-character confidence — pure JS.
 *
 * Derived from PaddleOCR's CTCLabelDecode post-processing (Apache-2.0),
 * trimmed to the VIN use case:
 * no space char is appended (the rec model emits 34 classes = CTC blank
 * + the 33 VIN-legal dictionary characters) and each decoded character
 * carries its own confidence (max softmax probability within its run of
 * identical argmax timesteps).
 */

/**
 * @typedef {{ text: string, charConfs: number[], meanConf: number,
 *             alts: Array<{char: string, conf: number}|null> }} RecRead
 *
 * `alts[i]` is the runner-up class (with its softmax probability) at the
 * strongest timestep of character i's run — the OCR's own second guess,
 * consumed by the n-gram rescorer. null when the runner-up was the blank.
 */

/**
 * @param {string} dictText  Dictionary file contents, one character per line.
 * @returns {{ decode: (preds: Float32Array, timeSteps: number, numClasses: number) => RecRead,
 *             numClasses: number }}
 */
export function createCtcDecoder(dictText) {
  const chars = String(dictText)
    .split("\n")
    .map((line) => line.replace(/\r$/, ""))
    .filter((line) => line.length > 0);

  // Index 0 is the CTC blank token.
  const characters = ["", ...chars];
  const numClasses = characters.length;

  /**
   * @param {Float32Array} preds  Flat [1, T, C] tensor data (logits or probs).
   * @param {number} timeSteps
   * @param {number} classes  C reported by the tensor — may exceed the dict
   *   (extra classes are ignored beyond argmax scope of known indices).
   */
  function decode(preds, timeSteps, classes) {
    const needSoftmax = !isProbabilityRow(preds, classes);

    const textChars = [];
    const charConfs = [];
    const alts = [];
    let prevIdx = -1;
    let runConf = 0;

    for (let t = 0; t < timeSteps; t += 1) {
      const rowOffset = t * classes;

      // Top-2 classes at this timestep (the runner-up feeds the rescorer).
      let bestIdx = 0;
      let secondIdx = -1;
      let bestProb;
      let secondProb = 0;
      if (needSoftmax) {
        let maxVal = -Infinity;
        let secondVal = -Infinity;
        for (let c = 0; c < classes; c += 1) {
          const v = preds[rowOffset + c];
          if (v > maxVal) {
            secondVal = maxVal;
            secondIdx = bestIdx;
            maxVal = v;
            bestIdx = c;
          } else if (v > secondVal) {
            secondVal = v;
            secondIdx = c;
          }
        }
        let sumExp = 0;
        for (let c = 0; c < classes; c += 1) {
          sumExp += Math.exp(preds[rowOffset + c] - maxVal);
        }
        bestProb = 1 / sumExp; // exp(max - max) / sumExp
        secondProb = secondIdx >= 0 ? Math.exp(secondVal - maxVal) / sumExp : 0;
      } else {
        bestProb = preds[rowOffset];
        for (let c = 1; c < classes; c += 1) {
          const v = preds[rowOffset + c];
          if (v > bestProb) {
            secondProb = bestProb;
            secondIdx = bestIdx;
            bestProb = v;
            bestIdx = c;
          } else if (v > secondProb) {
            secondProb = v;
            secondIdx = c;
          }
        }
      }

      const alt =
        secondIdx > 0 && secondIdx < characters.length
          ? { char: characters[secondIdx], conf: secondProb }
          : null;

      if (bestIdx === 0) {
        prevIdx = 0;
        continue;
      }
      if (bestIdx === prevIdx) {
        // Same character run — keep the strongest timestep as its confidence.
        if (bestProb > runConf && charConfs.length > 0) {
          runConf = bestProb;
          charConfs[charConfs.length - 1] = bestProb;
          alts[alts.length - 1] = alt;
        }
        continue;
      }
      prevIdx = bestIdx;
      runConf = bestProb;

      if (bestIdx < characters.length) {
        textChars.push(characters[bestIdx]);
        charConfs.push(bestProb);
        alts.push(alt);
      }
    }

    const text = textChars.join("");
    const meanConf =
      charConfs.length > 0 ? charConfs.reduce((a, b) => a + b, 0) / charConfs.length : 0;

    return { text, charConfs, meanConf, alts };
  }

  return { decode, numClasses };
}

function isProbabilityRow(data, numClasses) {
  if (data.length < numClasses) return false;
  let sum = 0;
  for (let c = 0; c < numClasses; c += 1) sum += data[c];
  return Math.abs(sum - 1.0) < 0.01;
}
