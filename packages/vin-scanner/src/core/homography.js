/**
 * 4-point perspective warp — pure JS, replaces OpenCV warpPerspective.
 *
 * Computes the homography mapping the rec-input strip rectangle onto the
 * detected quad and samples the source image with inverse mapping + bilinear
 * interpolation, writing normalized CHW floats straight into the preallocated
 * rec input buffer (normalization x/127.5−1 fused into the sampling loop,
 * zero padding to the right — PaddleOCR semantics).
 */

export const REC_INPUT_HEIGHT = 48;
export const REC_INPUT_MAX_WIDTH = 576;

/**
 * Solve the 8-DOF homography H such that H · (srcPts[i]) ≈ dstPts[i]
 * for the four correspondences. Returns row-major [h0..h8] with h8 = 1,
 * or null for degenerate input.
 *
 * @param {number[]} srcPts  8 floats [x0,y0,...,x3,y3]
 * @param {number[]} dstPts  8 floats
 * @returns {number[]|null}
 */
export function computeHomography(srcPts, dstPts) {
  // Build the 8×9 system A·h = b (h8 fixed to 1).
  const a = [];
  const b = [];
  for (let i = 0; i < 4; i += 1) {
    const sx = srcPts[i * 2];
    const sy = srcPts[i * 2 + 1];
    const dx = dstPts[i * 2];
    const dy = dstPts[i * 2 + 1];
    a.push([sx, sy, 1, 0, 0, 0, -sx * dx, -sy * dx]);
    b.push(dx);
    a.push([0, 0, 0, sx, sy, 1, -sx * dy, -sy * dy]);
    b.push(dy);
  }

  const h = solveLinearSystem(a, b);
  if (!h) return null;
  return [...h, 1];
}

/**
 * Order quad corners as [TL, TR, BR, BL] relative to the text direction:
 * the first edge (p0→p1) is the longer dimension and points left→right.
 * Detector quads come from rotated rects, so corner order is consistent —
 * we only normalize the starting corner and direction.
 *
 * @param {number[]} quad  8 floats.
 * @returns {number[]} reordered copy.
 */
export function orderQuad(quad) {
  let q = quad.slice();

  const edge01 = Math.hypot(q[2] - q[0], q[3] - q[1]);
  const edge12 = Math.hypot(q[4] - q[2], q[5] - q[3]);
  // Rotate corner order by one so the first edge is the long (text) axis.
  if (edge12 > edge01) {
    q = [q[2], q[3], q[4], q[5], q[6], q[7], q[0], q[1]];
  }
  // Flip 180° so the text axis points left→right.
  if (q[2] - q[0] < 0) {
    q = [q[4], q[5], q[6], q[7], q[0], q[1], q[2], q[3]];
  }
  return q;
}

/**
 * Grow a quad outward along its text axis and across it. The detector's OBB
 * hugs the glyphs with ~0 px horizontal margin, so any localization jitter
 * clips half of the first/last character; the recognizer was trained with
 * 5–15 % margins and tolerates the extra context. Fractions are per side,
 * relative to the quad's own width/height (so the expansion is
 * perspective-consistent — applied in image space along the quad edges).
 *
 * @param {number[]} quad  8 floats, any corner order.
 * @param {number} alongFrac  Fraction of quad width added on each end (0 = off).
 * @param {number} acrossFrac  Fraction of quad height added top and bottom.
 * @returns {number[]} new quad ordered [TL, TR, BR, BL].
 */
export function expandQuad(quad, alongFrac = 0, acrossFrac = 0) {
  const q = orderQuad(quad);
  if (!(alongFrac > 0) && !(acrossFrac > 0)) return q;
  const out = q.slice();
  // Each edge is pushed outward independently so skewed quads keep their shape:
  // top/bottom edges extend along their own direction, left/right edges along theirs.
  const extend = (i0, i1, frac) => {
    // Move corner i0 away from i1 and corner i1 away from i0 by frac × |edge|.
    const dx = q[i1] - q[i0];
    const dy = q[i1 + 1] - q[i0 + 1];
    out[i0] -= dx * frac;
    out[i0 + 1] -= dy * frac;
    out[i1] += dx * frac;
    out[i1 + 1] += dy * frac;
  };
  if (alongFrac > 0) {
    extend(0, 2, alongFrac); // TL ← → TR
    extend(6, 4, alongFrac); // BL ← → BR
  }
  if (acrossFrac > 0) {
    extend(0, 6, acrossFrac); // TL ↑ ↓ BL
    extend(2, 4, acrossFrac); // TR ↑ ↓ BR
  }
  return out;
}

/**
 * Warp the quad region of an RGBA image into a rec-model input buffer.
 *
 * @param {{ data: Uint8ClampedArray|Uint8Array, width: number, height: number }} image
 * @param {number[]} quad  8 floats in image pixel coordinates (any corner order).
 * @param {Float32Array} out  Preallocated CHW buffer, 3 * outH * maxW floats.
 * @param {{ outH?: number, maxW?: number }} [options]
 * @returns {{ width: number }|null}  Number of written columns, or null when degenerate.
 */
export function warpQuadToStrip(image, quad, out, options = {}) {
  const { outH = REC_INPUT_HEIGHT, maxW = REC_INPUT_MAX_WIDTH } = options;
  const ordered = orderQuad(quad);

  const quadW =
    (Math.hypot(ordered[2] - ordered[0], ordered[3] - ordered[1]) +
      Math.hypot(ordered[4] - ordered[6], ordered[5] - ordered[7])) /
    2;
  const quadH =
    (Math.hypot(ordered[6] - ordered[0], ordered[7] - ordered[1]) +
      Math.hypot(ordered[4] - ordered[2], ordered[5] - ordered[3])) /
    2;
  if (!(quadW > 1) || !(quadH > 1)) return null;

  const targetW = Math.max(16, Math.min(maxW, Math.round((outH * quadW) / quadH)));

  const h = computeHomography(
    [0, 0, targetW, 0, targetW, outH, 0, outH],
    ordered
  );
  if (!h) return null;

  out.fill(0);

  const { data, width: imgW, height: imgH } = image;
  const plane = outH * maxW;

  for (let y = 0; y < outH; y += 1) {
    const dy = y + 0.5;
    for (let x = 0; x < targetW; x += 1) {
      const dx = x + 0.5;
      const denom = h[6] * dx + h[7] * dy + h[8];
      const sx = (h[0] * dx + h[1] * dy + h[2]) / denom - 0.5;
      const sy = (h[3] * dx + h[4] * dy + h[5]) / denom - 0.5;

      const x0 = Math.floor(sx);
      const y0 = Math.floor(sy);
      const fx = sx - x0;
      const fy = sy - y0;
      const cx0 = Math.max(0, Math.min(imgW - 1, x0));
      const cx1 = Math.max(0, Math.min(imgW - 1, x0 + 1));
      const cy0 = Math.max(0, Math.min(imgH - 1, y0));
      const cy1 = Math.max(0, Math.min(imgH - 1, y0 + 1));

      const i00 = (cy0 * imgW + cx0) * 4;
      const i10 = (cy0 * imgW + cx1) * 4;
      const i01 = (cy1 * imgW + cx0) * 4;
      const i11 = (cy1 * imgW + cx1) * 4;
      const di = y * maxW + x;

      for (let c = 0; c < 3; c += 1) {
        const top = data[i00 + c] + (data[i10 + c] - data[i00 + c]) * fx;
        const bottom = data[i01 + c] + (data[i11 + c] - data[i01 + c]) * fx;
        out[c * plane + di] = (top + (bottom - top) * fy) / 127.5 - 1;
      }
    }
  }

  return { width: targetW };
}

/**
 * Gaussian elimination with partial pivoting for an n×n system.
 * @param {number[][]} a  n rows of n coefficients (mutated).
 * @param {number[]} b  n values (mutated).
 * @returns {number[]|null}
 */
function solveLinearSystem(a, b) {
  const n = a.length;
  for (let col = 0; col < n; col += 1) {
    let pivot = col;
    for (let row = col + 1; row < n; row += 1) {
      if (Math.abs(a[row][col]) > Math.abs(a[pivot][col])) pivot = row;
    }
    if (Math.abs(a[pivot][col]) < 1e-10) return null;
    if (pivot !== col) {
      [a[col], a[pivot]] = [a[pivot], a[col]];
      [b[col], b[pivot]] = [b[pivot], b[col]];
    }
    for (let row = col + 1; row < n; row += 1) {
      const factor = a[row][col] / a[col][col];
      for (let k = col; k < n; k += 1) a[row][k] -= factor * a[col][k];
      b[row] -= factor * b[col];
    }
  }
  const x = new Array(n);
  for (let row = n - 1; row >= 0; row -= 1) {
    let sum = b[row];
    for (let k = row + 1; k < n; k += 1) sum -= a[row][k] * x[k];
    x[row] = sum / a[row][row];
  }
  return x;
}
