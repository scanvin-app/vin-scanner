/**
 * Detector preprocessing — pure JS, no DOM/Worker/ORT imports.
 *
 * The detector consumes a 640×640 RGB CHW tensor normalized to [0, 1],
 * produced by plainly stretching the source frame (no letterbox — parity
 * with the original scanner whose model was validated with stretch resize).
 */

export const DET_INPUT_SIZE = 640;

/**
 * Fill a CHW float buffer from RGBA pixel data (x/255 normalization).
 *
 * @param {Uint8ClampedArray|Uint8Array} rgba  Pixels, length = pixelCount*4.
 * @param {number} pixelCount  width*height of the image.
 * @param {Float32Array} out  Preallocated buffer, length = pixelCount*3.
 */
export function fillChwFromRgba(rgba, pixelCount, out) {
  const gOffset = pixelCount;
  const bOffset = pixelCount * 2;
  for (let p = 0, i = 0; p < pixelCount; p += 1, i += 4) {
    out[p] = rgba[i] / 255;
    out[gOffset + p] = rgba[i + 1] / 255;
    out[bOffset + p] = rgba[i + 2] / 255;
  }
}

/**
 * Bilinear RGBA resize in pure JS. Used by the benchmark harness so frames
 * follow the same code path as production apart from the resample kernel
 * (the worker uses OffscreenCanvas drawImage instead).
 *
 * @param {Uint8ClampedArray|Uint8Array} src
 * @param {number} srcW
 * @param {number} srcH
 * @param {Uint8ClampedArray|Uint8Array} dst  Preallocated, dstW*dstH*4.
 * @param {number} dstW
 * @param {number} dstH
 */
export function bilinearResizeRgba(src, srcW, srcH, dst, dstW, dstH) {
  const xRatio = srcW / dstW;
  const yRatio = srcH / dstH;

  for (let y = 0; y < dstH; y += 1) {
    const srcY = (y + 0.5) * yRatio - 0.5;
    const y0 = Math.max(0, Math.floor(srcY));
    const y1 = Math.min(srcH - 1, y0 + 1);
    const fy = Math.min(1, Math.max(0, srcY - y0));

    for (let x = 0; x < dstW; x += 1) {
      const srcX = (x + 0.5) * xRatio - 0.5;
      const x0 = Math.max(0, Math.floor(srcX));
      const x1 = Math.min(srcW - 1, x0 + 1);
      const fx = Math.min(1, Math.max(0, srcX - x0));

      const i00 = (y0 * srcW + x0) * 4;
      const i10 = (y0 * srcW + x1) * 4;
      const i01 = (y1 * srcW + x0) * 4;
      const i11 = (y1 * srcW + x1) * 4;
      const di = (y * dstW + x) * 4;

      for (let c = 0; c < 4; c += 1) {
        const top = src[i00 + c] + (src[i10 + c] - src[i00 + c]) * fx;
        const bottom = src[i01 + c] + (src[i11 + c] - src[i01 + c]) * fx;
        dst[di + c] = top + (bottom - top) * fy;
      }
    }
  }
}
