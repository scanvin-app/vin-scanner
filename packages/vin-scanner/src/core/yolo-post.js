/**
 * VIN detector postprocessing — pure JS, no DOM/Worker/ORT imports.
 *
 * vin_detector.onnx is an end-to-end OBB export: output "output0" has shape
 * [batch, 300, 7] with NMS baked into the graph. Attributes per prediction:
 * cx, cy, w, h, score, class, angle(rad). Coordinates are in detector input
 * space (640×640) or normalized [0,1] — probed at runtime like the original
 * scanner did.
 */

const ATTR_COUNT = 7;

/**
 * @typedef {{ score: number, quad: number[], bbox: number[] }} Detection
 *   quad — 8 floats [x0,y0,...,x3,y3] in source-frame pixels;
 *   bbox — [x, y, width, height] axis-aligned clamp of the quad.
 */

/**
 * Decode raw detector output into detections in source-frame coordinates.
 *
 * @param {Float32Array} data  Flat tensor data.
 * @param {readonly number[]} dims  Tensor dims, expected [1, N, 7].
 * @param {{ sourceWidth: number, sourceHeight: number,
 *           inputWidth?: number, inputHeight?: number,
 *           scoreThreshold?: number }} options
 * @returns {Detection[]} sorted by score, descending.
 */
export function decodeDetections(data, dims, options) {
  const {
    sourceWidth,
    sourceHeight,
    inputWidth = 640,
    inputHeight = 640,
    scoreThreshold = 0.5,
  } = options;

  if (!Array.isArray(dims) || dims.length !== 3 || dims[2] !== ATTR_COUNT) {
    throw new Error(
      `Unexpected detector output shape [${dims?.join(", ")}] — expected [1, N, ${ATTR_COUNT}]`
    );
  }

  const numPredictions = dims[1];
  const coordScale = resolveCoordinateScale(data, numPredictions, inputWidth, inputHeight);
  const xRatio = sourceWidth / inputWidth;
  const yRatio = sourceHeight / inputHeight;

  /** @type {Detection[]} */
  const detections = [];

  for (let p = 0; p < numPredictions; p += 1) {
    const base = p * ATTR_COUNT;
    const score = data[base + 4];
    if (!Number.isFinite(score) || score < scoreThreshold) continue;

    const cx = data[base] * coordScale.x;
    const cy = data[base + 1] * coordScale.y;
    const w = Math.abs(data[base + 2] * coordScale.x);
    const h = Math.abs(data[base + 3] * coordScale.y);
    const angle = normalizeAngleRadians(data[base + 6]);
    if (
      !Number.isFinite(cx) ||
      !Number.isFinite(cy) ||
      !Number.isFinite(angle) ||
      w <= 0 ||
      h <= 0
    ) {
      continue;
    }

    const quad = rotatedRectToQuad(cx, cy, w, h, angle);

    // Map from detector input space to source-frame pixels and clamp.
    const scaled = new Array(8);
    let valid = true;
    for (let i = 0; i < 8; i += 2) {
      const sx = clamp(quad[i] * xRatio, 0, sourceWidth);
      const sy = clamp(quad[i + 1] * yRatio, 0, sourceHeight);
      if (!Number.isFinite(sx) || !Number.isFinite(sy)) {
        valid = false;
        break;
      }
      scaled[i] = sx;
      scaled[i + 1] = sy;
    }
    if (!valid) continue;

    const bbox = quadToBbox(scaled, sourceWidth, sourceHeight);
    if (!bbox) continue;

    detections.push({ score, quad: scaled, bbox });
  }

  detections.sort((a, b) => b.score - a.score);
  return detections;
}

/**
 * Pick the best detection subject to plausibility gates for a VIN strip.
 *
 * @param {Detection[]} detections  Sorted by score, descending.
 * @param {{ minAspect?: number, maxAspect?: number, minAreaFrac?: number,
 *           sourceWidth?: number, sourceHeight?: number }} [gates]
 * @returns {Detection|null}
 */
export function pickBestDetection(detections, gates = {}) {
  const { minAspect = 3, maxAspect = 30, minAreaFrac = 0, sourceWidth = 0, sourceHeight = 0 } = gates;
  const frameArea = sourceWidth * sourceHeight;

  for (const det of detections) {
    const { width, height } = quadDimensions(det.quad);
    if (height <= 0) continue;
    const aspect = width / height;
    if (aspect < minAspect || aspect > maxAspect) continue;
    if (frameArea > 0 && minAreaFrac > 0 && (width * height) / frameArea < minAreaFrac) continue;
    return det;
  }
  return null;
}

/**
 * Average edge lengths of a quad — width along p0→p1/p3→p2, height p0→p3/p1→p2.
 * @param {number[]} quad
 * @returns {{ width: number, height: number }}
 */
export function quadDimensions(quad) {
  const width =
    (Math.hypot(quad[2] - quad[0], quad[3] - quad[1]) +
      Math.hypot(quad[4] - quad[6], quad[5] - quad[7])) /
    2;
  const height =
    (Math.hypot(quad[6] - quad[0], quad[7] - quad[1]) +
      Math.hypot(quad[4] - quad[2], quad[5] - quad[3])) /
    2;
  return { width, height };
}

function resolveCoordinateScale(data, numPredictions, inputWidth, inputHeight) {
  const sampleCount = Math.min(numPredictions, 24);
  let maxAbs = 0;
  for (let p = 0; p < sampleCount; p += 1) {
    const base = p * ATTR_COUNT;
    for (let a = 0; a < 4; a += 1) {
      const v = data[base + a];
      if (Number.isFinite(v)) maxAbs = Math.max(maxAbs, Math.abs(v));
    }
  }
  // Coordinates ≤ 2.5 across the sample ⇒ normalized [0,1] output.
  if (maxAbs > 0 && maxAbs <= 2.5) {
    return { x: inputWidth, y: inputHeight };
  }
  return { x: 1, y: 1 };
}

function normalizeAngleRadians(rawAngle) {
  const angle = Number(rawAngle);
  if (!Number.isFinite(angle)) return NaN;
  // Some exports emit degrees; radians never exceed 2π in magnitude.
  if (Math.abs(angle) > Math.PI * 2 && Math.abs(angle) <= 360) {
    return (angle * Math.PI) / 180;
  }
  return angle;
}

function rotatedRectToQuad(cx, cy, width, height, angle) {
  const halfW = width / 2;
  const halfH = height / 2;
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  return [
    cx + -halfW * cos - -halfH * sin,
    cy + -halfW * sin + -halfH * cos,
    cx + halfW * cos - -halfH * sin,
    cy + halfW * sin + -halfH * cos,
    cx + halfW * cos - halfH * sin,
    cy + halfW * sin + halfH * cos,
    cx + -halfW * cos - halfH * sin,
    cy + -halfW * sin + halfH * cos,
  ];
}

function quadToBbox(quad, frameWidth, frameHeight) {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (let i = 0; i < 8; i += 2) {
    minX = Math.min(minX, quad[i]);
    maxX = Math.max(maxX, quad[i]);
    minY = Math.min(minY, quad[i + 1]);
    maxY = Math.max(maxY, quad[i + 1]);
  }
  const left = Math.max(0, minX);
  const top = Math.max(0, minY);
  const right = Math.min(frameWidth, maxX);
  const bottom = Math.min(frameHeight, maxY);
  if (right - left <= 1 || bottom - top <= 1) return null;
  return [left, top, right - left, bottom - top];
}

function clamp(value, min, max) {
  if (!Number.isFinite(value)) return NaN;
  return Math.max(min, Math.min(max, value));
}
