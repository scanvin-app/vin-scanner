/**
 * Scanner worker — the ONLY wasm host in the library. One onnxruntime-web
 * instance runs both sessions (detector + recognizer); the temporal
 * consensus also lives here so a frame message returns a complete verdict.
 *
 * Memory rules: two reusable OffscreenCanvases (detector stretch + crop),
 * all tensor buffers preallocated inside the pipeline, every received
 * ImageBitmap closed in finally. The main thread guarantees at most one
 * frame in flight. Hard teardown = worker.terminate() from the main thread.
 */

import * as ort from "onnxruntime-web/wasm";

import { MSG } from "./protocol.js";
import { fetchModel } from "./model-cache.js";
import { createVinPipeline } from "../core/pipeline.js";
import { createConsensusTracker } from "../core/consensus.js";
import { createNgramScorer } from "../core/ngram.js";
import { createBeamDecoder } from "../core/ctc-beam.js";
import { checkDigitOk } from "../core/vin.js";
import { quadDimensions } from "../core/yolo-post.js";
import { DET_INPUT_SIZE } from "../core/det-pre.js";

const CROP_MAX_W = 1280;
const CROP_MAX_H = 384;
/** Crop is downscaled so the quad is ~2× the rec input height (anti-alias). */
const TARGET_QUAD_HEIGHT = 96;
const BBOX_MARGIN = 0.08;
/** Manual guide-box fallback: strip height and vertical centers. */
const STRIP_HEIGHT_FRAC = 0.45;
const STRIP_CENTERS = [0.3, 0.5, 0.7];

let pipeline = null;
let tracker = null;
let ngram = null;

let detCanvas = null;
let detCtx = null;
let cropCanvas = null;
let cropCtx = null;

self.onmessage = async (event) => {
  const msg = event.data;
  try {
    if (msg.type === MSG.INIT) {
      await init(msg);
    } else if (msg.type === MSG.FRAME) {
      await handleFrame(msg);
    } else if (msg.type === MSG.RESET) {
      tracker?.reset();
    }
  } catch (error) {
    const type = pipeline ? MSG.ERROR : MSG.INIT_ERROR;
    self.postMessage({ type, message: error?.message ?? String(error) });
  }
};

async function init({ detectorUrl, recUrl, dictUrl, wasmUrl, ngramUrl = null, config = {} }) {
  ort.env.wasm.wasmPaths = { wasm: wasmUrl };
  // No COOP/COEP on the host page ⇒ no SharedArrayBuffer; pin single-thread
  // so ORT never probes for or spawns proxy workers.
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.proxy = false;

  const progress = { detector: 0, rec: 0, dict: 0 };
  // Sizes known at build time would be nicer; report bytes loaded instead.
  const report = () => {
    self.postMessage({
      type: MSG.PROGRESS,
      loaded: progress.detector + progress.rec + progress.dict,
      total: null,
    });
  };

  const [detectorModel, recModel, dictBuffer, ngramBuffer] = await Promise.all([
    fetchModel(detectorUrl, (loaded) => {
      progress.detector = loaded;
      report();
    }),
    fetchModel(recUrl, (loaded) => {
      progress.rec = loaded;
      report();
    }),
    fetchModel(dictUrl),
    // The n-gram model is an accuracy booster, never a gate — a missing or
    // broken file must not take the scanner down.
    ngramUrl ? fetchModel(ngramUrl).catch(() => null) : Promise.resolve(null),
  ]);
  const dictText = new TextDecoder().decode(dictBuffer);

  if (ngramBuffer) {
    try {
      const ngramModel = JSON.parse(new TextDecoder().decode(ngramBuffer));
      ngram = createNgramScorer(ngramModel, checkDigitOk, config.ngram);
    } catch {
      ngram = null;
    }
  }

  pipeline = await createVinPipeline({
    ort,
    detectorModel,
    recModel,
    dictText,
    config: config.pipeline,
    // With the n-gram available, decode with beam search + shallow fusion
    // (bench: 73% vs 41% exact per-frame reads, only variant that still
    // accepts clips at 1 fps). Without it, fall back to the greedy decoder.
    createDecoder: ngram ? (dict) => createBeamDecoder(dict, ngram) : undefined,
  });

  // Fused beams homogenize reads: a systematic misread repeats VERBATIM
  // frame after frame, so exact-repetition tiers must demand one more
  // independent read than with the noisier greedy decoder (bench: tier B at
  // 3 accepted a wrong serial digit; at 4 every run was clean).
  const consensusConfig = { ...(config.consensus ?? {}) };
  if (ngram) {
    consensusConfig.tierBCount = Math.max(4, consensusConfig.tierBCount ?? 0);
  }
  tracker = createConsensusTracker(consensusConfig);

  detCanvas = new OffscreenCanvas(DET_INPUT_SIZE, DET_INPUT_SIZE);
  detCtx = detCanvas.getContext("2d", { willReadFrequently: true });
  cropCanvas = new OffscreenCanvas(CROP_MAX_W, CROP_MAX_H);
  cropCtx = cropCanvas.getContext("2d", { willReadFrequently: true });

  // decoder: production observability — tells the host whether the n-gram
  // loaded and beam decoding is live, or the worker fell back to greedy.
  self.postMessage({ type: MSG.READY, decoder: ngram ? "beam" : "greedy" });
}

async function handleFrame({ bitmap, ts, manual = false, guideRect = null }) {
  try {
    if (!pipeline) return;

    const sourceWidth = bitmap.width;
    const sourceHeight = bitmap.height;

    detCtx.drawImage(bitmap, 0, 0, DET_INPUT_SIZE, DET_INPUT_SIZE);
    const rgba640 = detCtx.getImageData(0, 0, DET_INPUT_SIZE, DET_INPUT_SIZE);
    const { best } = await pipeline.detect(rgba640, sourceWidth, sourceHeight);

    let read = null;
    let accepted = null;

    if (best) {
      read = await recognizeRegion(bitmap, best.quad);
      if (read?.text) {
        if (ngram) read = ngram.rescoreRead(read);
        accepted = tracker.add(read, ts);
      }
    } else if (manual && guideRect) {
      const strips = await recognizeGuideStrips(bitmap, guideRect);
      for (let stripRead of strips) {
        if (ngram) stripRead = ngram.rescoreRead(stripRead);
        accepted = tracker.add(stripRead, ts) ?? accepted;
        if (!read || stripRead.meanConf > read.meanConf) read = stripRead;
      }
    }

    self.postMessage({
      type: MSG.RESULT,
      ts,
      manual,
      sourceWidth,
      sourceHeight,
      detection: best ? { quad: best.quad, score: best.score } : null,
      read: read ? { text: read.text, meanConf: read.meanConf } : null,
      accepted,
    });
  } finally {
    bitmap.close();
  }
}

/**
 * Downscale the quad's bounding box (plus margin) into the reusable crop
 * canvas, then run the recognizer on the warped quad.
 */
async function recognizeRegion(bitmap, quad) {
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
  const marginX = (maxX - minX) * BBOX_MARGIN;
  const marginY = (maxY - minY) * BBOX_MARGIN;
  const sx = Math.max(0, Math.floor(minX - marginX));
  const sy = Math.max(0, Math.floor(minY - marginY));
  const sw = Math.min(bitmap.width, Math.ceil(maxX + marginX)) - sx;
  const sh = Math.min(bitmap.height, Math.ceil(maxY + marginY)) - sy;
  if (sw < 8 || sh < 4) return null;

  const { height: quadH } = quadDimensions(quad);
  const scale = Math.min(
    1,
    quadH > 0 ? TARGET_QUAD_HEIGHT / quadH : 1,
    CROP_MAX_W / sw,
    CROP_MAX_H / sh
  );
  const dw = Math.max(1, Math.round(sw * scale));
  const dh = Math.max(1, Math.round(sh * scale));

  cropCtx.clearRect(0, 0, dw, dh);
  cropCtx.drawImage(bitmap, sx, sy, sw, sh, 0, 0, dw, dh);
  const cropImage = cropCtx.getImageData(0, 0, dw, dh);

  const quadInCrop = new Array(8);
  for (let i = 0; i < 8; i += 2) {
    quadInCrop[i] = (quad[i] - sx) * (dw / sw);
    quadInCrop[i + 1] = (quad[i + 1] - sy) * (dh / sh);
  }

  return pipeline.recognize(cropImage, quadInCrop);
}

/**
 * Manual shutter fallback without a detection: recognize three overlapping
 * horizontal strips of the guide-box region (axis-aligned quads through the
 * same warp code).
 */
async function recognizeGuideStrips(bitmap, guideRect) {
  const sx = Math.max(0, Math.floor(guideRect.x));
  const sy = Math.max(0, Math.floor(guideRect.y));
  const sw = Math.min(bitmap.width, Math.ceil(guideRect.x + guideRect.width)) - sx;
  const sh = Math.min(bitmap.height, Math.ceil(guideRect.y + guideRect.height)) - sy;
  if (sw < 32 || sh < 8) return [];

  const stripH = sh * STRIP_HEIGHT_FRAC;
  const scale = Math.min(1, TARGET_QUAD_HEIGHT / stripH, CROP_MAX_W / sw, CROP_MAX_H / sh);
  const dw = Math.max(1, Math.round(sw * scale));
  const dh = Math.max(1, Math.round(sh * scale));

  cropCtx.clearRect(0, 0, dw, dh);
  cropCtx.drawImage(bitmap, sx, sy, sw, sh, 0, 0, dw, dh);
  const cropImage = cropCtx.getImageData(0, 0, dw, dh);

  const reads = [];
  for (const centerFrac of STRIP_CENTERS) {
    const top = Math.max(0, (centerFrac - STRIP_HEIGHT_FRAC / 2) * dh);
    const bottom = Math.min(dh, (centerFrac + STRIP_HEIGHT_FRAC / 2) * dh);
    const quad = [0, top, dw, top, dw, bottom, 0, bottom];
    const read = await pipeline.recognize(cropImage, quad);
    if (read?.text) reads.push(read);
  }
  return reads;
}
