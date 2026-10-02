/**
 * VinPipeline — model inference orchestration, environment-agnostic.
 *
 * The ONNX Runtime namespace is INJECTED (worker passes onnxruntime-web,
 * the benchmark passes the same package loaded under Node), so this exact
 * code runs in both. No DOM, no Worker APIs, no direct ORT import.
 *
 * Owns both sessions and all preallocated buffers:
 *  - detector input Float32Array(3·640·640), filled from a stretched frame
 *  - rec input Float32Array(3·48·576), filled by the fused homography warp
 * Input shapes are fixed so the ORT memory arena stabilizes after warmup.
 */

import { DET_INPUT_SIZE, fillChwFromRgba } from "./det-pre.js";
import { decodeDetections, pickBestDetection } from "./yolo-post.js";
import { warpQuadToStrip, expandQuad, REC_INPUT_HEIGHT, REC_INPUT_MAX_WIDTH } from "./homography.js";
import { createCtcDecoder } from "./ctc.js";

export const DEFAULT_PIPELINE_CONFIG = Object.freeze({
  scoreThreshold: 0.5,
  minAspect: 3,
  maxAspect: 30,
  minAreaFrac: 0.0005,
  /**
   * Quad growth before the rec warp, per side, relative to quad width/height.
   * The detector's OBB touches the first/last glyph, so a few px of jitter
   * drops a character. Internal validation set (476 strips, rec v6): 0 → 0.754,
   * 0.03/0.03 → 0.800; 0.02–0.05 along and 0–0.06 across are a plateau,
   * 0.12 across already hurts. NOTE: the old v5 rec collapses with any
   * expansion (0.41 → 0.24) — it was trained on tight crops.
   * Must stay ≤ BBOX_MARGIN (0.08) in scanner.worker.js so the crop covers it.
   */
  quadExpandAlong: 0.03,
  quadExpandAcross: 0.03,
  /**
   * ORT graph optimization at session create. Shipped models are pre-optimized
   * offline at the "extended" level (onnxruntime.transformers.optimizer,
   * same ORT version as onnxruntime-web), so "basic" skips the fusion pass on
   * every cold start with bit-identical outputs. Use "all" for a raw model
   * dropped in without that offline pass.
   */
  graphOptimizationLevel: "basic",
});

/**
 * @typedef {{ data: Uint8ClampedArray|Uint8Array, width: number, height: number }} RgbaImage
 */

/**
 * @param {{
 *   ort: typeof import("onnxruntime-web"),
 *   detectorModel: ArrayBuffer|Uint8Array,
 *   recModel: ArrayBuffer|Uint8Array,
 *   dictText: string,
 *   config?: Partial<typeof DEFAULT_PIPELINE_CONFIG>,
 * }} options
 */
export async function createVinPipeline(options) {
  const { ort, dictText } = options;
  const config = { ...DEFAULT_PIPELINE_CONFIG, ...(options.config ?? {}) };

  const sessionOptions = {
    executionProviders: ["wasm"],
    graphOptimizationLevel: config.graphOptimizationLevel,
  };

  let detectorModel = toUint8(options.detectorModel);
  let recModel = toUint8(options.recModel);

  const detSession = await ort.InferenceSession.create(detectorModel, sessionOptions);
  const recSession = await ort.InferenceSession.create(recModel, sessionOptions);
  // ORT copies model bytes into the wasm heap — drop the JS copies now.
  detectorModel = null;
  recModel = null;

  const detInputName = detSession.inputNames[0];
  const detOutputName = detSession.outputNames[0];
  const recInputName = recSession.inputNames[0];
  const recOutputName = recSession.outputNames[0];

  // Optional decoder factory override (e.g. the beam decoder from ctc-beam.js).
  const decoder = (options.createDecoder ?? createCtcDecoder)(dictText);

  const detBuffer = new Float32Array(3 * DET_INPUT_SIZE * DET_INPUT_SIZE);
  const recBuffer = new Float32Array(3 * REC_INPUT_HEIGHT * REC_INPUT_MAX_WIDTH);
  const detDims = [1, 3, DET_INPUT_SIZE, DET_INPUT_SIZE];
  const recDims = [1, 3, REC_INPUT_HEIGHT, REC_INPUT_MAX_WIDTH];

  // Warmup — first inference compiles kernels and sizes the arena.
  await runDet();
  await runRec();

  async function runDet() {
    const input = new ort.Tensor("float32", detBuffer, detDims);
    const outputs = await detSession.run({ [detInputName]: input });
    input.dispose?.();
    return outputs;
  }

  async function runRec() {
    const input = new ort.Tensor("float32", recBuffer, recDims);
    const outputs = await recSession.run({ [recInputName]: input });
    input.dispose?.();
    return outputs;
  }

  /**
   * Detect VIN regions on a stretched 640×640 RGBA frame.
   *
   * @param {RgbaImage} rgba640  Exactly 640×640.
   * @param {number} sourceWidth  Original frame width the quads map back to.
   * @param {number} sourceHeight
   * @returns {Promise<{ best: import("./yolo-post.js").Detection|null,
   *                     detections: import("./yolo-post.js").Detection[] }>}
   */
  async function detect(rgba640, sourceWidth, sourceHeight) {
    if (rgba640.width !== DET_INPUT_SIZE || rgba640.height !== DET_INPUT_SIZE) {
      throw new Error(`detect() expects a ${DET_INPUT_SIZE}×${DET_INPUT_SIZE} frame`);
    }
    fillChwFromRgba(rgba640.data, DET_INPUT_SIZE * DET_INPUT_SIZE, detBuffer);

    const outputs = await runDet();
    const output = outputs[detOutputName];
    const detections = decodeDetections(output.data, output.dims, {
      sourceWidth,
      sourceHeight,
      inputWidth: DET_INPUT_SIZE,
      inputHeight: DET_INPUT_SIZE,
      scoreThreshold: config.scoreThreshold,
    });
    disposeOutputs(outputs);

    const best = pickBestDetection(detections, {
      minAspect: config.minAspect,
      maxAspect: config.maxAspect,
      minAreaFrac: config.minAreaFrac,
      sourceWidth,
      sourceHeight,
    });
    return { best, detections };
  }

  /**
   * Recognize the text inside a quad of an RGBA image.
   *
   * @param {RgbaImage} image  Any resolution; quad is in its pixel coords.
   * @param {number[]} quad  8 floats.
   * @returns {Promise<import("./ctc.js").RecRead|null>} null when the quad is degenerate.
   */
  async function recognize(image, quad) {
    const grown = expandQuad(quad, config.quadExpandAlong, config.quadExpandAcross);
    const warped = warpQuadToStrip(image, grown, recBuffer);
    if (!warped) return null;

    const outputs = await runRec();
    const output = outputs[recOutputName];
    const [, timeSteps, numClasses] = output.dims;
    const read = decoder.decode(output.data, timeSteps, numClasses);
    disposeOutputs(outputs);
    return read;
  }

  function dispose() {
    // Sessions hold wasm-heap memory; release is best-effort — callers that
    // need a guaranteed reclaim terminate the hosting worker instead.
    detSession.release?.();
    recSession.release?.();
  }

  return {
    detect,
    recognize,
    dispose,
    config,
    meta: {
      detInputName,
      recInputName,
      detInputShape: detDims,
      recInputShape: recDims,
    },
  };
}

function toUint8(model) {
  return model instanceof Uint8Array ? model : new Uint8Array(model);
}

function disposeOutputs(outputs) {
  for (const tensor of Object.values(outputs)) {
    tensor.dispose?.();
  }
}
