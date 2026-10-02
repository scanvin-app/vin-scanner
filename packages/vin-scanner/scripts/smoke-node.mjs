/**
 * Smoke test: run the full production pipeline (detect → warp → rec) on a
 * still image in Node, through exactly the same core code the browser uses.
 *
 *   node scripts/smoke-node.mjs <image-path>
 *
 * Generate a synthetic input with scripts/synthetic-vin-image.sh (repo root).
 */

import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

import { ort } from "./ort-env.mjs";
import { decodeImage } from "./frames.mjs";
import { createVinPipeline } from "../src/core/pipeline.js";
import { bilinearResizeRgba, DET_INPUT_SIZE } from "../src/core/det-pre.js";
import { isVinShape, checkDigitOk, wmiKnown } from "../src/core/vin.js";

const here = (p) => fileURLToPath(new URL(p, import.meta.url));

const imagePath = process.argv[2];
if (!imagePath) {
  console.error("Usage: node scripts/smoke-node.mjs <image-path>");
  process.exit(2);
}

console.log("Loading models…");
const [detectorModel, recModel, dictText] = await Promise.all([
  readFile(here("../models/vin_detector.onnx")),
  readFile(here("../models/ppocr-rec.onnx")),
  readFile(here("../models/vin_dict.txt"), "utf8"),
]);

const t0 = performance.now();
const pipeline = await createVinPipeline({ ort, detectorModel, recModel, dictText });
console.log(`Pipeline ready in ${(performance.now() - t0).toFixed(0)} ms`);
console.log("meta:", JSON.stringify(pipeline.meta));

const image = await decodeImage(imagePath);
console.log(`Image: ${imagePath} (${image.width}×${image.height})`);

// Stretch-resize to detector input (pure-JS path, same as the bench harness).
let det640 = image;
if (image.width !== DET_INPUT_SIZE || image.height !== DET_INPUT_SIZE) {
  const resized = new Uint8ClampedArray(DET_INPUT_SIZE * DET_INPUT_SIZE * 4);
  bilinearResizeRgba(image.data, image.width, image.height, resized, DET_INPUT_SIZE, DET_INPUT_SIZE);
  det640 = { data: resized, width: DET_INPUT_SIZE, height: DET_INPUT_SIZE };
}

const t1 = performance.now();
const { best, detections } = await pipeline.detect(det640, image.width, image.height);
console.log(`Detection: ${(performance.now() - t1).toFixed(0)} ms, ${detections.length} detections`);
for (const det of detections.slice(0, 3)) {
  console.log(
    `  score=${det.score.toFixed(3)} quad=[${det.quad.map((v) => v.toFixed(1)).join(", ")}]`
  );
}

if (!best) {
  console.error("No detection passed the gates — smoke test FAILED");
  process.exit(1);
}

const t2 = performance.now();
const read = await pipeline.recognize(image, best.quad);
console.log(`Recognition: ${(performance.now() - t2).toFixed(0)} ms`);

if (!read) {
  console.error("Recognizer returned null — smoke test FAILED");
  process.exit(1);
}

console.log(`Text: "${read.text}" (${read.text.length} chars)`);
console.log(`meanConf: ${read.meanConf.toFixed(3)}`);
console.log(
  `charConfs: [${read.charConfs.map((c) => c.toFixed(2)).join(", ")}]`
);
console.log(
  `isVinShape: ${isVinShape(read.text)}, checkDigitOk: ${checkDigitOk(read.text)}, wmiKnown: ${wmiKnown(read.text)}`
);

pipeline.dispose();

if (!isVinShape(read.text)) {
  console.error("Read is not a valid VIN shape — smoke test FAILED");
  process.exit(1);
}
console.log("Smoke test PASSED");
