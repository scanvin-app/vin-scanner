/**
 * Loads onnxruntime-web's wasm backend under Node for the benchmark —
 * the same package and binary the browser worker runs, so inference is
 * bit-exact with production.
 */

import * as ort from "onnxruntime-web/wasm";

ort.env.wasm.numThreads = 1;
ort.env.wasm.proxy = false;

export { ort };
