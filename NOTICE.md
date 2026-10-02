# Notices and attributions

vin-scanner — Copyright (c) 2026 scanvin.app

This product is licensed under the GNU Affero General Public License v3.0 (see LICENSE).
Please keep this notice and the "Powered by ScanVin (https://scanvin.app)" attribution in
works that include vin-scanner, as permitted by AGPL-3.0 section 7(b).

## Bundled models

### `models/vin_detector.onnx`

VIN detector based on the YOLO26n-OBB architecture, trained and exported with
[Ultralytics YOLO](https://github.com/ultralytics/ultralytics) on a dataset assembled by
scanvin.app, then quantized (INT8, dynamic) and graph-optimized with ONNX Runtime.
Ultralytics is licensed under the GNU Affero General Public License v3.0
(https://ultralytics.com/license); the model metadata carries that license.

### `models/ppocr-rec.onnx`

Text recognizer fine-tuned on VIN strips by scanvin.app, starting from the PP-OCRv6 tiny
recognition model of [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
(Copyright PaddlePaddle Authors), licensed under the Apache License, Version 2.0.
`models/vin_dict.txt` is the 33-character VIN alphabet used by that model.

### `models/vin_ngram.json`

Positional n-gram statistics (per-position unigrams for positions 1–17, bigrams and trigrams
for positions 1–11, contexts seen fewer than 3 times dropped, values quantized) computed by
scanvin.app from publicly available vehicle register data.

## Code derived from other projects

- `src/core/ctc.js` — greedy CTC decoding derived from PaddleOCR's `CTCLabelDecode`
  post-processing (Apache License 2.0), rewritten for per-character confidence.
- `src/core/homography.js` — input normalization (`x / 127.5 - 1`, right zero-padding)
  follows PaddleOCR's `resize_norm_img` (Apache License 2.0). The homography solver itself
  is an original implementation.
- `src/core/yolo-post.js` — decoding of the OBB output follows the Ultralytics
  `xywhr2xyxyxyxy` convention.
- `src/core/ctc-beam.js` — CTC prefix beam search with shallow fusion, after
  Hannun, Maas, Jurafsky, Ng: *First-Pass Large Vocabulary Continuous Speech Recognition
  using Bi-Directional Recurrent DNNs* (2014). Original implementation.
- `src/core/wmi-prefixes.js` — list of World Manufacturer Identifier codes compiled from the
  public WMI list (as published e.g. on Wikipedia, "List of vehicle manufacturer codes");
  codes only, no descriptions.
- `src/core/vin.js` — ISO 3779 / FMVSS 115 check-digit algorithm (public standard).

## Runtime dependencies

- [onnxruntime-web](https://github.com/microsoft/onnxruntime) — Copyright (c) Microsoft
  Corporation, MIT License. Not bundled in this repository; installed from npm.
