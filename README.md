# vin-scanner

[![npm](https://img.shields.io/npm/v/vin-scanner.svg)](https://www.npmjs.com/package/vin-scanner)
[![CI](https://github.com/scanvin-app/vin-scanner/actions/workflows/ci.yml/badge.svg)](https://github.com/scanvin-app/vin-scanner/actions/workflows/ci.yml)
[![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-blue.svg)](LICENSE)

Mobile-first VIN scanner that runs entirely in the browser. Point the phone camera at a
VIN plate, windshield sticker or stamped chassis number and get the 17-character VIN back
as a DOM event. No server, no upload, no native app. The whole ML stack weighs about 5 MB:
both models together are smaller than a single photo from a modern phone, download once
and stay cached offline.

**Live demo:** [scanvin.app/en/scan-vin/](https://scanvin.app/en/scan-vin/)

<!-- TODO: docs/demo.gif -->

## Features

- **Detection + recognition on-device** with [onnxruntime-web](https://github.com/microsoft/onnxruntime)
  (WebAssembly, single thread, no COOP/COEP headers required).
- **YOLO26n-OBB detector** finds the VIN as a rotated box; **PP-OCRv6 recognizer** fine-tuned on
  VINs reads it. Both models ship with the package (about 5 MB).
- **Temporal consensus** across video frames instead of trusting a single shot: a VIN is accepted
  only when repeated reads agree, with the ISO 3779 check digit as a bonus signal.
- **Positional n-gram language model** (optional) rescoring the CTC beam search, trained on
  aggregated statistics of real VIN corpora.
- **Designed for iPhone Safari**: one worker, one wasm heap, hard teardown on close, no OpenCV.
- **Web component**, framework-agnostic, UI strings in 10 languages
  (`en, pl, de, es, fr, it, ro, cs, hu, ar`).
- Manual fallback (shutter button), torch toggle, continuous autofocus where supported.

## Install

```bash
npm install vin-scanner
```

## Usage

The component needs URLs to four assets (plus one optional): the two ONNX models, the
character dictionary, the ORT wasm binary, and optionally the n-gram model. With Vite:

```js
import "vin-scanner";
import detectorUrl from "vin-scanner/models/vin_detector.onnx?url";
import recUrl from "vin-scanner/models/ppocr-rec.onnx?url";
import dictUrl from "vin-scanner/models/vin_dict.txt?url";
import ngramUrl from "vin-scanner/models/vin_ngram.json?url";
import wasmUrl from "onnxruntime-web/ort-wasm-simd-threaded.wasm?url";

const scanner = document.createElement("vin-scanner");
scanner.setAttribute("detector-src", detectorUrl);
scanner.setAttribute("rec-src", recUrl);
scanner.setAttribute("dict-src", dictUrl);
scanner.setAttribute("ngram-src", ngramUrl);
scanner.setAttribute("wasm-src", wasmUrl);
scanner.setAttribute("locale", "en");
document.body.append(scanner);

scanner.addEventListener("scan-vin-found", (e) => {
  console.log(e.detail.vin, e.detail.tier, e.detail.checkDigitOk);
});

const { supported, missing } = await scanner.open();
if (!supported) console.warn("Unsupported browser, missing:", missing);
```

Or declaratively, with the assets copied to your static directory:

```html
<vin-scanner
  detector-src="/models/vin_detector.onnx"
  rec-src="/models/ppocr-rec.onnx"
  dict-src="/models/vin_dict.txt"
  ngram-src="/models/vin_ngram.json"
  wasm-src="/ort/ort-wasm-simd-threaded.wasm"
  locale="en"
></vin-scanner>
<script type="module">
  import "vin-scanner";
  const scanner = document.querySelector("vin-scanner");
  scanner.addEventListener("scan-vin-found", (e) => console.log(e.detail.vin));
  await scanner.open();
</script>
```

Add `assetsInclude: ["**/*.onnx"]` to your Vite config if `?url` imports of `.onnx` files
are not picked up. The wasm binary must be served with `Content-Type: application/wasm`.

### Attributes

| Attribute | Required | Description |
|---|---|---|
| `detector-src` | yes | URL of `vin_detector.onnx` |
| `rec-src` | yes | URL of `ppocr-rec.onnx` |
| `dict-src` | yes | URL of `vin_dict.txt` |
| `wasm-src` | yes | URL of `ort-wasm-simd-threaded.wasm` from `onnxruntime-web` |
| `ngram-src` | no | URL of `vin_ngram.json`; enables beam search with LM rescoring |
| `locale` | no | UI language, default `en` |
| `detect-interval-ms` | no | Minimum ms between frames sent to the worker, default `200` |

### Methods

| Method | Description |
|---|---|
| `open()` | Loads models (cached in the Cache API after the first visit), opens the camera and starts scanning. Resolves to `{ supported, missing[] }`. |
| `close()` | Hard teardown: terminates the worker, stops camera tracks. Safe to call anytime. |

### Events

| Event | `detail` |
|---|---|
| `scan-vin-found` | `{ vin, confidence, tier, checkDigitOk, source }` where `tier` is `A`, `B` or `C` and `source` is `auto` or `manual` |
| `scan-vin-not-found` | `{ message }` after a manual shutter press with no readable VIN |
| `scan-state-changed` | `{ state }`: `idle`, `loading_models`, `requesting_camera`, `scanning`, `processing_crop` |
| `scan-model-progress` | `{ loaded }` bytes downloaded so far |
| `scan-camera-error` | `{ message, name }` (`NotAllowedError`, `NotFoundError`, …); the overlay stays open with a retry button |
| `scan-model-error` | `{ message }` |
| `scan-ocr-error` | `{ message }` |
| `scan-opened`, `scan-closed` | none |

### Helpers

The package also exports pure functions you can use without the component:

```js
import { isVinShape, checkDigitOk, computeCheckDigit, wmiKnown, createConsensusTracker } from "vin-scanner";
```

## How it works

1. Every ~200 ms a camera frame is sent to a worker that runs the YOLO26n-OBB detector
   (640×640 input, NMS inside the graph).
2. The best rotated box is warped to a 48 px high strip with a pure-JS homography and read
   by the PP-OCRv6 recognizer (CTC, greedy or beam search with n-gram shallow fusion).
3. Reads accumulate in a consensus tracker. A VIN is accepted when two high-confidence reads
   agree and the check digit passes (tier A), three reads agree (tier B), or a per-position
   weighted vote settles single-character flips such as `5`/`S` (tier C).
4. `close()` terminates the worker, which is the only wasm heap, so memory is returned
   deterministically. That is what keeps iOS Safari alive across repeated scans.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the details and the pitfalls we hit.

## Browser support

Requires a secure context, `Worker`, `OffscreenCanvas`, `createImageBitmap`, WebAssembly SIMD
and `getUserMedia`. Works on iOS Safari 16.4+, Chrome for Android and desktop Chrome. `open()`
reports what is missing instead of failing mid-load.

Cold start downloads about 7 MB on the wire (wasm runtime 3 MB, models 4 MB); later visits
load models from the Cache API.

## Development

```bash
npm install
npm run dev     # demo page at http://localhost:5173 (use --host to test on a phone)
npm test        # unit tests for the core (node --test)
npm run smoke   # full pipeline in Node on a synthetic VIN image (needs ffmpeg)
npm run e2e     # headless Chrome with a fake camera (needs google-chrome and ffmpeg)
```

Repository layout: `packages/vin-scanner` is the published package, `apps/demo` is the Vite
demo and e2e harness.

## License

This project is licensed under the **GNU Affero General Public License v3.0**. See
[LICENSE](LICENSE). In short: you may use, modify and redistribute it freely, including
commercially, as long as you publish the source of your modified version under the same
license, also when you only offer it to users over a network.

The bundled detector was trained with [Ultralytics YOLO](https://github.com/ultralytics/ultralytics),
which is itself AGPL-3.0, so the AGPL applies to the model weights regardless of this
project's license choice. Third-party attributions are listed in [NOTICE.md](NOTICE.md).

**Commercial license.** If you want to use vin-scanner in a closed-source product without
the AGPL obligations, contact <contact@scanvin.app>.

If you build something with it, a link back to [scanvin.app](https://scanvin.app) is
appreciated.

---

Built by [scanvin.app](https://scanvin.app), the guide to finding VIN numbers on any vehicle.
