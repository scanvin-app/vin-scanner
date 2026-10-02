# Architecture and pitfalls

This document describes how vin-scanner is put together and the non-obvious lessons behind
the design. File references point into `packages/vin-scanner/src/`.

## One worker, one wasm heap, two sessions

`worker/scanner.worker.js` hosts a single ONNX Runtime (wasm backend) instance with two
sessions:

- `vin_detector.onnx` — YOLO26n-OBB, end-to-end export with NMS inside the graph, output
  `[1, 300, 7]` (`x, y, w, h, score, class, angle`). INT8 dynamic quantization with the first
  convolution and the class branch kept in fp32; about 2.3× faster in wasm than the fp16
  export it replaced.
- `ppocr-rec.onnx` — PP-OCRv6 tiny recognizer fine-tuned on VIN strips, input
  `[1, 3, 48, W]`, output `[1, W/8, 34]` (CTC blank + 33 VIN characters).

There is no OpenCV and no separate text-detection model. The rotated box from YOLO is
warped to a 48 px strip with a pure-JS homography (`core/homography.js`) that writes
directly into the preallocated recognizer buffer with `x / 127.5 - 1` normalization.

**Module boundary:** `core/` never imports DOM, Worker or ONNX Runtime APIs (ORT is injected).
The Node smoke test (`scripts/smoke-node.mjs`) runs frames through exactly the production
code.

## Hard teardown (the expensive lesson)

An earlier scanner had two workers, each with its own ORT instance, plus OpenCV in the
bundle. iOS Safari killed the WebContent process (out of memory) on the second or third
scanner open. `InferenceSession.release()` does not return memory deterministically; the
wasm heap grows and never shrinks.

`close()` in `element/vin-scanner.js` therefore calls `worker.terminate()` unconditionally.
All memory goes away with the thread. Reopening builds a fresh worker; models come back
from the Cache API, so re-init is fast. Do not try to keep a warm session between opens.
That warm session is the OOM.

After `InferenceSession.create()` the pipeline drops its references to the model bytes.
ORT has copied them to the wasm heap already.

## ORT configuration you cannot guess

```js
ort.env.wasm.wasmPaths = { wasm: wasmUrl };
ort.env.wasm.numThreads = 1;
ort.env.wasm.proxy = false;
```

- Without COOP/COEP headers the page is not cross-origin isolated, there is no
  `SharedArrayBuffer`, and multi-threaded wasm cannot work anyway. If you do not pin
  `numThreads = 1` and `proxy = false`, ORT probes for threads and spawns proxy workers,
  which on iOS ends in a hang or a silent fallback.
- `wasmPaths` must point at the concrete `.wasm` file from your bundle. Otherwise ORT looks
  for it relative to the script and gets a 404 in production.
- Import `onnxruntime-web/wasm`, not `onnxruntime-web`, to keep WebGL/WebGPU backends out
  of the bundle.
- Vite: `optimizeDeps.exclude: ["onnxruntime-web"]` and `build.assetsInlineLimit: 0`.

Enabling COOP/COEP and threads would give a several-fold speedup, but it breaks embedding
third-party resources on the host page. Decide this at the start of a project; it sets the
per-frame time budget.

## Fixed shapes, preallocation, warmup

All tensor buffers are allocated once in `createVinPipeline` (`core/pipeline.js`) with
fixed shapes `[1,3,640,640]` and `[1,3,48,576]`. The warp writes into the recognizer
buffer with zero padding on the right instead of creating a new tensor of the "real" width
per frame. ORT's memory arena stabilizes after warmup and stops growing; variable
recognizer width (natural for PaddleOCR) caused arena reallocations at every new aspect
ratio.

Each session gets one empty `run()` after creation. The first inference compiles kernels
and is 3–10× slower; without warmup the first user frame stalls the UI.

## Preprocessing must match training bit for bit

The detector receives the frame **stretched** to 640×640, no letterbox (`core/det-pre.js`),
because that is how it was validated. If your model was trained with letterboxing
(Ultralytics default), you must pad and undo the padding in post-processing.

Two different normalizations live in one pipeline: detector `x / 255`, recognizer
`x / 127.5 - 1`. Mixing them up produces a model that "works but reads garbage", the
worst kind of bug, because nothing throws. Check this first when detections are good and
reads are bad.

## Decoding the YOLO output

`core/yolo-post.js`:

1. **Coordinate scale.** ONNX exports return either input pixels (0–640) or normalized
   0–1. `resolveCoordinateScale()` inspects the first 24 predictions and treats the output
   as normalized when `maxAbs <= 2.5`.
2. **Radians or degrees.** `normalizeAngleRadians()`: an angle above 2π but at most 360 is
   degrees.
3. **Corner order.** `rotatedRectToQuad()` yields a consistent order, and `orderQuad()` in
   `core/homography.js` normalizes it before the warp: first edge is the longer one (text
   axis), direction left to right. Without this you get upside-down or mirrored strips
   every few frames, which shows up as random garbage reads when the phone rotates.

Plausibility gates (`pickBestDetection`): aspect ratio 3–30, at least 0.05 % of the frame
area.

## Double downscale before OCR

`recognizeRegion()` in the worker does not warp from the full 1080p frame straight to
48 px. It first crops the bounding box plus an 8 % margin and scales it so the quad is
about 96 px high (2× the recognizer input height), then warps from that crop. Warping
from 1080p directly to 48 px point-samples every ~10th pixel and aliasing eats thin
strokes; `drawImage` does proper mipmapping on the way. The crop is capped at 1280×384 so
`getImageData` never copies the whole frame.

Both canvases use `getContext("2d", { willReadFrequently: true })`; otherwise Chrome keeps
the buffer on the GPU and every `getImageData` is a synchronous readback.

## Per-frame memory hygiene

- At most one frame in flight on the main thread. Otherwise a slow phone accumulates an
  unbounded queue of bitmaps.
- `bitmap.close()` in `finally`, not after success. `ImageBitmap` is outside GC.
- `tensor.dispose()` on the input and every output of every inference.
- The frame pump skips work while `document.visibilityState === "hidden"`.
- The detection interval is time-based (default 200 ms), not "every N-th rAF frame", so
  slower hardware adapts by itself.

## Camera and UI

- Request `1920×1080 ideal` with `facingMode: environment`, catch `OverconstrainedError`
  and retry with bare `facingMode` (`element/camera.js`). Older Android devices reject the
  resolution and otherwise report "no camera".
- Continuous autofocus via `applyConstraints({ advanced: [{ focusMode: "continuous" }] })`
  in try/catch. Ignored where unsupported, noticeably sharper text on Android.
- Torch via `capabilities.torch`.
- A camera error must not tear down the session: on permission denial the overlay stays
  open with a retry button and the worker with loaded models keeps running.
- Overlay coordinates under `object-fit: cover` need `scale = max(rectW/srcW, rectH/srcH)`
  plus centering offsets (`#coverTransform`). Naive scaling draws the box a few percent
  off.
- Manual fallback: when the detector finds nothing, the shutter button runs three
  overlapping horizontal strips of the guide frame through the same warp code
  (`recognizeGuideStrips`). This rescues cases such as a VIN printed in a registration
  document.
- Models are cached with `caches.open("vin-scanner-models-v1")`, falling back to a plain
  `fetch` where the Cache API is unavailable (private mode). Use content-hashed URLs so a
  model swap invalidates the cache by itself.
- Download progress is read from `response.body.getReader()`. `Content-Length` is often
  missing behind compressing CDNs, so the UI shows megabytes, not percent.
- `open()` checks secure context, `Worker`, `OffscreenCanvas`, `createImageBitmap`,
  WebAssembly and `getUserMedia` up front and returns the list of missing features.

## Temporal consensus (`core/consensus.js`)

Instead of soft substitutions, line stitching and single-frame scoring, 17-character
reads accumulate across frames. Acceptance tiers, first match wins:

- **Tier A** — two agreeing high-confidence reads plus the ISO 3779 check digit. The WMI
  alone turned out too weak as an external signal: it vouches for the first three
  characters only, and a repeated `0`→`8` flip in the serial once passed through it.
- **Tier B** — three agreeing reads (four when the n-gram beam decoder is active, because
  the beam homogenizes reads and repetition is worth less).
- **Tier C** — per-position weighted vote when no exact string repeats (single-character
  flips such as `5`/`S` across frames).

**Contested position:** when the runner-up character at some position has a higher mean
confidence than the weight-based winner, that is the signature of a systematic error
(for example glare turning a `0` into an `8`). Tier C is held back until tier B resolves
it. This eliminated the wrong acceptances seen in benchmarks.

The check digit is only a bonus. Many European VINs legitimately do not satisfy it.

## Language model (`core/ngram.js`, `core/ctc-beam.js`)

`models/vin_ngram.json` holds positional unigram, bigram and trigram statistics (positions
1–11 for the structural part, unigrams only in the serial). The worker uses it in two
places:

- CTC prefix beam search with shallow fusion (n-gram as the LM) instead of greedy decoding.
- Post-hoc length repair: an 18-character read loses one character, a 16-character read
  gains one, but only when the best candidate beats the runner-up by a margin
  (`lengthRepairMargin`). Inside the serial every digit scores alike, so the repair refuses
  to guess there; a pure edge deletion wins ties because OCR insertions come from
  neighbouring stamped characters at the strip edges.

Missing `ngram-src` or a failed download falls back to the greedy decoder. Scanner start is
never blocked by the n-gram.

## Asset budget (cold start)

| Asset | raw | on the wire (br/gzip) |
|---|---|---|
| element + worker JS | ~135 KB | ~40 KB |
| ort-wasm-simd-threaded.wasm | 12.3 MB | ~3.1 MB |
| vin_detector.onnx (INT8, optimized offline) | 2.9 MB | ~2.1 MB |
| ppocr-rec.onnx (v6 tiny, optimized offline) | 2.2 MB | ~1.7 MB |
| vin_ngram.json (optional) | 0.7 MB | ~0.19 MB |

Both models are passed offline through ORT graph optimization at the "extended" level with
the same ORT version as `onnxruntime-web`, and the session uses
`graphOptimizationLevel: "basic"`, skipping the fusion pass on every cold start with
bit-identical outputs (detector session creation 239 → 170 ms, recognizer 37 → 9 ms in
Node). A raw model dropped in should use `"all"`.

Repeat visits download only the JS; models and wasm come from the Cache API or HTTP
`immutable` caching.
