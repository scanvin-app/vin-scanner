# Contributing

Thanks for your interest in vin-scanner.

## Ground rules

- Issues and pull requests in English, please.
- By submitting a contribution you agree that it is licensed under the AGPL-3.0, like the
  rest of the project. There is no CLA.
- Keep `packages/vin-scanner/src/core/` free of DOM, Worker and ONNX Runtime imports. The
  core is exercised by unit tests and by the Node smoke test through exactly the same code
  path the browser uses.

## Before opening a pull request

```bash
npm install
npm test          # unit tests
npm run smoke     # full pipeline in Node (needs ffmpeg)
npm run e2e       # headless Chrome (needs google-chrome and ffmpeg)
```

All three must pass. If you change thresholds in `consensus.js`, `pipeline.js` or
`ngram.js`, please describe how you validated the change (recordings, device, lighting).

## Models

Model files are binary artifacts produced by training pipelines that are not part of this
repository. Pull requests that replace a model should state the architecture, the base
model and its license, and the validation result compared to the current file.

## Reporting a misread

A VIN read incorrectly is a bug. The most useful report contains: device and browser, where
the VIN was (windshield, door sticker, stamped), lighting, what was read versus what was
expected, and if possible a short video clip you are allowed to share.
