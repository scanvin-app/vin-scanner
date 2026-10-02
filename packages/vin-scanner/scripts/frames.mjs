/**
 * Frame extraction for the benchmark harness — decodes images and videos
 * to raw RGBA via ffmpeg (no native Node canvas/codec dependencies).
 */

import { spawn } from "node:child_process";

/**
 * Probe a media file. `width`/`height` are the dimensions of the frames
 * ffmpeg actually OUTPUTS: the CLI autorotates by the stream's display
 * matrix (phone portrait clips are stored landscape with rotation ±90),
 * so stored dimensions are swapped when needed.
 *
 * @param {string} filePath
 * @returns {Promise<{ width: number, height: number, durationSec: number|null }>}
 */
export async function probeMedia(filePath) {
  const out = await runCollect("ffprobe", [
    "-v", "error",
    "-select_streams", "v:0",
    "-show_streams",
    "-show_entries", "format=duration",
    "-of", "json",
    filePath,
  ]);
  const info = JSON.parse(out);
  const stream = info.streams?.[0];
  if (!stream?.width || !stream?.height) {
    throw new Error(`ffprobe found no video stream in ${filePath}`);
  }

  let rotation = Number(stream.tags?.rotate ?? 0);
  for (const sideData of stream.side_data_list ?? []) {
    if (Number.isFinite(Number(sideData.rotation))) {
      rotation = Number(sideData.rotation);
    }
  }
  const swapped = Math.abs(((rotation % 360) + 360) % 360) % 180 === 90;

  const duration = Number(info.format?.duration);
  return {
    width: swapped ? stream.height : stream.width,
    height: swapped ? stream.width : stream.height,
    durationSec: Number.isFinite(duration) ? duration : null,
  };
}

/**
 * Decode a still image to RGBA.
 *
 * @param {string} filePath
 * @returns {Promise<{ data: Uint8ClampedArray, width: number, height: number }>}
 */
export async function decodeImage(filePath) {
  const { width, height } = await probeMedia(filePath);
  const raw = await runCollectBinary("ffmpeg", [
    "-v", "error",
    "-i", filePath,
    "-frames:v", "1",
    "-f", "rawvideo",
    "-pix_fmt", "rgba",
    "-",
  ]);
  const expected = width * height * 4;
  if (raw.length !== expected) {
    throw new Error(`Unexpected frame size ${raw.length}, expected ${expected}`);
  }
  return { data: new Uint8ClampedArray(raw.buffer, raw.byteOffset, expected), width, height };
}

/**
 * Stream video frames at a fixed sampling rate as RGBA images.
 *
 * @param {string} filePath
 * @param {{ fps?: number }} [options]
 * @returns {AsyncGenerator<{ data: Uint8ClampedArray, width: number, height: number,
 *                            index: number, timeSec: number }>}
 */
export async function* streamVideoFrames(filePath, options = {}) {
  const { fps = 5 } = options;
  const { width, height } = await probeMedia(filePath);
  const frameBytes = width * height * 4;

  const proc = spawn("ffmpeg", [
    "-v", "error",
    "-i", filePath,
    "-vf", `fps=${fps}`,
    "-f", "rawvideo",
    "-pix_fmt", "rgba",
    "-",
  ]);

  let stderr = "";
  proc.stderr.on("data", (chunk) => {
    stderr += chunk;
  });

  // Register before consuming stdout: with a slow consumer ffmpeg can exit
  // (and emit "close") while frames are still being processed — attaching the
  // listener afterwards would await forever and the process would exit early
  // with "unsettled top-level await".
  const closed = new Promise((resolve) => proc.on("close", resolve));

  let buffered = Buffer.alloc(0);
  let index = 0;

  for await (const chunk of proc.stdout) {
    buffered = buffered.length === 0 ? chunk : Buffer.concat([buffered, chunk]);
    while (buffered.length >= frameBytes) {
      const frame = buffered.subarray(0, frameBytes);
      buffered = buffered.subarray(frameBytes);
      yield {
        // Copy: the subarray view aliases the growing buffer.
        data: new Uint8ClampedArray(frame),
        width,
        height,
        index,
        timeSec: index / fps,
      };
      index += 1;
    }
  }

  const exitCode = await closed;
  if (exitCode !== 0) {
    throw new Error(`ffmpeg exited with code ${exitCode}: ${stderr}`);
  }
}

function runCollect(command, args) {
  return runCollectBinary(command, args).then((buf) => buf.toString("utf8"));
}

function runCollectBinary(command, args) {
  return new Promise((resolve, reject) => {
    const proc = spawn(command, args);
    const chunks = [];
    let stderr = "";
    proc.stdout.on("data", (chunk) => chunks.push(chunk));
    proc.stderr.on("data", (chunk) => {
      stderr += chunk;
    });
    proc.on("error", reject);
    proc.on("close", (code) => {
      if (code === 0) resolve(Buffer.concat(chunks));
      else reject(new Error(`${command} exited with code ${code}: ${stderr}`));
    });
  });
}
