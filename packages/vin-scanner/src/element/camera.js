/**
 * Camera stream helpers — acquisition, torch, teardown.
 */

const CONSTRAINTS = {
  audio: false,
  video: {
    facingMode: { ideal: "environment" },
    width: { ideal: 1920 },
    height: { ideal: 1080 },
  },
};

/** Fallback when the ideal resolution is rejected (OverconstrainedError). */
const RELAXED_CONSTRAINTS = {
  audio: false,
  video: { facingMode: { ideal: "environment" } },
};

/**
 * @returns {Promise<MediaStream>}
 * @throws {DOMException}
 */
export async function openCameraStream() {
  const stream = await getUserMediaWithFallback();
  const [track] = stream.getVideoTracks();
  // Continuous focus where supported (ignored elsewhere).
  try {
    const capabilities = track.getCapabilities?.();
    if (capabilities?.focusMode?.includes("continuous")) {
      await track.applyConstraints({ advanced: [{ focusMode: "continuous" }] });
    }
  } catch {
    // Non-fatal.
  }
  return stream;
}

/**
 * Try the ideal 1080p constraints first; on OverconstrainedError retry with
 * bare facingMode so older/unusual cameras that reject the resolution still
 * open. Other errors (denied, not found, in use) propagate unchanged.
 * @returns {Promise<MediaStream>}
 */
async function getUserMediaWithFallback() {
  try {
    return await navigator.mediaDevices.getUserMedia(CONSTRAINTS);
  } catch (error) {
    if (error?.name === "OverconstrainedError") {
      return navigator.mediaDevices.getUserMedia(RELAXED_CONSTRAINTS);
    }
    throw error;
  }
}

/**
 * @param {MediaStream|null} stream
 */
export function stopCameraStream(stream) {
  if (!stream) return;
  for (const track of stream.getTracks()) {
    try {
      track.stop();
    } catch {
      // Already stopped.
    }
  }
}

/**
 * @param {MediaStream|null} stream
 * @returns {boolean}
 */
export function torchSupported(stream) {
  const track = stream?.getVideoTracks()[0];
  const capabilities = track?.getCapabilities?.();
  return Boolean(capabilities?.torch);
}

/**
 * @param {MediaStream|null} stream
 * @param {boolean} on
 * @returns {Promise<boolean>} actual torch state after the call.
 */
export async function setTorch(stream, on) {
  const track = stream?.getVideoTracks()[0];
  if (!track) return false;
  try {
    await track.applyConstraints({ advanced: [{ torch: on }] });
    return on;
  } catch {
    return false;
  }
}
