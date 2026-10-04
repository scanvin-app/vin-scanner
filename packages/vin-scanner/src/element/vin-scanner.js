/**
 * <vin-scanner> — minimal mobile VIN scanner web component.
 *
 * Attributes:
 *   detector-src, rec-src, dict-src, wasm-src  — asset URLs (required)
 *   detect-interval-ms  — min ms between frames sent to the worker (default 200)
 *   locale              — UI language (default: en)
 *
 * Methods:  open({source}) → Promise<{supported, missing[]}>,  close()
 *
 * Events:
 *   scan-state-changed {state}   idle|loading_models|requesting_camera|scanning|processing_crop
 *   scan-model-progress {loaded, total}
 *   scan-vin-found {vin, confidence, tier, checkDigitOk, source}
 *   scan-vin-not-found {message}
 *   scan-camera-error {message, name} / scan-model-error {message} / scan-ocr-error {message}
 *   scan-opened / scan-closed
 *
 * Lifecycle contract (the iOS OOM fix): close() is a HARD teardown — the
 * worker (sole wasm heap) is terminated unconditionally, camera tracks are
 * stopped, refs nulled. Reopening builds a fresh worker; models come back
 * from the Cache API so re-init is fast.
 */

import { MSG } from "../worker/protocol.js";
import { createT } from "./i18n.js";
import { openCameraStream, stopCameraStream, torchSupported, setTorch } from "./camera.js";

const STATE = Object.freeze({
  IDLE: "idle",
  LOADING_MODELS: "loading_models",
  REQUESTING_CAMERA: "requesting_camera",
  SCANNING: "scanning",
  PROCESSING: "processing_crop",
});

const DEFAULT_DETECT_INTERVAL_MS = 200;
const MANUAL_BURST_FRAMES = 3;
const HINT_AFTER_MS = 15000;

// Importing this module must not touch the DOM: SSR frameworks (Astro, Next, Nuxt) evaluate
// it on the server when the host page imports the package root.
const BaseElement = typeof HTMLElement === "undefined" ? class {} : HTMLElement;

let template = null;
function getTemplate() {
  if (!template) {
    template = document.createElement("template");
    template.innerHTML = TEMPLATE_HTML;
  }
  return template;
}

const TEMPLATE_HTML = `
  <style>
    :host { display: contents; }
    .overlay {
      position: fixed; inset: 0; z-index: 2147483000;
      display: none; flex-direction: column;
      background: #000; color: #fff;
      font-family: system-ui, -apple-system, sans-serif;
    }
    .overlay.open { display: flex; }
    .stage { position: relative; flex: 1; overflow: hidden; }
    video { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover; }
    canvas.quads { position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none; }
    .guide {
      position: absolute; left: 50%; top: 50%; transform: translate(-50%, -50%);
      width: 82%; aspect-ratio: 8 / 1; max-height: 18%;
      border: 2px solid rgba(255,255,255,0.85); border-radius: 10px;
      box-shadow: 0 0 0 100vmax rgba(0,0,0,0.35); pointer-events: none;
    }
    .status {
      position: absolute; left: 50%; bottom: calc(24% + 72px); transform: translateX(-50%);
      max-width: 90%; padding: 6px 14px; border-radius: 999px;
      background: rgba(0,0,0,0.6); font-size: 14px; text-align: center; white-space: nowrap;
    }
    .controls {
      position: absolute; left: 0; right: 0; bottom: 0; height: 24%;
      min-height: 96px; display: flex; align-items: center; justify-content: center; gap: 48px;
    }
    button { border: 0; background: transparent; color: #fff; cursor: pointer; -webkit-tap-highlight-color: transparent; }
    .shutter {
      width: 68px; height: 68px; border-radius: 50%;
      background: #fff; border: 4px solid rgba(255,255,255,0.4); background-clip: padding-box;
    }
    .shutter:active { transform: scale(0.94); }
    .side { width: 48px; height: 48px; border-radius: 50%; background: rgba(255,255,255,0.15); font-size: 22px; line-height: 1; }
    .side[hidden] { display: none; }
    .side.on { background: rgba(255,255,255,0.5); }
    .close {
      position: absolute; top: max(12px, env(safe-area-inset-top)); right: 12px;
      width: 44px; height: 44px; border-radius: 50%;
      background: rgba(0,0,0,0.5); font-size: 22px; line-height: 1;
    }
    .progress {
      position: absolute; left: 50%; top: 50%; transform: translate(-50%, -50%);
      max-width: 80%; font-size: 15px; line-height: 1.4; text-align: center;
    }
    .retry {
      position: absolute; left: 50%; top: calc(50% + 52px); transform: translateX(-50%);
      padding: 10px 22px; border-radius: 999px;
      background: rgba(255,255,255,0.15); font-size: 15px; white-space: nowrap;
    }
    .retry[hidden] { display: none; }
  </style>
  <div class="overlay" part="overlay">
    <div class="stage">
      <video playsinline muted autoplay></video>
      <canvas class="quads"></canvas>
      <div class="guide" hidden></div>
      <div class="progress" hidden></div>
      <button class="retry" type="button" hidden></button>
      <div class="status" hidden></div>
      <button class="close" type="button">✕</button>
      <div class="controls">
        <button class="side torch" type="button" hidden>🔦</button>
        <button class="shutter" type="button"></button>
        <button class="side spacer" type="button" hidden></button>
      </div>
    </div>
  </div>
`;

export class VinScanner extends BaseElement {
  #state = STATE.IDLE;
  #worker = null;
  #stream = null;
  #frameInFlight = false;
  #lastFrameTs = 0;
  #rafId = 0;
  #manualFramesLeft = 0;
  #torchOn = false;
  #hintTimer = 0;
  #openPromise = null;
  #t = createT("en");

  #video;
  #quadCanvas;
  #guideEl;
  #progressEl;
  #retryBtn;
  #statusEl;
  #overlayEl;
  #torchBtn;

  connectedCallback() {
    if (!this.shadowRoot) {
      const root = this.attachShadow({ mode: "open" });
      root.appendChild(getTemplate().content.cloneNode(true));

      this.#overlayEl = root.querySelector(".overlay");
      this.#video = root.querySelector("video");
      this.#quadCanvas = root.querySelector("canvas.quads");
      this.#guideEl = root.querySelector(".guide");
      this.#progressEl = root.querySelector(".progress");
      this.#retryBtn = root.querySelector(".retry");
      this.#statusEl = root.querySelector(".status");
      this.#torchBtn = root.querySelector(".torch");

      root.querySelector(".close").addEventListener("click", () => this.close());
      root.querySelector(".shutter").addEventListener("click", () => this.#onShutter());
      this.#retryBtn.addEventListener("click", () => this.#onRetryCamera());
      this.#torchBtn.addEventListener("click", () => this.#onTorchToggle());
    }
    this.#applyLocale();
  }

  disconnectedCallback() {
    this.close();
  }

  static get observedAttributes() {
    return ["locale"];
  }

  attributeChangedCallback() {
    this.#applyLocale();
  }

  get state() {
    return this.#state;
  }

  get isOpen() {
    return this.#state !== STATE.IDLE;
  }

  /**
   * Feature-check, load models in a fresh worker, open the camera, start
   * the scan loop. Safe to call once per session; concurrent calls share
   * one promise.
   *
   * @returns {Promise<{supported: boolean, missing: string[]}>}
   */
  open() {
    if (this.#openPromise) return this.#openPromise;
    this.#openPromise = this.#doOpen().catch((error) => {
      this.close();
      throw error;
    });
    return this.#openPromise;
  }

  async #doOpen() {
    const support = checkSupport();
    if (!support.supported) {
      this.#emit("scan-model-error", { message: `Unsupported: ${support.missing.join(", ")}` });
      return support;
    }

    this.#overlayEl.classList.add("open");
    this.#emit("scan-opened", {});
    this.#setState(STATE.LOADING_MODELS);
    this.#progressEl.hidden = false;
    this.#progressEl.textContent = this.#t("loading");

    await this.#startWorker();

    // Camera failure keeps the overlay open with a retry affordance (see
    // #startCamera) rather than tearing everything down.
    await this.#startCamera();

    return support;
  }

  /**
   * Acquire the camera and start the scan loop. On failure it surfaces a
   * localized error + retry button and leaves the overlay (and the loaded
   * worker) intact, so retry re-acquires the camera without reloading models.
   * @returns {Promise<boolean>} whether the camera started.
   */
  async #startCamera() {
    this.#setState(STATE.REQUESTING_CAMERA);
    this.#hideCameraError();
    try {
      this.#stream = await openCameraStream();
    } catch (error) {
      this.#showCameraError(error);
      return false;
    }

    this.#video.srcObject = this.#stream;
    await this.#video.play().catch(() => {});
    this.#torchBtn.hidden = !torchSupported(this.#stream);

    this.#progressEl.hidden = true;
    this.#guideEl.hidden = false;
    this.#setStatus(this.#t("aim"));
    this.#setState(STATE.SCANNING);
    this.#armHintTimer();
    this.#pump();
    return true;
  }

  /** Map a getUserMedia error to a localized message and show retry. */
  #showCameraError(error) {
    const name = error?.name ?? "CameraError";
    const key =
      name === "NotAllowedError" || name === "SecurityError"
        ? "cameraDenied"
        : name === "NotFoundError" || name === "DevicesNotFoundError"
          ? "cameraNotFound"
          : name === "NotReadableError" || name === "TrackStartError"
            ? "cameraInUse"
            : "cameraError";
    const message = this.#t(key);
    this.#emit("scan-camera-error", { message, name });

    this.#guideEl.hidden = true;
    this.#setStatus("");
    this.#progressEl.hidden = false;
    this.#progressEl.textContent = message;
    this.#retryBtn.hidden = false;
    this.#retryBtn.textContent = this.#t("retry");
  }

  #hideCameraError() {
    this.#retryBtn.hidden = true;
  }

  #onRetryCamera() {
    if (this.#state !== STATE.REQUESTING_CAMERA) return;
    this.#hideCameraError();
    this.#progressEl.hidden = true;
    this.#startCamera();
  }

  /**
   * Hard teardown — terminate the worker (the only wasm heap), stop camera
   * tracks, cancel loops, null refs. Always safe to call.
   */
  close() {
    this.#openPromise = null;
    cancelAnimationFrame(this.#rafId);
    clearTimeout(this.#hintTimer);
    this.#hintTimer = 0;

    if (this.#worker) {
      this.#worker.terminate();
      this.#worker = null;
    }
    stopCameraStream(this.#stream);
    this.#stream = null;
    if (this.#video) {
      this.#video.srcObject = null;
    }
    this.#frameInFlight = false;
    this.#manualFramesLeft = 0;
    this.#torchOn = false;

    if (this.#overlayEl) {
      this.#overlayEl.classList.remove("open");
      this.#guideEl.hidden = true;
      this.#statusEl.hidden = true;
      this.#progressEl.hidden = true;
      this.#retryBtn.hidden = true;
      this.#clearQuads();
    }

    if (this.#state !== STATE.IDLE) {
      this.#setState(STATE.IDLE);
      this.#emit("scan-closed", {});
    }
  }

  #startWorker() {
    return new Promise((resolve, reject) => {
      const worker = new Worker(new URL("../worker/scanner.worker.js", import.meta.url), {
        type: "module",
      });
      this.#worker = worker;

      worker.onmessage = (event) => {
        const msg = event.data;
        if (msg.type === MSG.PROGRESS) {
          this.#emit("scan-model-progress", { loaded: msg.loaded, total: msg.total });
          this.#updateProgress(msg.loaded, msg.total);
        } else if (msg.type === MSG.READY) {
          /** Which decoder the worker runs: "beam" (n-gram loaded) or "greedy". */
          this.decoder = msg.decoder ?? "greedy";
          resolve();
        } else if (msg.type === MSG.INIT_ERROR) {
          this.#emit("scan-model-error", { message: msg.message });
          reject(new Error(msg.message));
        } else if (msg.type === MSG.RESULT) {
          this.#onResult(msg);
        } else if (msg.type === MSG.ERROR) {
          this.#emit("scan-ocr-error", { message: msg.message });
          this.#frameInFlight = false;
        }
      };
      worker.onerror = (event) => {
        const message = event.message ?? "Worker error";
        this.#emit("scan-model-error", { message });
        reject(new Error(message));
      };

      worker.postMessage({
        type: MSG.INIT,
        detectorUrl: absoluteUrl(this.getAttribute("detector-src")),
        recUrl: absoluteUrl(this.getAttribute("rec-src")),
        dictUrl: absoluteUrl(this.getAttribute("dict-src")),
        wasmUrl: absoluteUrl(this.getAttribute("wasm-src")),
        // Optional: n-gram rescorer model. Absent attribute = scanner runs without it.
        ngramUrl: this.getAttribute("ngram-src")
          ? absoluteUrl(this.getAttribute("ngram-src"))
          : null,
        config: {},
      });
    });
  }

  // ---- frame pump -------------------------------------------------------

  #pump() {
    if (this.#state !== STATE.SCANNING && this.#state !== STATE.PROCESSING) return;
    this.#rafId = requestAnimationFrame(() => this.#pump());

    const interval = Number(this.getAttribute("detect-interval-ms")) || DEFAULT_DETECT_INTERVAL_MS;
    const now = performance.now();
    if (this.#frameInFlight || now - this.#lastFrameTs < interval) return;
    if (this.#video.readyState < 2 || this.#video.videoWidth === 0) return;
    if (document.visibilityState === "hidden") return;

    this.#lastFrameTs = now;
    this.#captureFrame(now);
  }

  async #captureFrame(ts) {
    this.#frameInFlight = true;
    try {
      const bitmap = await createImageBitmap(this.#video);
      const manual = this.#manualFramesLeft > 0;
      if (manual) this.#manualFramesLeft -= 1;
      this.#worker?.postMessage(
        {
          type: MSG.FRAME,
          bitmap,
          ts,
          manual,
          guideRect: manual ? this.#guideRectInVideoCoords() : null,
        },
        [bitmap]
      );
    } catch {
      this.#frameInFlight = false;
    }
  }

  #onResult(msg) {
    this.#frameInFlight = false;

    this.#renderQuad(msg.detection, msg.sourceWidth, msg.sourceHeight);

    if (msg.accepted) {
      const { vin, tier, confidence, checkDigitOk } = msg.accepted;
      this.#emit("scan-vin-found", {
        vin,
        confidence,
        tier,
        checkDigitOk,
        source: msg.manual ? "manual" : "auto",
      });
      this.close();
      return;
    }

    if (msg.manual && this.#manualFramesLeft === 0) {
      // Manual burst exhausted without acceptance.
      this.#setState(STATE.SCANNING);
      if (!msg.read) {
        this.#emit("scan-vin-not-found", { message: this.#t("notFound") });
        this.#setStatus(this.#t("notFound"));
      }
    }
  }

  #onShutter() {
    if (this.#state !== STATE.SCANNING) return;
    this.#manualFramesLeft = MANUAL_BURST_FRAMES;
    this.#setState(STATE.PROCESSING);
    this.#setStatus(this.#t("processing"));
  }

  async #onTorchToggle() {
    this.#torchOn = await setTorch(this.#stream, !this.#torchOn);
    this.#torchBtn.classList.toggle("on", this.#torchOn);
  }

  // ---- overlay rendering ------------------------------------------------

  /** Map video pixel coords → displayed CSS pixels under object-fit: cover. */
  #coverTransform(sourceWidth, sourceHeight) {
    const rect = this.#video.getBoundingClientRect();
    const scale = Math.max(rect.width / sourceWidth, rect.height / sourceHeight);
    return {
      scale,
      offsetX: (rect.width - sourceWidth * scale) / 2,
      offsetY: (rect.height - sourceHeight * scale) / 2,
      width: rect.width,
      height: rect.height,
    };
  }

  #renderQuad(detection, sourceWidth, sourceHeight) {
    const canvas = this.#quadCanvas;
    const rect = this.#video.getBoundingClientRect();
    if (canvas.width !== rect.width || canvas.height !== rect.height) {
      canvas.width = rect.width;
      canvas.height = rect.height;
    }
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!detection) return;

    const { scale, offsetX, offsetY } = this.#coverTransform(sourceWidth, sourceHeight);
    const q = detection.quad;
    ctx.beginPath();
    ctx.moveTo(q[0] * scale + offsetX, q[1] * scale + offsetY);
    for (let i = 2; i < 8; i += 2) {
      ctx.lineTo(q[i] * scale + offsetX, q[i + 1] * scale + offsetY);
    }
    ctx.closePath();
    ctx.lineWidth = 3;
    ctx.strokeStyle = "rgba(74, 222, 128, 0.95)";
    ctx.stroke();
  }

  #clearQuads() {
    const ctx = this.#quadCanvas?.getContext("2d");
    ctx?.clearRect(0, 0, this.#quadCanvas.width, this.#quadCanvas.height);
  }

  /** Guide box display rect mapped back to video pixel coordinates. */
  #guideRectInVideoCoords() {
    const sourceWidth = this.#video.videoWidth;
    const sourceHeight = this.#video.videoHeight;
    if (!sourceWidth || !sourceHeight) return null;

    const videoRect = this.#video.getBoundingClientRect();
    const guideRect = this.#guideEl.getBoundingClientRect();
    const { scale, offsetX, offsetY } = this.#coverTransform(sourceWidth, sourceHeight);

    return {
      x: (guideRect.left - videoRect.left - offsetX) / scale,
      y: (guideRect.top - videoRect.top - offsetY) / scale,
      width: guideRect.width / scale,
      height: guideRect.height / scale,
    };
  }

  // ---- misc --------------------------------------------------------------

  #armHintTimer() {
    clearTimeout(this.#hintTimer);
    this.#hintTimer = setTimeout(() => {
      if (this.#state === STATE.SCANNING) {
        this.#setStatus(this.#t("hintCloser"));
      }
    }, HINT_AFTER_MS);
  }

  /** Reflect model download progress in the overlay (bytes; % if total known). */
  #updateProgress(loaded, total) {
    if (this.#state !== STATE.LOADING_MODELS) return;
    const base = this.#t("loading");
    if (total > 0) {
      const pct = Math.min(100, Math.round((loaded / total) * 100));
      this.#progressEl.textContent = `${base} ${pct}%`;
    } else if (loaded > 0) {
      const mb = (loaded / (1024 * 1024)).toFixed(1);
      this.#progressEl.textContent = `${base} ${mb} MB`;
    }
  }

  #applyLocale() {
    this.#t = createT(this.getAttribute("locale") || "en");
    if (this.shadowRoot) {
      this.shadowRoot.querySelector(".close")?.setAttribute("aria-label", this.#t("closeAria"));
      this.shadowRoot
        .querySelector(".shutter")
        ?.setAttribute("aria-label", this.#t("shutterAria"));
      this.#torchBtn?.setAttribute("aria-label", this.#t("torchAria"));
      if (this.#retryBtn) this.#retryBtn.textContent = this.#t("retry");
    }
  }

  #setStatus(text) {
    this.#statusEl.hidden = !text;
    this.#statusEl.textContent = text ?? "";
  }

  #setState(state) {
    if (this.#state === state) return;
    this.#state = state;
    this.#emit("scan-state-changed", { state });
  }

  #emit(name, detail) {
    this.dispatchEvent(new CustomEvent(name, { detail, bubbles: true, composed: true }));
  }
}

function checkSupport() {
  const missing = [];
  if (typeof window !== "undefined" && window.isSecureContext === false) {
    missing.push("secureContext");
  }
  if (typeof Worker === "undefined") missing.push("Worker");
  if (typeof OffscreenCanvas === "undefined") missing.push("OffscreenCanvas");
  if (typeof createImageBitmap === "undefined") missing.push("createImageBitmap");
  if (typeof WebAssembly === "undefined") missing.push("WebAssembly");
  if (!navigator.mediaDevices?.getUserMedia) missing.push("getUserMedia");
  return { supported: missing.length === 0, missing };
}

function absoluteUrl(value) {
  if (!value) throw new Error("Missing required model URL attribute on <vin-scanner>");
  return new URL(value, document.baseURI).href;
}
