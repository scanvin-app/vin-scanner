#!/usr/bin/env bash
# Real end-to-end smoke: headless Chrome with a fake capture device playing a
# synthetic VIN image into getUserMedia; passes when <vin-scanner> emits
# scan-vin-found with the expected VIN via the full worker pipeline.
#
# Requires: google-chrome (or set CHROME_BIN), ffmpeg, curl.
set -euo pipefail

DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="$(cd "${DEMO_DIR}/../.." && pwd)"
HOST="${VIN_SCANNER_E2E_HOST:-127.0.0.1}"
PORT="${VIN_SCANNER_E2E_PORT:-4179}"
OUT_DIR="${DEMO_DIR}/e2e/out"
PNG="${OUT_DIR}/synthetic-vin.png"
Y4M="${OUT_DIR}/synthetic-vin.y4m"
SERVER_LOG="${OUT_DIR}/dev-server.log"
SERVER_PID=""

cleanup() {
  if [[ -n "${SERVER_PID}" ]]; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

mkdir -p "${OUT_DIR}"

# Fake camera feed: loop the synthetic VIN image for a few seconds.
VIN="$(bash "${REPO_DIR}/scripts/synthetic-vin-image.sh" "${PNG}")"
ffmpeg -v error -loop 1 -i "${PNG}" -t 6 -r 15 -pix_fmt yuv420p -f yuv4mpegpipe -y "${Y4M}"

URL="http://${HOST}:${PORT}/e2e.html?vin=${VIN}"

npm --prefix "${DEMO_DIR}" run dev -- --host "${HOST}" --port "${PORT}" --strictPort >"${SERVER_LOG}" 2>&1 &
SERVER_PID="$!"

for _ in $(seq 1 60); do
  if curl -fsS "${URL}" >/dev/null 2>&1; then
    break
  fi
  sleep 0.5
done

if ! curl -fsS "${URL}" >/dev/null 2>&1; then
  echo "E2E failed: dev server not responding at ${URL}" >&2
  echo "Server log: ${SERVER_LOG}" >&2
  exit 1
fi

# dump-dom freezes page timers and virtual-time races ahead of wasm work,
# so the verdict is polled over CDP in real time instead.
if node "${DEMO_DIR}/e2e/cdp.mjs" "${URL}" 40000 \
  --use-fake-ui-for-media-stream \
  --use-fake-device-for-media-stream \
  --use-file-for-fake-video-capture="${Y4M}" \
  --autoplay-policy=no-user-gesture-required; then
  echo "E2E passed (${VIN})"
  exit 0
fi

echo "E2E failed. Server log: ${SERVER_LOG}" >&2
exit 1
