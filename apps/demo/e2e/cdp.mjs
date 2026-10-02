/**
 * Drive headless Chrome over CDP (no dependencies — Node's global WebSocket)
 * and wait for the e2e page verdict in real time. dump-dom freezes timers,
 * virtual-time races ahead of wasm work — polling over CDP does neither.
 *
 *   node e2e/cdp.mjs <url> [timeoutMs] [extra chrome flags...]
 *
 * Set CHROME_BIN to point at a Chrome/Chromium binary (default: google-chrome).
 */

import { spawn } from "node:child_process";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";

const url = process.argv[2];
const timeoutMs = Number(process.argv[3] ?? 40000);
const extraFlags = process.argv.slice(4);
if (!url) {
  console.error("usage: node cdp.mjs <url> [timeoutMs] [chrome flags...]");
  process.exit(2);
}

const chrome = spawn(process.env.CHROME_BIN ?? "google-chrome", [
  "--headless=new",
  "--disable-gpu",
  "--no-sandbox",
  "--remote-debugging-port=0",
  // Fresh profile per run: a persistent one would keep the models in the
  // Cache API under the hash-less dev URL, so a swapped model would never load.
  `--user-data-dir=${mkdtempSync(path.join(tmpdir(), "vin-scanner-e2e-"))}`,
  ...extraFlags,
  "about:blank",
]);

let wsUrl = null;
let stderrBuf = "";
chrome.stderr.on("data", (chunk) => {
  stderrBuf += chunk;
  const match = stderrBuf.match(/DevTools listening on (ws:\/\/\S+)/);
  if (match && !wsUrl) {
    wsUrl = match[1];
    run().catch((error) => finish(1, `CDP error: ${error.message}`));
  }
});

const killTimer = setTimeout(() => finish(1, "fail: global timeout"), timeoutMs + 15000);

function finish(code, message) {
  clearTimeout(killTimer);
  console.log(message);
  chrome.kill("SIGKILL");
  process.exit(code);
}

async function run() {
  // Find the page target via the browser endpoint.
  const httpBase = wsUrl.replace("ws://", "http://").split("/devtools")[0];
  let pageWs = null;
  for (let i = 0; i < 50 && !pageWs; i += 1) {
    const targets = await fetch(`${httpBase}/json/list`).then((r) => r.json());
    pageWs = targets.find((t) => t.type === "page")?.webSocketDebuggerUrl ?? null;
    if (!pageWs) await new Promise((r) => setTimeout(r, 200));
  }
  if (!pageWs) throw new Error("no page target");

  const ws = new WebSocket(pageWs);
  await new Promise((resolve, reject) => {
    ws.onopen = resolve;
    ws.onerror = () => reject(new Error("ws connect failed"));
  });

  let nextId = 1;
  const pending = new Map();
  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    if (msg.id && pending.has(msg.id)) {
      pending.get(msg.id)(msg);
      pending.delete(msg.id);
    }
  };
  const send = (method, params = {}) =>
    new Promise((resolve) => {
      const id = nextId++;
      pending.set(id, resolve);
      ws.send(JSON.stringify({ id, method, params }));
    });

  await send("Page.enable");
  await send("Runtime.enable");
  await send("Page.navigate", { url });

  const evaluate = async (expression) => {
    const reply = await send("Runtime.evaluate", { expression, returnByValue: true });
    return reply.result?.result?.value;
  };

  const t0 = Date.now();
  for (;;) {
    if (Date.now() - t0 > timeoutMs) {
      const logText = await evaluate("document.getElementById('log')?.textContent ?? ''");
      finish(1, `fail: timeout\n--- page log ---\n${logText}`);
    }
    const verdict = await evaluate(
      "document.getElementById('result')?.dataset.testResult ?? 'missing'"
    );
    if (verdict === "pass" || verdict === "fail") {
      const text = await evaluate("document.getElementById('result').textContent");
      const logText = await evaluate("document.getElementById('log')?.textContent ?? ''");
      finish(verdict === "pass" ? 0 : 1, `${text}\n--- page log ---\n${logText}`);
    }
    await new Promise((r) => setTimeout(r, 500));
  }
}
