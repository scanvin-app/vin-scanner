import { defineConfig } from "vite";
import { readFile } from "node:fs/promises";
import { createRequire } from "node:module";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
// Resolve through node_modules so the demo works both inside this monorepo and
// in a standalone install of the published package.
const ortDist = path.dirname(require.resolve("onnxruntime-web/ort-wasm-simd-threaded.wasm"));
const modelsDir = path.join(path.dirname(require.resolve("vin-scanner/package.json")), "models");

/**
 * Serve the package models under /models/* and the ORT wasm binary under
 * /ort/* in dev. A production app would resolve these via Vite ?url imports
 * (or copy them to its static assets) instead.
 */
function serveModelsPlugin() {
  const contentTypes = {
    ".onnx": "application/octet-stream",
    ".txt": "text/plain; charset=utf-8",
    ".wasm": "application/wasm",
  };
  const roots = {
    "/models/": modelsDir,
    "/ort/": ortDist,
  };

  return {
    name: "vin-scanner-serve-models",
    configureServer(server) {
      server.middlewares.use(async (req, res, next) => {
        const prefix = Object.keys(roots).find((p) => req.url?.startsWith(p));
        if (!prefix) return next();

        const requested = decodeURIComponent(req.url.split("?")[0].slice(prefix.length));
        const filePath = path.normalize(path.join(roots[prefix], requested));
        if (!filePath.startsWith(roots[prefix])) {
          res.statusCode = 403;
          return res.end("Forbidden");
        }
        try {
          const data = await readFile(filePath);
          res.setHeader(
            "Content-Type",
            contentTypes[path.extname(filePath)] ?? "application/octet-stream"
          );
          res.setHeader("Cache-Control", "no-store");
          res.end(data);
        } catch {
          res.statusCode = 404;
          res.end("Not found");
        }
      });
    },
  };
}

export default defineConfig({
  root: here,
  plugins: [serveModelsPlugin()],
  optimizeDeps: {
    exclude: ["onnxruntime-web"],
  },
  build: {
    assetsInlineLimit: 0,
  },
  server: {
    host: true,
  },
});
