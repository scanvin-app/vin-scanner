/**
 * Model fetching with Cache API persistence and download progress.
 *
 * Model URLs are content-hashed by the host bundler (/_astro/…), so cached
 * entries are naturally invalidated when a model changes. The Cache API
 * survives page reloads — a warm reopen costs no network at all.
 */

const CACHE_NAME = "vin-scanner-models-v1";

/**
 * @param {string} url
 * @param {(loadedBytes: number) => void} [onProgress]
 * @returns {Promise<ArrayBuffer>}
 */
export async function fetchModel(url, onProgress) {
  let cache = null;
  try {
    cache = await caches.open(CACHE_NAME);
    const hit = await cache.match(url);
    if (hit) {
      const buffer = await hit.arrayBuffer();
      onProgress?.(buffer.byteLength);
      return buffer;
    }
  } catch {
    // Cache API unavailable (e.g. some private-browsing modes) — plain fetch.
  }

  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch ${url}: HTTP ${response.status}`);
  }

  if (cache) {
    try {
      await cache.put(url, response.clone());
    } catch {
      // Quota exceeded etc. — not fatal.
    }
  }

  if (!response.body || !onProgress) {
    const buffer = await response.arrayBuffer();
    onProgress?.(buffer.byteLength);
    return buffer;
  }

  const reader = response.body.getReader();
  const chunks = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.byteLength;
    onProgress(loaded);
  }

  const buffer = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    buffer.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return buffer.buffer;
}
