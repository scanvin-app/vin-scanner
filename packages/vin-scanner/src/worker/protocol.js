/**
 * Main-thread ↔ worker message protocol.
 */

export const MSG = Object.freeze({
  // main → worker
  INIT: "init",
  FRAME: "frame",
  RESET: "reset",

  // worker → main
  PROGRESS: "progress",
  READY: "ready",
  INIT_ERROR: "init-error",
  RESULT: "result",
  ERROR: "error",
});
