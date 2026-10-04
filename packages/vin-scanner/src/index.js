/**
 * vin-scanner — public entry point.
 */

import { VinScanner } from "./element/vin-scanner.js";

export { VinScanner };
export { isVinShape, checkDigitOk, computeCheckDigit, wmiKnown } from "./core/vin.js";
export { createConsensusTracker, DEFAULT_CONSENSUS_CONFIG } from "./core/consensus.js";
export { SUPPORTED_LOCALES } from "./element/i18n.js";

export function registerVinScanner(tagName = "vin-scanner") {
  if (typeof customElements !== "undefined" && !customElements.get(tagName)) {
    customElements.define(tagName, VinScanner);
  }
}

registerVinScanner();
