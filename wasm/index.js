/**
 * @quantcpp/wasm — CommonJS entry point
 *
 * Re-exports the ESM module for CommonJS consumers.
 */
'use strict';

let _esmPromise = null;

function loadEsm() {
  if (!_esmPromise) {
    _esmPromise = import('./index.mjs');
  }
  return _esmPromise;
}

module.exports = {
  /**
   * Async-load the ESM Quant class.
   * Usage:
   *   const { Quant } = await require('@quantcpp/wasm').load();
   */
  load: async function () {
    return loadEsm();
  },
};
