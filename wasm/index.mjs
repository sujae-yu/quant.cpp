/**
 * @quantcpp/wasm — ESM entry point
 *
 * Single-header C LLM inference engine in your browser.
 *
 * Usage:
 *
 *   import { Quant } from '@quantcpp/wasm';
 *
 *   const q = await Quant.create({
 *     modelUrl: '/models/SmolLM2-135M-Instruct-Q8_0.gguf',
 *     kvType: 'uniform_4b',
 *     vQuant: 'q4',
 *     onStatus: (msg) => console.log('[status]', msg),
 *   });
 *
 *   await q.generate('Hello, my name is', {
 *     maxTokens: 64,
 *     temperature: 0.7,
 *     onToken: (text) => process.stdout.write(text),
 *   });
 *
 *   q.free();
 */

let _modulePromise = null;

function loadEmscriptenModule(scriptUrl) {
  if (_modulePromise) return _modulePromise;
  _modulePromise = new Promise((resolve, reject) => {
    if (typeof window === 'undefined') {
      reject(new Error('Node.js loader not implemented yet — use the browser build for now'));
      return;
    }
    const script = document.createElement('script');
    script.src = scriptUrl;
    script.onload = () => {
      // Emscripten modularize=0 attaches Module to globalThis
      if (typeof Module === 'undefined') {
        reject(new Error('quant.js loaded but Module is undefined'));
        return;
      }
      Module.onRuntimeInitialized = () => resolve(Module);
    };
    script.onerror = () => reject(new Error(`Failed to load ${scriptUrl}`));
    document.head.appendChild(script);
  });
  return _modulePromise;
}

export class Quant {
  constructor(module) {
    this._m = module;
    this._loaded = false;
  }

  /**
   * Create a Quant instance, optionally loading a model.
   * @param {object} opts
   * @param {string} [opts.scriptUrl='./quant.js']
   * @param {string} [opts.modelUrl] - URL to a .gguf model file
   * @param {string} [opts.kvType='uniform_4b'] - one of fp32, uniform_4b, turbo_kv_3b, ...
   * @param {string} [opts.vQuant='fp16'] - one of fp16, q4, q2
   * @param {function} [opts.onStatus] - status callback
   */
  static async create(opts = {}) {
    const scriptUrl = opts.scriptUrl || './quant.js';
    const module = await loadEmscriptenModule(scriptUrl);

    if (opts.onStatus) module.onStatus = opts.onStatus;

    const q = new Quant(module);

    if (opts.modelUrl) {
      await q.loadModel(opts.modelUrl);
    }

    return q;
  }

  /**
   * Load a GGUF model from a URL into the WASM filesystem.
   */
  async loadModel(url) {
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`Failed to fetch model: ${resp.status} ${resp.statusText}`);
    const buf = new Uint8Array(await resp.arrayBuffer());
    const path = '/model.gguf';
    this._m.FS.writeFile(path, buf);
    const ret = this._m.ccall('wasm_load_model', 'number', ['string'], [path]);
    if (ret !== 0) throw new Error(`wasm_load_model failed (rc=${ret})`);
    this._loaded = true;
  }

  /**
   * Generate text from a prompt.
   * @param {string} prompt
   * @param {object} opts
   * @param {number} [opts.maxTokens=128]
   * @param {number} [opts.temperature=0.7]
   * @param {function} [opts.onToken] - called per token with (text) string
   * @param {function} [opts.onDone] - called with (nTokens, elapsedMs)
   */
  generate(prompt, opts = {}) {
    if (!this._loaded) throw new Error('No model loaded — call loadModel() first or pass modelUrl to create()');
    if (opts.onToken) this._m.onToken = opts.onToken;
    if (opts.onDone) this._m.onDone = opts.onDone;
    const maxTokens = opts.maxTokens ?? 128;
    const temperature = opts.temperature ?? 0.7;
    return new Promise((resolve) => {
      this._m.onDone = (nTokens, elapsedMs) => {
        if (opts.onDone) opts.onDone(nTokens, elapsedMs);
        resolve({ nTokens, elapsedMs });
      };
      this._m.ccall('wasm_generate', 'number', ['string', 'number', 'number'], [prompt, maxTokens, temperature]);
    });
  }

  /**
   * Free model resources. Call when done.
   */
  free() {
    if (this._loaded) {
      this._m.ccall('wasm_free_model', null, [], []);
      this._loaded = false;
    }
  }

  isReady() {
    return this._loaded;
  }
}

export default Quant;
