/**
 * Type definitions for @quantcpp/wasm
 */

export type KVType =
  | 'fp32'
  | 'uniform_4b'
  | 'uniform_2b'
  | 'uniform_3b'
  | 'polar_3b'
  | 'polar_4b'
  | 'qjl_1b'
  | 'turbo_3b'
  | 'turbo_4b'
  | 'turbo_kv_1b'
  | 'turbo_kv_3b'
  | 'turbo_kv_4b';

export type VQuant = 'fp16' | 'q4' | 'q2';

export interface QuantCreateOptions {
  /** URL to load quant.js from. Default: './quant.js' */
  scriptUrl?: string;
  /** Optional URL of a .gguf model to load on creation. */
  modelUrl?: string;
  /** KV cache quantization type. Default: 'uniform_4b'. */
  kvType?: KVType;
  /** Value cache quantization. Default: 'fp16'. */
  vQuant?: VQuant;
  /** Status callback for engine messages. */
  onStatus?: (message: string) => void;
}

export interface GenerateOptions {
  /** Maximum tokens to generate. Default: 128. */
  maxTokens?: number;
  /** Sampling temperature. Default: 0.7. */
  temperature?: number;
  /** Per-token callback for streaming. */
  onToken?: (text: string) => void;
  /** Called when generation completes. */
  onDone?: (nTokens: number, elapsedMs: number) => void;
}

export interface GenerateResult {
  nTokens: number;
  elapsedMs: number;
}

export class Quant {
  private constructor(module: unknown);

  /**
   * Create a Quant instance. If `modelUrl` is provided, the model is
   * loaded before the promise resolves.
   */
  static create(options?: QuantCreateOptions): Promise<Quant>;

  /** Load a GGUF model from a URL. */
  loadModel(url: string): Promise<void>;

  /** Generate text from a prompt. Returns when generation completes. */
  generate(prompt: string, options?: GenerateOptions): Promise<GenerateResult>;

  /** Free model resources. */
  free(): void;

  /** Whether a model is currently loaded. */
  isReady(): boolean;
}

export default Quant;
