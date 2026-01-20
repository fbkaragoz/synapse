// This module handles loading the Wasm logic
// SvelteKit + Vite handles .wasm files if configured, but using wasm-pack's generated JS is easier.

import type { WasmParser } from './types';

export async function loadWasm(): Promise<WasmParser | null> {
  try {
    const wasm = await import('$lib/wasm_pkg/wasm_parser.js');
    await wasm.default();
    wasm.init_panic_hook();
    return wasm;
  } catch (e) {
    console.error('Failed to load Wasm:', e);
    return null;
  }
}
