// This module handles loading the Wasm logic
// SvelteKit + Vite handles .wasm files if configured, but using wasm-pack's generated JS is easier.

// We import from the generated package. 
// Note: This file will not exist until `wasm-pack build` runs.
// We use a dynamic import to avoid build-time crashes if file is missing initially.

export async function loadWasm() {
    try {
        // @ts-ignore
        const wasm = await import('$lib/wasm_pkg/wasm_parser.js');
        await wasm.default(); // Initialize the Wasm module
        // init panic hook
        wasm.init_panic_hook();
        return wasm;
    } catch (e) {
        console.error('Failed to load Wasm:', e);
        return null;
    }
}
