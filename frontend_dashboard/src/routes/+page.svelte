<script lang="ts">
    import { onMount, tick } from 'svelte';
    import { SceneController } from '$lib/three/scene';
    import { 
        isConnected, packetCount, layerSummaries, sparseActivations, modelMeta,
        configThreshold, sendControl, setWasmParser, connect 
    } from '$lib/ws_client';
    import { loadWasm } from '$lib/wasm_bridge';

    let canvas: HTMLCanvasElement;
    let sceneController: SceneController;

    // Local state for debouncing
    let thresholdLocal = 0.5;

    const thresholdId = 'nf-threshold';
    const themeId = 'nf-theme';
    
    function onThresholdChange(e: Event) {
        const val = parseFloat((e.target as HTMLInputElement).value);
        thresholdLocal = val;
        sendControl(1, 0, val); // Opcode 1 = SET_THRESHOLD
    }
    
    function toggleBloom(e: Event) {
        const checked = (e.target as HTMLInputElement).checked;
        if (sceneController) sceneController.setBloom(checked);
    }

    function onThemeChange(e: Event) {
        const theme = (e.target as HTMLSelectElement).value;
        if (sceneController) sceneController.setTheme(theme);
    }

    // Subscribe to store updates to sync backend state
    onMount(() => {
        let unsubscribeThreshold = () => {};
        let unsubscribeMeta = () => {};
        let unsubscribeSummaries = () => {};
        let unsubscribeSparse = () => {};
        const onResize = () => sceneController?.resize(window.innerWidth, window.innerHeight);
        let destroyed = false;

        (async () => {
            await tick();
            if (destroyed) return;

            sceneController = new SceneController(canvas);

            const wasm = await loadWasm();
            if (destroyed) return;

            if (wasm) {
                setWasmParser(wasm);
            } else {
                console.warn('[UI] Wasm parser failed to load; packets will not parse.');
            }

            connect();

            unsubscribeThreshold = configThreshold.subscribe((val: number) => {
                if (Math.abs(val - thresholdLocal) > 0.01) {
                    thresholdLocal = val;
                }
            });

            unsubscribeMeta = modelMeta.subscribe((meta) => {
                if (sceneController && meta) {
                    sceneController.setupTopology(meta);
                }
            });

            unsubscribeSummaries = layerSummaries.subscribe((data) => {
                if (sceneController && data.length > 0) {
                    sceneController.updateLayers(data);
                }
            });

            unsubscribeSparse = sparseActivations.subscribe((data) => {
                if (sceneController && data) {
                    sceneController.updateSparse(data);
                }
            });

            window.addEventListener('resize', onResize);
        })();

        return () => {
            destroyed = true;
            unsubscribeThreshold();
            unsubscribeMeta();
            unsubscribeSummaries();
            unsubscribeSparse();
            window.removeEventListener('resize', onResize);
        };
    });
</script>

<div class="overlay">
    <h1>Neural-Flow</h1>
    <div class="status">
        Status: <span class:connected={$isConnected}>{$isConnected ? 'Connected' : 'Disconnected'}</span>
        <br>
        Packets: {$packetCount}
    </div>
</div>

<div class="controls">
    <h2>Control Panel</h2>
    
    <div class="control-group">
        <label for={thresholdId}>Threshold: {thresholdLocal.toFixed(2)}</label>
        <input type="range" min="0.0" max="1.0" step="0.01" 
               id={thresholdId}
               value={thresholdLocal} 
               on:input={onThresholdChange} />
    </div>
    
    <div class="control-group">
        <label for={themeId}>Theme</label>
        <select id={themeId} on:change={onThemeChange}>
            <option value="magma">Magma (Red)</option>
            <option value="matrix">Matrix (Green)</option>
            <option value="cyberpunk">Cyberpunk (Cyan)</option>
        </select>
    </div>

    <div class="control-group">
        <label>
            <input type="checkbox" on:change={toggleBloom} /> Enable Bloom
        </label>
    </div>
</div>

<canvas bind:this={canvas} />

<style>
    :global(body) {
        margin: 0;
        overflow: hidden;
        background-color: #111;
        color: white;
        font-family: 'Inter', sans-serif;
    }
    .overlay {
        position: absolute;
        top: 20px;
        left: 20px;
        pointer-events: none;
    }
    .status {
        margin-top: 10px;
        font-family: monospace;
        color: #888;
    }
    .connected {
        color: #0f0;
    }
    
    .controls {
        position: absolute;
        top: 20px;
        right: 20px;
        width: 250px;
        background: rgba(30, 30, 30, 0.8);
        backdrop-filter: blur(10px);
        padding: 20px;
        border-radius: 8px;
        border: 1px solid #333;
    }
    .controls h2 {
        margin-top: 0;
        font-size: 1.2rem;
        border-bottom: 1px solid #444;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
    .control-group {
        margin-bottom: 15px;
    }
    .control-group label {
        display: block;
        margin-bottom: 5px;
        font-size: 0.9rem;
        color: #ccc;
    }
    input[type=range] {
        width: 100%;
    }
    select {
        width: 100%;
        background: #222;
        color: white;
        border: 1px solid #555;
        padding: 5px;
        border-radius: 4px;
    }
</style>
