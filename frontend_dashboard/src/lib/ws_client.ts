import { writable, derived } from 'svelte/store';

// Connection state stores
export const isConnected = writable(false);
export const isReconnecting = writable(false);
export const reconnectAttempts = writable(0);
export const packetCount = writable(0);

// Connection health metrics
export interface ConnectionMetrics {
    latencyMs: number;
    packetsReceived: number;
    packetsDropped: number;
    bytesReceived: number;
    lastPacketTime: number;
    connectTime: number;
    reconnectCount: number;
}

export const connectionMetrics = writable<ConnectionMetrics>({
    latencyMs: 0,
    packetsReceived: 0,
    packetsDropped: 0,
    bytesReceived: 0,
    lastPacketTime: 0,
    connectTime: 0,
    reconnectCount: 0
});

// Derived store for connection quality
export const connectionQuality = derived(
    connectionMetrics,
    ($metrics) => {
        if ($metrics.packetsReceived === 0) return 'unknown';
        const dropRate = $metrics.packetsDropped / ($metrics.packetsReceived + $metrics.packetsDropped);
        if (dropRate > 0.1) return 'poor';
        if (dropRate > 0.01) return 'fair';
        if ($metrics.latencyMs > 100) return 'fair';
        return 'good';
    }
);

// Data stores
export const layerSummaries = writable<any[]>([]);
export const sparseActivations = writable<any>(null);
export const modelMeta = writable<any>(null);
export const gradientBatch = writable<any>(null);
export const attentionPattern = writable<any>(null);

// Config stores
export const configThreshold = writable<number>(0.5);
export const configAccumSteps = writable<number>(0);
export const configBroadcastInterval = writable<number>(0);
export const configMaxSparse = writable<number>(0);

// Opcodes
const OP_SET_THRESHOLD = 1;
const OP_SET_ACCUMULATION_STEPS = 2;
const OP_SET_BROADCAST_INTERVAL = 3;
const OP_SET_MAX_SPARSE_POINTS = 4;

// Reconnection configuration
interface ReconnectionConfig {
    maxAttempts: number;
    baseDelayMs: number;
    maxDelayMs: number;
    backoffFactor: number;
    jitterFactor: number;
}

const defaultReconnectConfig: ReconnectionConfig = {
    maxAttempts: 10,
    baseDelayMs: 100,
    maxDelayMs: 30000,
    backoffFactor: 2.0,
    jitterFactor: 0.1
};

let socket: WebSocket | null = null;
let wasmParser: any = null;
let reconnectTimeout: ReturnType<typeof setTimeout> | null = null;
let reconnectConfig: ReconnectionConfig = { ...defaultReconnectConfig };
let currentReconnectAttempt = 0;
let intentionalClose = false;
let pendingPings: Map<number, number> = new Map();
let pingId = 0;

const layerSummaryById = new Map<number, any>();

function resetState() {
    layerSummaryById.clear();
    layerSummaries.set([]);
    sparseActivations.set(null);
    modelMeta.set(null);
    gradientBatch.set(null);
    attentionPattern.set(null);
    packetCount.set(0);
}

function computeReconnectDelay(attempt: number): number {
    const exponential = reconnectConfig.baseDelayMs * Math.pow(reconnectConfig.backoffFactor, attempt);
    const capped = Math.min(exponential, reconnectConfig.maxDelayMs);
    const jitter = capped * reconnectConfig.jitterFactor * (Math.random() * 2 - 1);
    return Math.max(0, Math.floor(capped + jitter));
}

function scheduleReconnect(url: string) {
    if (intentionalClose) return;
    if (currentReconnectAttempt >= reconnectConfig.maxAttempts) {
        console.log('[WS] Max reconnection attempts reached');
        isReconnecting.set(false);
        return;
    }

    const delay = computeReconnectDelay(currentReconnectAttempt);
    console.log(`[WS] Reconnecting in ${delay}ms (attempt ${currentReconnectAttempt + 1}/${reconnectConfig.maxAttempts})`);
    
    isReconnecting.set(true);
    reconnectAttempts.set(currentReconnectAttempt + 1);
    
    reconnectTimeout = setTimeout(() => {
        currentReconnectAttempt++;
        doConnect(url);
    }, delay);
}

function doConnect(url: string) {
    if (socket) {
        socket.onopen = null;
        socket.onclose = null;
        socket.onmessage = null;
        socket.onerror = null;
        try { socket.close(); } catch {}
        socket = null;
    }

    try {
        socket = new WebSocket(url);
        socket.binaryType = 'arraybuffer';
        
        socket.onopen = () => {
            console.log('[WS] Connected');
            isConnected.set(true);
            isReconnecting.set(false);
            reconnectAttempts.set(0);
            currentReconnectAttempt = 0;
            intentionalClose = false;
            
            connectionMetrics.update(m => ({
                ...m,
                connectTime: Date.now(),
                reconnectCount: m.reconnectCount + (currentReconnectAttempt > 0 ? 1 : 0)
            }));
            
            resetState();
        };
        
        socket.onclose = (event) => {
            console.log(`[WS] Disconnected: code=${event.code}, reason=${event.reason}`);
            isConnected.set(false);
            
            connectionMetrics.update(m => ({
                ...m,
                latencyMs: 0
            }));
            
            if (!intentionalClose) {
                scheduleReconnect(url);
            }
        };
        
        socket.onerror = (error) => {
            console.error('[WS] Error:', error);
            connectionMetrics.update(m => ({
                ...m,
                packetsDropped: m.packetsDropped + 1
            }));
        };
        
        socket.onmessage = async (event) => {
            if (!wasmParser) {
                console.warn('[WS] WASM parser not ready');
                return;
            }

            try {
                const buffer = new Uint8Array(event.data);
                const parsed = wasmParser.parse_packet(buffer);
                
                packetCount.update(n => n + 1);
                connectionMetrics.update(m => ({
                    ...m,
                    packetsReceived: m.packetsReceived + 1,
                    bytesReceived: m.bytesReceived + buffer.length,
                    lastPacketTime: Date.now()
                }));
                
                // Calculate latency from timestamp_ns if available
                if (parsed.header?.timestamp_ns) {
                    const serverTime = Number(parsed.header.timestamp_ns);
                    const clientTime = Date.now() * 1e6; // Convert to nanoseconds
                    const latency = Math.abs(clientTime - serverTime) / 1e6; // Back to ms
                    if (latency < 10000) { // Sanity check
                        connectionMetrics.update(m => ({
                            ...m,
                            latencyMs: latency
                        }));
                    }
                }

                if (parsed.control) {
                    const c = parsed.control;
                    if (c.opcode === OP_SET_THRESHOLD) {
                        configThreshold.set(c.value_f32);
                    } else if (c.opcode === OP_SET_ACCUMULATION_STEPS) {
                        configAccumSteps.set(c.value_u32);
                    } else if (c.opcode === OP_SET_BROADCAST_INTERVAL) {
                        configBroadcastInterval.set(c.value_u32);
                    } else if (c.opcode === OP_SET_MAX_SPARSE_POINTS) {
                        configMaxSparse.set(c.value_u32);
                    }
                }

                if (parsed.summaries) {
                    for (const s of parsed.summaries) {
                        layerSummaryById.set(s.layer_id, s);
                    }
                    layerSummaries.set(
                        Array.from(layerSummaryById.values()).sort((a, b) => a.layer_id - b.layer_id)
                    );
                }
                
                if (parsed.v2_summaries) {
                    for (const s of parsed.v2_summaries) {
                        layerSummaryById.set(s.layer_id, s);
                    }
                    layerSummaries.set(
                        Array.from(layerSummaryById.values()).sort((a, b) => a.layer_id - b.layer_id)
                    );
                }
                
                if (parsed.sparse) {
                    sparseActivations.set(parsed.sparse);
                }
                
                if (parsed.meta) {
                    modelMeta.set(parsed.meta);
                }
                
                if (parsed.gradients) {
                    gradientBatch.set(parsed.gradients);
                }
                
                if (parsed.attention) {
                    attentionPattern.set(parsed.attention);
                }

            } catch (e) {
                console.error('[WS] Parse error:', e);
                connectionMetrics.update(m => ({
                    ...m,
                    packetsDropped: m.packetsDropped + 1
                }));
            }
        };
    } catch (e) {
        console.error('[WS] Connection failed:', e);
        scheduleReconnect(url);
    }
}

export function setWasmParser(parser: any) {
    wasmParser = parser;
}

export function sendControl(opcode: number, valU32: number, valF32: number) {
    if (!socket || socket.readyState !== WebSocket.OPEN) return;

    const buffer = new ArrayBuffer(48);
    const view = new DataView(buffer);

    view.setUint32(0, 0x574C464E, true);
    view.setUint16(4, 1, true);
    view.setUint16(6, 3, true);
    view.setUint32(8, 0, true);
    view.setUint32(28, 16, true);

    view.setUint32(32, opcode, true);
    view.setUint32(36, valU32, true);
    view.setFloat32(40, valF32, true);
    view.setUint32(44, 0, true);

    socket.send(buffer);
}

export function connect(url: string = 'ws://localhost:9000', config?: Partial<ReconnectionConfig>) {
    if (config) {
        reconnectConfig = { ...defaultReconnectConfig, ...config };
    }
    
    intentionalClose = false;
    doConnect(url);
}

export function disconnect() {
    intentionalClose = true;
    if (reconnectTimeout) {
        clearTimeout(reconnectTimeout);
        reconnectTimeout = null;
    }
    if (socket) {
        socket.close();
        socket = null;
    }
    isConnected.set(false);
    isReconnecting.set(false);
    resetState();
}

export function reconnect(url: string = 'ws://localhost:9000') {
    disconnect();
    intentionalClose = false;
    currentReconnectAttempt = 0;
    doConnect(url);
}

export function setReconnectConfig(config: Partial<ReconnectionConfig>) {
    reconnectConfig = { ...reconnectConfig, ...config };
}

export function getReadyState(): number {
    return socket?.readyState ?? WebSocket.CLOSED;
}
