import { writable } from 'svelte/store';

export const isConnected = writable(false);
export const packetCount = writable(0);

// We will expose a way to subscribe to parsed packets.
// Stores
export const layerSummaries = writable<any[]>([]);
export const sparseActivations = writable<any>(null);
export const modelMeta = writable<any>(null);

// Stores for Backend Config
export const configThreshold = writable<number>(0.5);
export const configAccumSteps = writable<number>(0);
export const configBroadcastInterval = writable<number>(0);
export const configMaxSparse = writable<number>(0);

// Opcodes
const OP_SET_THRESHOLD = 1;
const OP_SET_ACCUMULATION_STEPS = 2;
const OP_SET_BROADCAST_INTERVAL = 3;
const OP_SET_MAX_SPARSE_POINTS = 4;

let socket: WebSocket | null = null;
let wasmParser: any = null;

// Accumulate per-layer macro summaries so the UI can render a full stack.
const layerSummaryById = new Map<number, any>();

function resetState() {
    layerSummaryById.clear();
    layerSummaries.set([]);
    sparseActivations.set(null);
    modelMeta.set(null);
    packetCount.set(0);
}

export function setWasmParser(parser: any) {
    wasmParser = parser;
}

export function sendControl(opcode: number, valU32: number, valF32: number) {
    if (!socket || socket.readyState !== WebSocket.OPEN) return;

    const buffer = new ArrayBuffer(48); // 32 header + 16 payload
    const view = new DataView(buffer);

    // Header
    view.setUint32(0, 0x574C464E, true); // Magic
    view.setUint16(4, 1, true);          // Version
    view.setUint16(6, 3, true);          // MsgType = NF_MSG_CONTROL
    view.setUint32(8, 0, true);          // Flags
    view.setUint32(28, 16, true);        // PayloadBytes

    // Payload (ControlPacket)
    view.setUint32(32, opcode, true);
    view.setUint32(36, valU32, true);
    view.setFloat32(40, valF32, true);
    view.setUint32(44, 0, true);         // Reserved

    socket.send(buffer);
}

export function connect(url: string = 'ws://localhost:9000') {
    if (socket) {
        socket.close();
    }

    resetState();

    socket = new WebSocket(url);
    socket.binaryType = 'arraybuffer';

    socket.onopen = () => {
        console.log('[WS] Connected');
        isConnected.set(true);
    };

    socket.onclose = () => {
        console.log('[WS] Disconnected');
        isConnected.set(false);
        resetState();
    };

    socket.onmessage = async (event) => {
        // console.log('[WS] Raw event:', event.data.byteLength);
        if (!wasmParser) {
             console.warn('[WS] Wasm parser not ready');
             return;
        }

        try {
            const buffer = new Uint8Array(event.data);
            // console.log('[WS] Parsing buffer value:', buffer[0], buffer.length); 
            const parsed = wasmParser.parse_packet(buffer);
            
            packetCount.update(n => n + 1);
            // console.log('[WS] Parsed:', parsed);

            if (parsed.control) {
                const c = parsed.control;
                console.log('[WS] Control Packet:', c);
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
                layerSummaries.set(Array.from(layerSummaryById.values()).sort((a, b) => a.layer_id - b.layer_id));
            }
            if (parsed.sparse) {
                sparseActivations.set(parsed.sparse);
            }
            if (parsed.meta) {
                modelMeta.set(parsed.meta);
            }

        } catch (e) {
            console.error('[WS] Error parsing packet:', e);
        }
    };
}
