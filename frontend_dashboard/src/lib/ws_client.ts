import { writable } from 'svelte/store';
import type { WasmParser, ParsedPacket, LayerSummary, SparseActivationData, ModelMeta } from './types';
import { Logger, safeTry } from './logger';

export const isConnected = writable(false);
export const packetCount = writable(0);

export const layerSummaries = writable<LayerSummary[]>([]);
export const sparseActivations = writable<SparseActivationData | null>(null);
export const modelMeta = writable<ModelMeta | null>(null);

export const configThreshold = writable(0.5);
export const configAccumSteps = writable(0);
export const configBroadcastInterval = writable(0);
export const configMaxSparse = writable(0);


const OP_SET_THRESHOLD = 1;
const OP_SET_ACCUMULATION_STEPS = 2;
const OP_SET_BROADCAST_INTERVAL = 3;
const OP_SET_MAX_SPARSE_POINTS = 4;

const DEFAULT_WS_URL = 'ws://localhost:9000';

let socket: WebSocket | null = null;
let wasmParser: WasmParser | null = null;

const layerSummaryById = new Map<number, LayerSummary>();

function resetState(): void {
    layerSummaryById.clear();
    layerSummaries.set([]);
    sparseActivations.set(null);
    modelMeta.set(null);
    packetCount.set(0);
    Logger.info('ws_client', 'State reset');
}


export function setWasmParser(parser: WasmParser | null): void {
    wasmParser = parser;
    Logger.info('ws_client', `WASM parser ${parser ? 'loaded' : 'unloaded'}`);
}

export function sendControl(opcode: number, valU32: number, valF32: number): void {
    if (!socket || socket.readyState !== WebSocket.OPEN) {
        Logger.warn('ws_client', 'Cannot send control: socket not connected');
        return;
    }

    try {
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
        Logger.debug('ws_client', `Control sent: opcode=${opcode}, val_u32=${valU32}, val_f32=${valF32}`);
    } catch (error) {
        Logger.error('ws_client', 'Failed to send control packet', error);
    }
}

export function connect(url: string = DEFAULT_WS_URL): void {
    if (socket) {
        Logger.info('ws_client', 'Closing existing connection');
        safeTry('ws_client', 'Close socket', () => socket?.close());
        socket = null;
    }

    resetState();

    socket = safeTry('ws_client', 'Create WebSocket', () => new WebSocket(url), null);

    if (!socket) {
        Logger.error('ws_client', 'Failed to create WebSocket');
        return;
    }

    socket.binaryType = 'arraybuffer';

    socket.onopen = () => {
        Logger.info('ws_client', `Connected to ${url}`);
        isConnected.set(true);
    };

    socket.onclose = (event: CloseEvent) => {
        Logger.info('ws_client', `Disconnected: code=${event.code}, reason=${event.reason}`);
        isConnected.set(false);
        resetState();
    };

    socket.onerror = (event: Event) => {
        Logger.error('ws_client', 'WebSocket error', event);
        isConnected.set(false);
    };

    socket.onmessage = async (event: MessageEvent): Promise<void> => {
        const parser = wasmParser;
        if (!parser) {
            Logger.warn('ws_client', 'WASM parser not ready, skipping packet');
            return;
        }

        const buffer = new Uint8Array(event.data as ArrayBuffer);
        packetCount.update((n) => n + 1);

        safeTry('ws_client', 'Parse packet', () => {
            const parsed: ParsedPacket = parser.parse_packet(buffer);
            Logger.debug('ws_client', `Parsed packet: type=${parsed.header.msg_type}`);

            if (parsed.control) {
                const c = parsed.control;
                Logger.debug('ws_client', `Control packet: opcode=${c.opcode}`);

                switch (c.opcode) {
                    case OP_SET_THRESHOLD:
                        configThreshold.set(c.value_f32);
                        break;
                    case OP_SET_ACCUMULATION_STEPS:
                        configAccumSteps.set(c.value_u32);
                        break;
                    case OP_SET_BROADCAST_INTERVAL:
                        configBroadcastInterval.set(c.value_u32);
                        break;
                    case OP_SET_MAX_SPARSE_POINTS:
                        configMaxSparse.set(c.value_u32);
                        break;
                }
            }

            if (parsed.summaries) {
                for (const s of parsed.summaries) {
                    layerSummaryById.set(s.layer_id, s);
                }
                layerSummaries.set(Array.from(layerSummaryById.values()).sort((a, b) => a.layer_id - b.layer_id));
                Logger.debug('ws_client', `Updated ${parsed.summaries.length} layer summaries`);
            }

            if (parsed.sparse) {
                sparseActivations.set(parsed.sparse);
                Logger.debug('ws_client', `Updated sparse activation: layer=${parsed.sparse.layer_id}, count=${parsed.sparse.count}`);
            }

            if (parsed.meta) {
                modelMeta.set(parsed.meta);
                Logger.info('ws_client', `Received model metadata: ${parsed.meta.total_layers} layers`);
            }
        });
    };
}
