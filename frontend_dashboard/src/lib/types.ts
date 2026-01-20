export interface PacketHeader {
  magic: number;
  version: number;
  msg_type: number;
  flags: number;
  seq: number;
  timestamp_ns: number;
  payload_bytes: number;
}

export interface LayerSummary {
  layer_id: number;
  neuron_count: number;
  mean: number;
  max: number;
}

export interface SparseActivationData {
  layer_id: number;
  count: number;
  indices: number[];
  values: number[];
}

export interface ControlPacket {
  opcode: number;
  value_u32: number;
  value_f32: number;
}

export interface LayerInfo {
  layer_id: number;
  neuron_count: number;
  recommended_width: number;
  recommended_height: number;
}

export interface ModelMeta {
  total_layers: number;
  layers: LayerInfo[];
}

export interface ParsedPacket {
  header: PacketHeader;
  summaries?: LayerSummary[];
  sparse?: SparseActivationData;
  control?: ControlPacket;
  meta?: ModelMeta;
}

export interface WasmParser {
  init_panic_hook: () => void;
  parse_packet: (data: Uint8Array) => ParsedPacket;
}
