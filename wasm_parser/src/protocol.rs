use serde::Serialize;
use wasm_bindgen::prelude::*;

// Constants mirroring C++
pub const NF_MAGIC: u32 = 0x574C464E;
pub const NF_MSG_LAYER_SUMMARY_BATCH: u16 = 1;
pub const NF_MSG_SPARSE_ACTIVATIONS: u16 = 2;
pub const NF_MSG_CONTROL: u16 = 3;
pub const NF_MSG_MODEL_META: u16 = 4;
pub const NF_MSG_GRADIENT_BATCH: u16 = 5;
pub const NF_MSG_LAYER_SUMMARY_BATCH_V2: u16 = 6;
pub const NF_MSG_ATTENTION_PATTERN: u16 = 7;

pub const NF_FLAG_NONE: u32 = 0;
pub const NF_FLAG_FP16: u32 = 1u32 << 0;
pub const NF_FLAG_FP32: u32 = 1u32 << 1;
pub const NF_FLAG_COMPRESSED: u32 = 1u32 << 2;

pub const NF_OP_SET_THRESHOLD: u32 = 1;
pub const NF_OP_SET_ACCUMULATION_STEPS: u32 = 2;
pub const NF_OP_SET_BROADCAST_INTERVAL: u32 = 3;
pub const NF_OP_SET_MAX_SPARSE_POINTS: u32 = 4;

const HEADER_SIZE: usize = 32;
const ENTRY_SIZE_LAYER_SUMMARY: usize = 16;
const ENTRY_SIZE_SPARSE: usize = 8;
const HEADER_SIZE_LAYER_SUMMARY_BATCH: usize = 8;
const HEADER_SIZE_SPARSE: usize = 16;
const HEADER_SIZE_MODEL_META: usize = 4;
const HEADER_SIZE_LAYER_INFO: usize = 12;
const HEADER_SIZE_CONTROL: usize = 16;
const HEADER_SIZE_GRADIENT_BATCH: usize = 12;
const ENTRY_SIZE_GRADIENT_SUMMARY: usize = 36;
const HEADER_SIZE_LAYER_SUMMARY_BATCH_V2: usize = 8;
const ENTRY_SIZE_LAYER_SUMMARY_V2: usize = 64;
const HEADER_SIZE_ATTENTION_PATTERN: usize = 20;
const ENTRY_SIZE_ATTENTION: usize = 8;

#[derive(Debug, Clone, Serialize)]
pub enum ParseError {
  BufferTooShort,
  InvalidMagic,
  PayloadTooShort,
  PayloadTruncated,
  UnsupportedMessageType(u16),
  InvalidVersion(u16),
}

#[derive(Serialize, Clone)]
pub struct PacketHeader {
    pub magic: u32,
    pub version: u16,
    pub msg_type: u16,
    pub flags: u32,
    pub seq: u64,
    pub timestamp_ns: u64,
    pub payload_bytes: u32,
}

#[derive(Serialize)]
pub struct LayerSummary {
    pub layer_id: u32,
    pub neuron_count: u32,
    pub mean: f32,
    pub max: f32,
}

#[derive(Serialize)]
pub struct LayerSummaryV2 {
    pub layer_id: u32,
    pub neuron_count: u32,
    pub mean: f32,
    pub std: f32,
    pub min: f32,
    pub max: f32,
    pub l2_norm: f32,
    pub zero_ratio: f32,
    pub p5: f32,
    pub p25: f32,
    pub p75: f32,
    pub p95: f32,
    pub kurtosis: f32,
    pub skewness: f32,
    pub flags: u32,
}

#[derive(Serialize)]
pub struct SparseActivations {
    pub layer_id: u32,
    pub count: u32,
    pub indices: Vec<u32>,
    pub values: Vec<f32>,
}

#[derive(Serialize)]
pub struct ControlPacket {
    pub opcode: u32,
    pub value_u32: u32,
    pub value_f32: f32,
}

#[derive(Serialize)]
pub struct LayerInfo {
    pub layer_id: u32,
    pub neuron_count: u32,
    pub recommended_width: u16,
    pub recommended_height: u16,
}

#[derive(Serialize)]
pub struct ModelMeta {
    pub total_layers: u32,
    pub layers: Vec<LayerInfo>,
}

#[derive(Serialize)]
pub struct GradientSummary {
    pub layer_id: u32,
    pub param_count: u32,
    pub grad_mean: f32,
    pub grad_std: f32,
    pub grad_min: f32,
    pub grad_max: f32,
    pub grad_l2_norm: f32,
    pub weight_l2_norm: f32,
    pub grad_to_weight: f32,
}

#[derive(Serialize)]
pub struct GradientBatch {
    pub training_step: u32,
    pub global_grad_norm: f32,
    pub gradients: Vec<GradientSummary>,
}

#[derive(Serialize)]
pub struct AttentionEntry {
    pub src_idx: u16,
    pub tgt_idx: u16,
    pub weight: f32,
}

#[derive(Serialize)]
pub struct AttentionPattern {
    pub layer_id: u32,
    pub head_id: u32,
    pub seq_len: u16,
    pub tgt_len: u16,
    pub mode: u8,
    pub entries: Vec<AttentionEntry>,
}

#[derive(Serialize)]
pub struct ParsedPacket {
    pub header: PacketHeader,
    pub summaries: Option<Vec<LayerSummary>>,
    pub v2_summaries: Option<Vec<LayerSummaryV2>>,
    pub sparse: Option<SparseActivations>,
    pub control: Option<ControlPacket>,
    pub meta: Option<ModelMeta>,
    pub gradients: Option<GradientBatch>,
    pub attention: Option<AttentionPattern>,
}

pub fn parse_header(data: &[u8]) -> Result<PacketHeader, ParseError> {
    if data.len() < HEADER_SIZE {
        return Err(ParseError::BufferTooShort);
    }

    let magic = u32::from_le_bytes(
        data[0..4]
            .try_into()
            .map_err(|_| ParseError::BufferTooShort)?,
    );

    if magic != NF_MAGIC {
        return Err(ParseError::InvalidMagic);
    }

    let version = u16::from_le_bytes(
        data[4..6]
            .try_into()
            .map_err(|_| ParseError::BufferTooShort)?,
    );

    if version != 1 {
        return Err(ParseError::InvalidVersion(version));
    }

    let msg_type = u16::from_le_bytes(
        data[6..8]
            .try_into()
            .map_err(|_| ParseError::BufferTooShort)?,
    );

    let flags = u32::from_le_bytes(
        data[8..12]
            .try_into()
            .map_err(|_| ParseError::BufferTooShort)?,
    );

    let seq = u64::from_le_bytes(
        data[12..20]
            .try_into()
            .map_err(|_| ParseError::BufferTooShort)?,
    );

    let timestamp_ns = u64::from_le_bytes(
        data[20..28]
            .try_into()
            .map_err(|_| ParseError::BufferTooShort)?,
    );

    let payload_bytes = u32::from_le_bytes(
        data[28..32]
            .try_into()
            .map_err(|_| ParseError::BufferTooShort)?,
    );

    Ok(PacketHeader {
        magic,
        version,
        msg_type,
        flags,
        seq,
        timestamp_ns,
        payload_bytes,
    })
}

pub fn parse_payload(data: &[u8], header: &PacketHeader) -> Result<ParsedPacket, ParseError> {
    let payload_offset = HEADER_SIZE;
    let expected_len = payload_offset + header.payload_bytes as usize;

    if data.len() < expected_len {
        return Err(ParseError::PayloadTooShort);
    }

    let payload = &data[payload_offset..expected_len];
    let mut summaries = None;
    let mut v2_summaries = None;
    let mut sparse = None;
    let mut control = None;
    let mut meta = None;
    let mut gradients = None;
    let mut attention = None;

    match header.msg_type {
        NF_MSG_LAYER_SUMMARY_BATCH => {
            if payload.len() < HEADER_SIZE_LAYER_SUMMARY_BATCH {
                return Err(ParseError::PayloadTooShort);
            }

            let count = u32::from_le_bytes(
                payload[0..4]
                    .try_into()
                    .map_err(|_| ParseError::PayloadTooShort)?,
            ) as usize;

            let entry_size = ENTRY_SIZE_LAYER_SUMMARY;
            let start = HEADER_SIZE_LAYER_SUMMARY_BATCH;

            if payload.len() < start + count * entry_size {
                return Err(ParseError::PayloadTruncated);
            }

            let mut list = Vec::with_capacity(count);

            for i in 0..count {
                let offset = start + i * entry_size;
                let chunk = &payload[offset..offset + entry_size];

                list.push(LayerSummary {
                    layer_id: u32::from_le_bytes(
                        chunk[0..4]
                            .try_into()
                            .map_err(|_| ParseError::PayloadTruncated)?,
                    ),
                    neuron_count: u32::from_le_bytes(
                        chunk[4..8]
                            .try_into()
                            .map_err(|_| ParseError::PayloadTruncated)?,
                    ),
                    mean: f32::from_le_bytes(
                        chunk[8..12]
                            .try_into()
                            .map_err(|_| ParseError::PayloadTruncated)?,
                    ),
                    max: f32::from_le_bytes(
                        chunk[12..16]
                            .try_into()
                            .map_err(|_| ParseError::PayloadTruncated)?,
                    ),
                });
            }

            summaries = Some(list);
        }
        NF_MSG_SPARSE_ACTIVATIONS => {
            if payload.len() < HEADER_SIZE_SPARSE {
                return Err(ParseError::PayloadTooShort);
            }

            let layer_id = u32::from_le_bytes(
                payload[0..4]
                    .try_into()
                    .map_err(|_| ParseError::PayloadTooShort)?,
            );

            let count = u32::from_le_bytes(
                payload[4..8]
                    .try_into()
                    .map_err(|_| ParseError::PayloadTooShort)?,
            ) as usize;

            let entry_size = ENTRY_SIZE_SPARSE;
            let start = HEADER_SIZE_SPARSE;

            if payload.len() < start + count * entry_size {
                return Err(ParseError::PayloadTruncated);
            }

            let mut indices = Vec::with_capacity(count);
            let mut values = Vec::with_capacity(count);

            for i in 0..count {
                let offset = start + i * entry_size;
                let chunk = &payload[offset..offset + entry_size];

                indices.push(u32::from_le_bytes(
                    chunk[0..4]
                        .try_into()
                        .map_err(|_| ParseError::PayloadTruncated)?,
                ));
                values.push(f32::from_le_bytes(
                    chunk[4..8]
                        .try_into()
                        .map_err(|_| ParseError::PayloadTruncated)?,
                ));
            }

            sparse = Some(SparseActivations {
                layer_id,
                count: indices.len() as u32,
                indices,
                values,
            });
        }
        NF_MSG_CONTROL => {
            if payload.len() < HEADER_SIZE_CONTROL {
                return Err(ParseError::PayloadTooShort);
            }

            let opcode = u32::from_le_bytes(
                payload[0..4]
                    .try_into()
                    .map_err(|_| ParseError::PayloadTooShort)?,
            );
            let val_u32 = u32::from_le_bytes(
                payload[4..8]
                    .try_into()
                    .map_err(|_| ParseError::PayloadTooShort)?,
            );
            let val_f32 = f32::from_le_bytes(
                payload[8..12]
                    .try_into()
                    .map_err(|_| ParseError::PayloadTooShort)?,
            );

            control = Some(ControlPacket {
                opcode,
                value_u32: val_u32,
                value_f32: val_f32,
            });
        }
        NF_MSG_MODEL_META => {
            if payload.len() < HEADER_SIZE_MODEL_META {
                return Err(ParseError::PayloadTooShort);
            }

            let total_layers = u32::from_le_bytes(
                payload[0..4]
                    .try_into()
                    .map_err(|_| ParseError::PayloadTooShort)?,
            ) as usize;

            let entry_size = HEADER_SIZE_LAYER_INFO;
            let start = HEADER_SIZE_MODEL_META;

            if payload.len() < start + total_layers * entry_size {
                return Err(ParseError::PayloadTruncated);
            }

            let mut layers = Vec::with_capacity(total_layers);

            for i in 0..total_layers {
                let offset = start + i * entry_size;
                let chunk = &payload[offset..offset + entry_size];

                layers.push(LayerInfo {
                    layer_id: u32::from_le_bytes(
                        chunk[0..4]
                            .try_into()
                            .map_err(|_| ParseError::PayloadTruncated)?,
                    ),
                    neuron_count: u32::from_le_bytes(
                        chunk[4..8]
                            .try_into()
                            .map_err(|_| ParseError::PayloadTruncated)?,
                    ),
                    recommended_width: u16::from_le_bytes(
                        chunk[8..10]
                            .try_into()
                            .map_err(|_| ParseError::PayloadTruncated)?,
                    ),
                    recommended_height: u16::from_le_bytes(
                        chunk[10..12]
                            .try_into()
                            .map_err(|_| ParseError::PayloadTruncated)?,
                    ),
                });
            }

            meta = Some(ModelMeta {
                total_layers: layers.len() as u32,
                layers,
            });
        }
        NF_MSG_GRADIENT_BATCH => {
            if payload.len() < HEADER_SIZE_GRADIENT_BATCH {
                return Err(ParseError::PayloadTooShort);
            }

            let count = u32::from_le_bytes(
                payload[0..4]
                    .try_into()
                    .map_err(|_| ParseError::PayloadTooShort)?,
            ) as usize;
            let training_step = u32::from_le_bytes(
                payload[4..8]
                    .try_into()
                    .map_err(|_| ParseError::PayloadTooShort)?,
            );
            let global_grad_norm = f32::from_le_bytes(
                payload[8..12]
                    .try_into()
                    .map_err(|_| ParseError::PayloadTooShort)?,
            );

            let start = HEADER_SIZE_GRADIENT_BATCH;
            let entry_size = ENTRY_SIZE_GRADIENT_SUMMARY;

            if payload.len() < start + count * entry_size {
                return Err(ParseError::PayloadTruncated);
            }

            let mut grad_list = Vec::with_capacity(count);

            for i in 0..count {
                let offset = start + i * entry_size;
                let chunk = &payload[offset..offset + entry_size];

                grad_list.push(GradientSummary {
                    layer_id: u32::from_le_bytes(
                        chunk[0..4]
                            .try_into()
                            .map_err(|_| ParseError::PayloadTruncated)?,
                    ),
                    param_count: u32::from_le_bytes(
                        chunk[4..8]
                            .try_into()
                            .map_err(|_| ParseError::PayloadTruncated)?,
                    ),
                    grad_mean: f32::from_le_bytes(
                        chunk[8..12]
                            .try_into()
                            .map_err(|_| ParseError::PayloadTruncated)?,
                    ),
                    grad_std: f32::from_le_bytes(
                        chunk[12..16]
                            .try_into()
                            .map_err(|_| ParseError::PayloadTruncated)?,
                    ),
                    grad_min: f32::from_le_bytes(
                        chunk[16..20]
                            .try_into()
                            .map_err(|_| ParseError::PayloadTruncated)?,
                    ),
                    grad_max: f32::from_le_bytes(
                        chunk[20..24]
                            .try_into()
                            .map_err(|_| ParseError::PayloadTruncated)?,
                    ),
                    grad_l2_norm: f32::from_le_bytes(
                        chunk[24..28]
                            .try_into()
                            .map_err(|_| ParseError::PayloadTruncated)?,
                    ),
                    weight_l2_norm: f32::from_le_bytes(
                        chunk[28..32]
                            .try_into()
                            .map_err(|_| ParseError::PayloadTruncated)?,
                    ),
                    grad_to_weight: f32::from_le_bytes(
                        chunk[32..36]
                            .try_into()
                            .map_err(|_| ParseError::PayloadTruncated)?,
                    ),
                });
            }

            gradients = Some(GradientBatch {
                training_step,
                global_grad_norm,
                gradients: grad_list,
            });
        }
<<<<<<< HEAD
        NF_MSG_LAYER_SUMMARY_BATCH_V2 => {
            if payload.len() < HEADER_SIZE_LAYER_SUMMARY_BATCH_V2 {
                return Err(ParseError::PayloadTooShort);
            }

            let count = u32::from_le_bytes(
                payload[0..4]
                    .try_into()
                    .map_err(|_| ParseError::PayloadTooShort)?,
            ) as usize;
            let entry_size = ENTRY_SIZE_LAYER_SUMMARY_V2;
            let start = HEADER_SIZE_LAYER_SUMMARY_BATCH_V2;

            if payload.len() < start + count * entry_size {
                return Err(ParseError::PayloadTruncated);
            }

            let mut v2_list = Vec::with_capacity(count);

            for i in 0..count {
                let offset = start + i * entry_size;
                let chunk = &payload[offset..offset + entry_size];

                v2_list.push(LayerSummaryV2 {
                    layer_id: u32::from_le_bytes(
                        chunk[0..4]
                            .try_into()
                            .map_err(|_| ParseError::PayloadTruncated)?,
                    ),
                    neuron_count: u32::from_le_bytes(
                        chunk[4..8]
                            .try_into()
                            .map_err(|_| ParseError::PayloadTruncated)?,
                    ),
                    mean: f32::from_le_bytes(
                        chunk[8..12]
                            .try_into()
                            .map_err(|_| ParseError::PayloadTruncated)?,
                    ),
                    std: f32::from_le_bytes(
                        chunk[12..16]
                            .try_into()
                            .map_err(|_| ParseError::PayloadTruncated)?,
                    ),
                    min: f32::from_le_bytes(
                        chunk[16..20]
                            .try_into()
                            .map_err(|_| ParseError::PayloadTruncated)?,
                    ),
                    max: f32::from_le_bytes(
                        chunk[20..24]
                            .try_into()
                            .map_err(|_| ParseError::PayloadTruncated)?,
                    ),
                    l2_norm: f32::from_le_bytes(
                        chunk[24..28]
                            .try_into()
                            .map_err(|_| ParseError::PayloadTruncated)?,
                    ),
                    zero_ratio: f32::from_le_bytes(
                        chunk[28..32]
                            .try_into()
                            .map_err(|_| ParseError::PayloadTruncated)?,
                    ),
                    p5: f32::from_le_bytes(
                        chunk[32..36]
                            .try_into()
                            .map_err(|_| ParseError::PayloadTruncated)?,
                    ),
                    p25: f32::from_le_bytes(
                        chunk[36..40]
                            .try_into()
                            .map_err(|_| ParseError::PayloadTruncated)?,
                    ),
                    p75: f32::from_le_bytes(
                        chunk[40..44]
                            .try_into()
                            .map_err(|_| ParseError::PayloadTruncated)?,
                    ),
                    p95: f32::from_le_bytes(
                        chunk[44..48]
                            .try_into()
                            .map_err(|_| ParseError::PayloadTruncated)?,
                    ),
                    kurtosis: f32::from_le_bytes(
                        chunk[48..52]
                            .try_into()
                            .map_err(|_| ParseError::PayloadTruncated)?,
                    ),
                    skewness: f32::from_le_bytes(
                        chunk[52..56]
                            .try_into()
                            .map_err(|_| ParseError::PayloadTruncated)?,
                    ),
                    flags: u32::from_le_bytes(
                        chunk[56..60]
                            .try_into()
                            .map_err(|_| ParseError::PayloadTruncated)?,
                    ),
                });
            }

            v2_summaries = Some(v2_list);
        }
        NF_MSG_ATTENTION_PATTERN => {
            if payload.len() < HEADER_SIZE_ATTENTION_PATTERN {
                return Err(ParseError::PayloadTooShort);
            }

            let layer_id = u32::from_le_bytes(
                payload[0..4]
                    .try_into()
                    .map_err(|_| ParseError::PayloadTooShort)?,
            );
            let head_id = u32::from_le_bytes(
                payload[4..8]
                    .try_into()
                    .map_err(|_| ParseError::PayloadTooShort)?,
            );
            let seq_len = u16::from_le_bytes(
                payload[8..10]
                    .try_into()
                    .map_err(|_| ParseError::PayloadTooShort)?,
            );
            let tgt_len = u16::from_le_bytes(
                payload[10..12]
                    .try_into()
                    .map_err(|_| ParseError::PayloadTooShort)?,
            );
            let mode = payload[12];
            let entry_count = u16::from_le_bytes(
                payload[14..16]
                    .try_into()
                    .map_err(|_| ParseError::PayloadTooShort)?,
            ) as usize;

            let start = HEADER_SIZE_ATTENTION_PATTERN;
            let entry_size = ENTRY_SIZE_ATTENTION;

            if payload.len() < start + entry_count * entry_size {
                return Err(ParseError::PayloadTruncated);
            }

            let mut attn_entries = Vec::with_capacity(entry_count);

            for i in 0..entry_count {
                let offset = start + i * entry_size;
                let chunk = &payload[offset..offset + entry_size];

                attn_entries.push(AttentionEntry {
                    src_idx: u16::from_le_bytes(
                        chunk[0..2]
                            .try_into()
                            .map_err(|_| ParseError::PayloadTruncated)?,
                    ),
                    tgt_idx: u16::from_le_bytes(
                        chunk[2..4]
                            .try_into()
                            .map_err(|_| ParseError::PayloadTruncated)?,
                    ),
                    weight: f32::from_le_bytes(
                        chunk[4..8]
                            .try_into()
                            .map_err(|_| ParseError::PayloadTruncated)?,
                    ),
                });
            }

            attention = Some(AttentionPattern {
                layer_id,
                head_id,
                seq_len,
                tgt_len,
                mode,
                entries: attn_entries,
            });
        }
        _ => return Err(ParseError::UnsupportedMessageType(header.msg_type)),
    }

    Ok(ParsedPacket {
        header: header.clone(),
        summaries,
        v2_summaries,
        sparse,
        control,
        meta,
        gradients,
        attention,
    })
}
