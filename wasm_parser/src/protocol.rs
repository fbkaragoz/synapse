use wasm_bindgen::prelude::*;
use serde::{Serialize, Deserialize};

// Constants mirroring C++
pub const NF_MAGIC: u32 = 0x574C464E; // "NFLW"
pub const NF_MSG_LAYER_SUMMARY_BATCH: u16 = 1;
pub const NF_MSG_SPARSE_ACTIVATIONS: u16 = 2;
pub const NF_MSG_CONTROL: u16 = 3;
pub const NF_MSG_MODEL_META: u16 = 4;

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
pub struct ParsedPacket {
    pub header: PacketHeader,
    pub summaries: Option<Vec<LayerSummary>>,
    pub sparse: Option<SparseActivations>,
    pub control: Option<ControlPacket>,
    pub meta: Option<ModelMeta>,
}

pub fn parse_header(data: &[u8]) -> Result<PacketHeader, JsValue> {
    if data.len() < 32 {
        return Err(JsValue::from_str("Buffer too short for header"));
    }

    // Little-endian parsing
    let magic = u32::from_le_bytes(data[0..4].try_into().unwrap());
    if magic != NF_MAGIC {
        return Err(JsValue::from_str("Invalid magic number"));
    }

    let version = u16::from_le_bytes(data[4..6].try_into().unwrap());
    let msg_type = u16::from_le_bytes(data[6..8].try_into().unwrap());
    let flags = u32::from_le_bytes(data[8..12].try_into().unwrap());
    let seq = u64::from_le_bytes(data[12..20].try_into().unwrap());
    let timestamp_ns = u64::from_le_bytes(data[20..28].try_into().unwrap());
    let payload_bytes = u32::from_le_bytes(data[28..32].try_into().unwrap());

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

pub fn parse_payload(data: &[u8], header: &PacketHeader) -> Result<ParsedPacket, JsValue> {
    // ... setup ...
    let payload_offset = 32;
    let expected_len = payload_offset + header.payload_bytes as usize;
    if data.len() < expected_len {
        return Err(JsValue::from_str("Buffer too short for payload"));
    }

    let payload = &data[payload_offset..expected_len];
    let mut summaries = None;
    let mut sparse = None;
    let mut control = None;
    let mut meta = None;

    if header.msg_type == NF_MSG_LAYER_SUMMARY_BATCH {
        // ... parsing logic ...
        if payload.len() < 8 {
             return Err(JsValue::from_str("Payload too short for Batch Header"));
        }
        let count = u32::from_le_bytes(payload[0..4].try_into().unwrap());
        
        let mut list = Vec::with_capacity(count as usize);
        let entry_size = 16;
        let start = 8;
        
        for i in 0..count {
            let offset = start + (i as usize * entry_size);
            if offset + entry_size > payload.len() {
                break;
            }
            let chunk = &payload[offset..offset+entry_size];
            list.push(LayerSummary {
                layer_id: u32::from_le_bytes(chunk[0..4].try_into().unwrap()),
                neuron_count: u32::from_le_bytes(chunk[4..8].try_into().unwrap()),
                mean: f32::from_le_bytes(chunk[8..12].try_into().unwrap()),
                max: f32::from_le_bytes(chunk[12..16].try_into().unwrap()),
            });
        }
        summaries = Some(list);
    } else if header.msg_type == NF_MSG_SPARSE_ACTIVATIONS {
         if payload.len() < 16 { return Err(JsValue::from_str("Payload too short")); }
         let layer_id = u32::from_le_bytes(payload[0..4].try_into().unwrap());
         let count = u32::from_le_bytes(payload[4..8].try_into().unwrap());
         
         let entry_size = 8;
         let start = 16;
         let mut indices = Vec::with_capacity(count as usize);
         let mut values = Vec::with_capacity(count as usize);
         
         for i in 0..count {
             let offset = start + (i as usize * entry_size);
             if offset + entry_size > payload.len() {
                 break;
             }
             let chunk = &payload[offset..offset+entry_size];
             indices.push(u32::from_le_bytes(chunk[0..4].try_into().unwrap()));
             values.push(f32::from_le_bytes(chunk[4..8].try_into().unwrap()));
         }
         sparse = Some(SparseActivations { layer_id, count: indices.len() as u32, indices, values });
    } else if header.msg_type == NF_MSG_CONTROL {
        if payload.len() < 16 { return Err(JsValue::from_str("Control payload too short")); }
        let opcode = u32::from_le_bytes(payload[0..4].try_into().unwrap());
        let val_u32 = u32::from_le_bytes(payload[4..8].try_into().unwrap());
        let val_f32 = f32::from_le_bytes(payload[8..12].try_into().unwrap());
        control = Some(ControlPacket { opcode, value_u32: val_u32, value_f32: val_f32 });
    } else if header.msg_type == NF_MSG_MODEL_META {
        if payload.len() < 4 { return Err(JsValue::from_str("Meta payload too short")); }
        let total_layers = u32::from_le_bytes(payload[0..4].try_into().unwrap());
        
        let entry_size = 12; // 4 + 4 + 2 + 2
        let start = 4;
        let mut layers = Vec::with_capacity(total_layers as usize);
        
        for i in 0..total_layers {
            let offset = start + (i as usize * entry_size);
            if offset + entry_size > payload.len() {
                break;
            }
            let chunk = &payload[offset..offset+entry_size];
            layers.push(LayerInfo {
                layer_id: u32::from_le_bytes(chunk[0..4].try_into().unwrap()),
                neuron_count: u32::from_le_bytes(chunk[4..8].try_into().unwrap()),
                recommended_width: u16::from_le_bytes(chunk[8..10].try_into().unwrap()),
                recommended_height: u16::from_le_bytes(chunk[10..12].try_into().unwrap()),
            });
        }
        meta = Some(ModelMeta { total_layers: layers.len() as u32, layers });
    }

    Ok(ParsedPacket {
        header: header.clone(),
        summaries,
        sparse,
        control,
        meta,
    })
}
