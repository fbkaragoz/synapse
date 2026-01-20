#[cfg(test)]
mod tests {
    use crate::protocol::*;
    use wasm_bindgen_test::*;

    #[wasm_bindgen_test]
    fn test_parse_header_valid() {
        let mut data = vec![0u8; 32];
        data[0..4].copy_from_slice(&0x574C464Eu32.to_le_bytes());
        data[4..6].copy_from_slice(&1u16.to_le_bytes());
        data[6..8].copy_from_slice(&1u16.to_le_bytes());
        data[8..12].copy_from_slice(&0u32.to_le_bytes());
        data[12..20].copy_from_slice(&42u64.to_le_bytes());
        data[20..28].copy_from_slice(&123456u64.to_le_bytes());
        data[28..32].copy_from_slice(&16u32.to_le_bytes());

        let result = parse_header(&data);
        assert!(result.is_ok(), "Should parse valid header");

        let header = result.unwrap();
        assert_eq!(header.magic, 0x574C464E);
        assert_eq!(header.version, 1);
        assert_eq!(header.msg_type, NF_MSG_LAYER_SUMMARY_BATCH);
        assert_eq!(header.flags, 0);
        assert_eq!(header.seq, 42);
        assert_eq!(header.timestamp_ns, 123456);
        assert_eq!(header.payload_bytes, 16);
    }

    #[wasm_bindgen_test]
    fn test_parse_header_invalid_magic() {
        let mut data = vec![0u8; 32];
        data[0..4].copy_from_slice(&0xDEADBEEFu32.to_le_bytes());
        data[4..32].copy_from_slice(&[0u8; 28]);

        let result = parse_header(&data);
        assert!(result.is_err(), "Should reject invalid magic");
    }

    #[wasm_bindgen_test]
    fn test_parse_header_too_short() {
        let data = vec![0u8; 10];
        let result = parse_header(&data);
        assert!(result.is_err(), "Should reject too short buffer");
    }

    #[wasm_bindgen_test]
    fn test_parse_layer_summary_batch() {
        let mut data = vec![0u8; 48];
        data[0..4].copy_from_slice(&0x574C464Eu32.to_le_bytes());
        data[4..6].copy_from_slice(&1u16.to_le_bytes());
        data[6..8].copy_from_slice(&NF_MSG_LAYER_SUMMARY_BATCH.to_le_bytes());
        data[8..12].copy_from_slice(&NF_FLAG_FP32.to_le_bytes());
        data[12..20].copy_from_slice(&1u64.to_le_bytes());
        data[20..28].copy_from_slice(&0u64.to_le_bytes());
        data[28..32].copy_from_slice(&16u32.to_le_bytes());

        data[32..36].copy_from_slice(&2u32.to_le_bytes());
        data[36..40].copy_from_slice(&0u32.to_le_bytes());

        data[40..44].copy_from_slice(&1u32.to_le_bytes());
        data[44..48].copy_from_slice(&100u32.to_le_bytes());
        data.extend_from_slice(&1000f32.to_le_bytes());
        data.extend_from_slice(&0.95f32.to_le_bytes());

        let header = parse_header(&data).unwrap();
        let result = parse_payload(&data, &header);

        assert!(result.is_ok());
        let packet = result.unwrap();
        assert!(packet.summaries.is_some());
        assert!(packet.control.is_none());
        assert!(packet.sparse.is_none());

        let summaries = packet.summaries.unwrap();
        assert_eq!(summaries.len(), 2);
        assert_eq!(summaries[0].layer_id, 1);
        assert_eq!(summaries[0].neuron_count, 100);
        assert!((summaries[0].mean - 1000.0).abs() < 0.01);
        assert!((summaries[0].max - 0.95).abs() < 0.01);
    }

    #[wasm_bindgen_test]
    fn test_parse_sparse_activations() {
        let mut data = vec![0u8; 32];
        data[0..4].copy_from_slice(&0x574C464Eu32.to_le_bytes());
        data[4..6].copy_from_slice(&1u16.to_le_bytes());
        data[6..8].copy_from_slice(&NF_MSG_SPARSE_ACTIVATIONS.to_le_bytes());
        data[8..12].copy_from_slice(&NF_FLAG_FP32.to_le_bytes());
        data[12..20].copy_from_slice(&2u64.to_le_bytes());
        data[20..28].copy_from_slice(&0u64.to_le_bytes());
        data[28..32].copy_from_slice(&32u32.to_le_bytes());

        data.extend_from_slice(&5u32.to_le_bytes());
        data.extend_from_slice(&0u32.to_le_bytes());
        data.extend_from_slice(&0u32.to_le_bytes());
        data.extend_from_slice(&0u32.to_le_bytes());

        for i in 0..5 {
            data.extend_from_slice(&i.to_le_bytes());
            data.extend_from_slice(&(i as f32 * 0.1).to_le_bytes());
        }

        let header = parse_header(&data).unwrap();
        let result = parse_payload(&data, &header);

        assert!(result.is_ok());
        let packet = result.unwrap();
        assert!(packet.sparse.is_some());

        let sparse = packet.sparse.unwrap();
        assert_eq!(sparse.layer_id, 5);
        assert_eq!(sparse.count, 5);
        assert_eq!(sparse.indices.len(), 5);
        assert_eq!(sparse.values.len(), 5);

        for i in 0..5 {
            assert_eq!(sparse.indices[i], i);
            assert!((sparse.values[i] - (i as f32 * 0.1)).abs() < 0.01);
        }
    }

    #[wasm_bindgen_test]
    fn test_parse_control_packet() {
        let mut data = vec![0u8; 48];
        data[0..4].copy_from_slice(&0x574C464Eu32.to_le_bytes());
        data[4..6].copy_from_slice(&1u16.to_le_bytes());
        data[6..8].copy_from_slice(&NF_MSG_CONTROL.to_le_bytes());
        data[8..12].copy_from_slice(&0u32.to_le_bytes());
        data[12..20].copy_from_slice(&3u64.to_le_bytes());
        data[20..28].copy_from_slice(&0u64.to_le_bytes());
        data[28..32].copy_from_slice(&16u32.to_le_bytes());

        data.extend_from_slice(&NF_OP_SET_THRESHOLD.to_le_bytes());
        data.extend_from_slice(&0u32.to_le_bytes());
        data.extend_from_slice(&0.75f32.to_le_bytes());
        data.extend_from_slice(&0u32.to_le_bytes());

        let header = parse_header(&data).unwrap();
        let result = parse_payload(&data, &header);

        assert!(result.is_ok());
        let packet = result.unwrap();
        assert!(packet.control.is_some());

        let control = packet.control.unwrap();
        assert_eq!(control.opcode, NF_OP_SET_THRESHOLD);
        assert_eq!(control.value_u32, 0);
        assert!((control.value_f32 - 0.75).abs() < 0.01);
    }

    #[wasm_bindgen_test]
    fn test_parse_model_meta() {
        let mut data = vec![0u8; 60];
        data[0..4].copy_from_slice(&0x574C464Eu32.to_le_bytes());
        data[4..6].copy_from_slice(&1u16.to_le_bytes());
        data[6..8].copy_from_slice(&NF_MSG_MODEL_META.to_le_bytes());
        data[8..12].copy_from_slice(&0u32.to_le_bytes());
        data[12..20].copy_from_slice(&4u64.to_le_bytes());
        data[20..28].copy_from_slice(&0u64.to_le_bytes());
        data[28..32].copy_from_slice(&28u32.to_le_bytes());

        data.extend_from_slice(&2u32.to_le_bytes());

        data.extend_from_slice(&1u32.to_le_bytes());
        data.extend_from_slice(&1000u32.to_le_bytes());
        data.extend_from_slice(&32u16.to_le_bytes());
        data.extend_from_slice(&32u16.to_le_bytes());

        data.extend_from_slice(&2u32.to_le_bytes());
        data.extend_from_slice(&2000u32.to_le_bytes());
        data.extend_from_slice(&50u16.to_le_bytes());
        data.extend_from_slice(&40u16.to_le_bytes());

        let header = parse_header(&data).unwrap();
        let result = parse_payload(&data, &header);

        assert!(result.is_ok());
        let packet = result.unwrap();
        assert!(packet.meta.is_some());

        let meta = packet.meta.unwrap();
        assert_eq!(meta.total_layers, 2);
        assert_eq!(meta.layers.len(), 2);

        assert_eq!(meta.layers[0].layer_id, 1);
        assert_eq!(meta.layers[0].neuron_count, 1000);
        assert_eq!(meta.layers[0].recommended_width, 32);
        assert_eq!(meta.layers[0].recommended_height, 32);

        assert_eq!(meta.layers[1].layer_id, 2);
        assert_eq!(meta.layers[1].neuron_count, 2000);
        assert_eq!(meta.layers[1].recommended_width, 50);
        assert_eq!(meta.layers[1].recommended_height, 40);
    }

    #[wasm_bindgen_test]
    fn test_payload_too_short() {
        let mut data = vec![0u8; 48];
        data[0..4].copy_from_slice(&0x574C464Eu32.to_le_bytes());
        data[4..6].copy_from_slice(&1u16.to_le_bytes());
        data[6..8].copy_from_slice(&1u16.to_le_bytes());
        data[8..12].copy_from_slice(&0u32.to_le_bytes());
        data[12..20].copy_from_slice(&1u64.to_le_bytes());
        data[20..28].copy_from_slice(&0u64.to_le_bytes());
        data[28..32].copy_from_slice(&100u32.to_le_bytes());

        let header = parse_header(&data).unwrap();
        let result = parse_payload(&data, &header);

        assert!(result.is_err(), "Should reject payload too short");
    }

    #[wasm_bindgen_test]
    fn test_constants() {
        assert_eq!(NF_MAGIC, 0x574C464E);
        assert_eq!(NF_MSG_LAYER_SUMMARY_BATCH, 1);
        assert_eq!(NF_MSG_SPARSE_ACTIVATIONS, 2);
        assert_eq!(NF_MSG_CONTROL, 3);
        assert_eq!(NF_MSG_MODEL_META, 4);
        assert_eq!(NF_OP_SET_THRESHOLD, 1);
        assert_eq!(NF_OP_SET_ACCUMULATION_STEPS, 2);
        assert_eq!(NF_OP_SET_BROADCAST_INTERVAL, 3);
        assert_eq!(NF_OP_SET_MAX_SPARSE_POINTS, 4);
        assert_eq!(NF_FLAG_FP32, 2);
        assert_eq!(NF_FLAG_NONE, 0);
    }
}
