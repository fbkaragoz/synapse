import { describe, it, expect } from 'vitest';
import type {
  PacketHeader,
  LayerSummary,
  SparseActivationData,
  ControlPacket,
  LayerInfo,
  ModelMeta,
  ParsedPacket,
} from '$lib/types';

describe('types', () => {
  it('should have correct PacketHeader structure', () => {
    const header: PacketHeader = {
      magic: 0x574C464E,
      version: 1,
      msg_type: 1,
      flags: 0,
      seq: 0,
      timestamp_ns: 0,
      payload_bytes: 16
    };

    expect(header.magic).toBe(0x574C464E);
    expect(header.version).toBe(1);
    expect(header.msg_type).toBe(1);
    expect(header.flags).toBe(0);
    expect(header.seq).toBe(0);
    expect(header.timestamp_ns).toBe(0);
    expect(header.payload_bytes).toBe(16);
  });

  it('should have correct LayerSummary structure', () => {
    const summary: LayerSummary = {
      layer_id: 1,
      neuron_count: 100,
      mean: 0.5,
      max: 1.0
    };

    expect(summary.layer_id).toBe(1);
    expect(summary.neuron_count).toBe(100);
    expect(summary.mean).toBe(0.5);
    expect(summary.max).toBe(1.0);
  });

  it('should have correct SparseActivationData structure', () => {
    const sparse: SparseActivationData = {
      layer_id: 1,
      count: 3,
      indices: [0, 1, 2],
      values: [0.1, 0.2, 0.3]
    };

    expect(sparse.layer_id).toBe(1);
    expect(sparse.count).toBe(3);
    expect(sparse.indices).toEqual([0, 1, 2]);
    expect(sparse.values).toEqual([0.1, 0.2, 0.3]);
  });

  it('should have correct ControlPacket structure', () => {
    const control: ControlPacket = {
      opcode: 1,
      value_u32: 0,
      value_f32: 0.75
    };

    expect(control.opcode).toBe(1);
    expect(control.value_u32).toBe(0);
    expect(control.value_f32).toBe(0.75);
  });

  it('should have correct LayerInfo structure', () => {
    const info: LayerInfo = {
      layer_id: 1,
      neuron_count: 100,
      recommended_width: 10,
      recommended_height: 10
    };

    expect(info.layer_id).toBe(1);
    expect(info.neuron_count).toBe(100);
    expect(info.recommended_width).toBe(10);
    expect(info.recommended_height).toBe(10);
  });

  it('should have correct ModelMeta structure', () => {
    const meta: ModelMeta = {
      total_layers: 2,
      layers: [
        { layer_id: 1, neuron_count: 100, recommended_width: 10, recommended_height: 10 }
      ]
    };

    expect(meta.total_layers).toBe(2);
    expect(meta.layers).toHaveLength(1);
    expect(meta.layers[0].layer_id).toBe(1);
  });

  it('should have correct ParsedPacket structure', () => {
    const packet: ParsedPacket = {
      header: {
        magic: 0x574C464E,
        version: 1,
        msg_type: 1,
        flags: 0,
        seq: 0,
        timestamp_ns: 0,
        payload_bytes: 16
      },
      summaries: [{ layer_id: 1, neuron_count: 100, mean: 0.5, max: 1.0 }]
    };

    expect(packet.header.magic).toBe(0x574C464E);
    expect(packet.summaries).toBeDefined();
    expect(packet.summaries).toHaveLength(1);
  });

  it('should allow optional fields in ParsedPacket', () => {
    const packet1: ParsedPacket = {
      header: {
        magic: 0x574C464E,
        version: 1,
        msg_type: 1,
        flags: 0,
        seq: 0,
        timestamp_ns: 0,
        payload_bytes: 16
      }
    };

    const packet2: ParsedPacket = {
      header: {
        magic: 0x574C464E,
        version: 1,
        msg_type: 2,
        flags: 0,
        seq: 0,
        timestamp_ns: 0,
        payload_bytes: 32
      },
      sparse: { layer_id: 1, count: 2, indices: [0, 1], values: [0.1, 0.2] }
    };

    const packet3: ParsedPacket = {
      header: {
        magic: 0x574C464E,
        version: 1,
        msg_type: 3,
        flags: 0,
        seq: 0,
        timestamp_ns: 0,
        payload_bytes: 16
      },
      control: { opcode: 1, value_u32: 0, value_f32: 0.5 }
    };

    const packet4: ParsedPacket = {
      header: {
        magic: 0x574C464E,
        version: 1,
        msg_type: 4,
        flags: 0,
        seq: 0,
        timestamp_ns: 0,
        payload_bytes: 16
      },
      meta: { total_layers: 1, layers: [{ layer_id: 1, neuron_count: 100, recommended_width: 10, recommended_height: 10 }] }
    };

    expect(packet1.summaries).toBeUndefined();
    expect(packet1.sparse).toBeUndefined();
    expect(packet1.control).toBeUndefined();
    expect(packet1.meta).toBeUndefined();

    expect(packet2.summaries).toBeUndefined();
    expect(packet2.sparse).toBeDefined();
    expect(packet2.control).toBeUndefined();
    expect(packet2.meta).toBeUndefined();

    expect(packet3.summaries).toBeUndefined();
    expect(packet3.sparse).toBeUndefined();
    expect(packet3.control).toBeDefined();
    expect(packet3.meta).toBeUndefined();

    expect(packet4.summaries).toBeUndefined();
    expect(packet4.sparse).toBeUndefined();
    expect(packet4.control).toBeUndefined();
    expect(packet4.meta).toBeDefined();
  });
});
