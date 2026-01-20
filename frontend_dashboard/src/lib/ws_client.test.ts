import { describe, it, expect } from 'vitest';
import { get } from 'svelte/store';
import {
  isConnected,
  packetCount,
  layerSummaries,
  sparseActivations,
  modelMeta,
  configThreshold,
  configAccumSteps,
  configBroadcastInterval,
  configMaxSparse,
  setWasmParser,
  sendControl
} from './ws_client';
import type { LayerSummary, SparseActivationData, ModelMeta } from './types';

describe('ws_client stores', () => {
  it('should initialize with default values', () => {
    expect(get(isConnected)).toBe(false);
    expect(get(packetCount)).toBe(0);
    expect(get(layerSummaries)).toEqual([]);
    expect(get(sparseActivations)).toBe(null);
    expect(get(modelMeta)).toBe(null);
    expect(get(configThreshold)).toBe(0.5);
    expect(get(configAccumSteps)).toBe(0);
    expect(get(configBroadcastInterval)).toBe(0);
    expect(get(configMaxSparse)).toBe(0);
  });

  it('should update config threshold store', () => {
    configThreshold.set(0.75);
    expect(get(configThreshold)).toBe(0.75);
  });

  it('should update layer summaries store', () => {
    const summaries: LayerSummary[] = [
      { layer_id: 1, neuron_count: 100, mean: 0.5, max: 1.0 },
      { layer_id: 2, neuron_count: 200, mean: 0.3, max: 0.8 }
    ];
    layerSummaries.set(summaries);
    expect(get(layerSummaries)).toEqual(summaries);
  });

  it('should update sparse activations store', () => {
    const sparse: SparseActivationData = {
      layer_id: 1,
      count: 3,
      indices: [0, 1, 2],
      values: [0.1, 0.2, 0.3]
    };
    sparseActivations.set(sparse);
    expect(get(sparseActivations)).toEqual(sparse);
  });

  it('should update model meta store', () => {
    const meta: ModelMeta = {
      total_layers: 2,
      layers: [
        { layer_id: 1, neuron_count: 100, recommended_width: 10, recommended_height: 10 },
        { layer_id: 2, neuron_count: 200, recommended_width: 20, recommended_height: 10 }
      ]
    };
    modelMeta.set(meta);
    expect(get(modelMeta)).toEqual(meta);
  });

  it('should update connection state', () => {
    isConnected.set(true);
    expect(get(isConnected)).toBe(true);

    isConnected.set(false);
    expect(get(isConnected)).toBe(false);
  });

  it('should update packet count', () => {
    packetCount.set(42);
    expect(get(packetCount)).toBe(42);
  });

  it('should not crash when setting wasm parser to null', () => {
    expect(() => setWasmParser(null)).not.toThrow();
  });

  it('should send control without error when parser not set', () => {
    expect(() => sendControl(1, 0, 0.5)).not.toThrow();
  });
});
