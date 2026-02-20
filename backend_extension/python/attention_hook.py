"""
Attention Capture for Transformer Models

This module provides utilities to capture attention weights from
transformer models during inference/training and stream them to
the Synapse dashboard.

Usage:
    import torch
    import neural_probe
    from attention_hook import AttentionCapture

    # Start the probe server
    neural_probe.start_server("localhost", 9000)

    # Create capture for attention layers
    capture = AttentionCapture(model)
    capture.register_hooks()

    # Inference/training loop - attention captured automatically
    output = model(input_ids)
"""

import re
from typing import List, Optional, Dict
import torch
import torch.nn as nn

try:
    import neural_probe
except ImportError:
    neural_probe = None
    print("Warning: neural_probe not installed. Attention capture disabled.")


class AttentionCapture:
    def __init__(
        self,
        model: nn.Module,
        layers: Optional[List[int]] = None,
        heads: Optional[List[int]] = None,
        mode: int = 1,  # NF_ATTENTION_TOP_K
        max_entries: int = 1000,
        threshold: float = 0.0,
    ):
        """
        Initialize attention capture for a transformer model.

        Args:
            model: PyTorch transformer model
            layers: List of layer indices to capture (None = all)
            heads: List of head indices to capture (None = all)
            mode: Attention capture mode
                0 = FULL (all weights)
                1 = TOP_K (top k per layer)
                2 = THRESHOLD (weights above threshold)
                3 = BAND (local attention window)
            max_entries: Maximum entries per attention pattern
            threshold: Threshold for THRESHOLD mode
        """
        self.model = model
        self.target_layers = set(layers) if layers else None
        self.target_heads = set(heads) if heads else None
        self.mode = mode
        self.max_entries = max_entries
        self.threshold = threshold
        self.handles = []
        self._attention_module_map: Dict[str, tuple] = {}
        self._build_attention_map()

    def _build_attention_map(self):
        """Build mapping from module names to (layer_id, head_id) tuples."""
        layer_pattern = re.compile(r"layers?\.(\d+)")
        head_pattern = re.compile(r"heads?\.(\d+)|(\d+)\.attention")

        for name, module in self.model.named_modules():
            if "attention" in name.lower() or "attn" in name.lower():
                layer_match = layer_pattern.search(name)
                layer_id = int(layer_match.group(1)) if layer_match else 0

                head_match = head_pattern.search(name)
                head_id = (
                    int(head_match.group(1) or head_match.group(2)) if head_match else 0
                )

                self._attention_module_map[name] = (layer_id, head_id)

    def _should_capture(self, layer_id: int, head_id: int) -> bool:
        """Check if this layer/head should be captured."""
        if self.target_layers is not None and layer_id not in self.target_layers:
            return False
        if self.target_heads is not None and head_id not in self.target_heads:
            return False
        return True

    def _make_hook(self, name: str, layer_id: int, head_id: int):
        """Create a forward hook for attention weights."""

        def hook(module, input, output):
            if neural_probe is None:
                return

            attn_weights = None

            # Try different common patterns for attention weights
            if isinstance(output, tuple) and len(output) >= 2:
                if isinstance(output[1], torch.Tensor):
                    attn_weights = output[1]
                elif isinstance(output[1], dict) and "attention_weights" in output[1]:
                    attn_weights = output[1]["attention_weights"]

            if hasattr(module, "_attention_weights"):
                attn_weights = module._attention_weights

            if attn_weights is None:
                return

            if not self._should_capture(layer_id, head_id):
                return

            # Move to CPU and convert to numpy
            attn_cpu = attn_weights.detach().cpu().float()

            # Handle different shapes
            if attn_cpu.ndim == 4:
                attn_cpu = attn_cpu[0, head_id % attn_cpu.shape[1]]
            elif attn_cpu.ndim == 3:
                attn_cpu = attn_cpu[0]

            neural_probe.log_attention(
                layer_id,
                head_id,
                attn_cpu.numpy(),
                self.mode,
                self.max_entries,
                self.threshold,
            )

        return hook

    def register_hooks(self):
        """Register forward hooks on attention modules."""
        self.remove_hooks()

        for name, module in self.model.named_modules():
            if name not in self._attention_module_map:
                continue

            layer_id, head_id = self._attention_module_map[name]

            if not self._should_capture(layer_id, head_id):
                continue

            handle = module.register_forward_hook(
                self._make_hook(name, layer_id, head_id)
            )
            self.handles.append(handle)

        print(f"[AttentionCapture] Registered {len(self.handles)} attention hooks")

    def remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def set_mode(self, mode: int):
        """Set attention capture mode."""
        self.mode = mode

    def set_max_entries(self, max_entries: int):
        """Set maximum entries per pattern."""
        self.max_entries = max_entries

    def set_threshold(self, threshold: float):
        """Set threshold for THRESHOLD mode."""
        self.threshold = threshold


class HuggingFaceAttentionCapture(AttentionCapture):
    """
    Specialized capture for HuggingFace transformers.
    """

    def _build_attention_map(self):
        """Build mapping for HuggingFace model structure."""
        for name, module in self.model.named_modules():
            if "attention" in name.lower() or "attn" in name.lower():
                # HuggingFace typically uses: encoder.layer.N.attention
                import re

                layer_match = re.search(r"layer\.(\d+)", name)
                layer_id = int(layer_match.group(1)) if layer_match else 0

                head_match = re.search(r"head\.(\d+)", name)
                head_id = int(head_match.group(1)) if head_match else 0

                self._attention_module_map[name] = (layer_id, head_id)

    def _make_hook(self, name: str, layer_id: int, head_id: int):
        """Create hook for HuggingFace attention output."""

        def hook(module, input, output):
            if neural_probe is None:
                return

            attn_weights = None

            # HuggingFace pattern: output is tuple, attention in output[1]
            if isinstance(output, tuple) and len(output) >= 2:
                if isinstance(output[1], torch.Tensor):
                    attn_weights = output[1]

            if attn_weights is None:
                return

            if not self._should_capture(layer_id, head_id):
                return

            attn_cpu = attn_weights.detach().cpu().float()

            # Shape: [batch, heads, seq, seq] -> [seq, seq]
            if attn_cpu.ndim == 4:
                attn_cpu = attn_cpu[0, 0]
            elif attn_cpu.ndim == 3:
                attn_cpu = attn_cpu[0]

            neural_probe.log_attention(
                layer_id,
                head_id,
                attn_cpu.numpy(),
                self.mode,
                self.max_entries,
                self.threshold,
            )

        return hook


def create_attention_capture(
    model: nn.Module, model_type: str = "auto"
) -> AttentionCapture:
    """
    Factory function to create appropriate attention capture.

    Args:
        model: PyTorch model
        model_type: "auto", "huggingface", or "generic"

    Returns:
        AttentionCapture instance
    """
    if model_type == "auto":
        # Detect model type
        model_class = type(model).__name__.lower()
        if any(x in model_class for x in ["bert", "gpt", "roberta", "llama", "opt"]):
            model_type = "huggingface"
        else:
            model_type = "generic"

    if model_type == "huggingface":
        return HuggingFaceAttentionCapture(model)
    return AttentionCapture(model)
