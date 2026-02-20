"""
Gradient Capture Example for PyTorch Models

This module provides utilities to capture gradient statistics from
PyTorch models during training and stream them to the Synapse dashboard.

Usage:
    import torch
    import neural_probe
    from gradient_hook import GradientCapture

    # Start the probe server
    neural_probe.start_server("localhost", 9000)

    # Create capture for specific layers
    capture = GradientCapture(model, layers=[0, 5, 10, 15])
    capture.register_hooks()

    # Training loop
    for step, batch in enumerate(dataloader):
        neural_probe.set_training_step(step)

        # ... forward pass, loss, backward ...
        loss.backward()

        # Flush gradients after backward pass
        capture.flush()
        optimizer.step()
        optimizer.zero_grad()
"""

import re
from typing import List, Optional, Dict, Set
import torch
import torch.nn as nn

try:
    import neural_probe
except ImportError:
    neural_probe = None
    print("Warning: neural_probe not installed. Gradient capture disabled.")


class GradientCapture:
    def __init__(
        self,
        model: nn.Module,
        layers: Optional[List[int]] = None,
        capture_all: bool = False,
        flush_every: int = 1,
    ):
        """
        Initialize gradient capture for a model.

        Args:
            model: PyTorch model to capture gradients from
            layers: List of layer indices to capture (None = capture all if capture_all=True)
            capture_all: If True and layers is None, capture all parameter gradients
            flush_every: Flush gradient batch every N backward passes
        """
        self.model = model
        self.target_layers: Set[int] = set(layers) if layers else set()
        self.capture_all = capture_all or (layers is None)
        self.flush_every = flush_every
        self.backward_count = 0
        self.handles: List = []
        self._layer_name_map: Dict[str, int] = {}
        self._build_layer_map()

    def _build_layer_map(self):
        """Build mapping from parameter names to layer IDs."""
        layer_pattern = re.compile(r"layers\.(\d+)")

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            match = layer_pattern.search(name)
            if match:
                layer_id = int(match.group(1))
            else:
                layer_id = hash(name) % 10000

            self._layer_name_map[name] = layer_id

    def _should_capture(self, name: str) -> bool:
        """Check if this parameter should be captured."""
        if self.capture_all:
            return True

        layer_id = self._layer_name_map.get(name, -1)
        return layer_id in self.target_layers

    def _make_hook(self, name: str, param: nn.Parameter):
        """Create a backward hook for a parameter."""
        layer_id = self._layer_name_map.get(name, 0)

        def hook(grad):
            if grad is None:
                return

            if neural_probe is None:
                return

            grad_cpu = grad.detach().cpu().contiguous().float()
            weight_cpu = param.detach().cpu().contiguous().float()

            neural_probe.log_gradient(layer_id, grad_cpu.numpy(), weight_cpu.numpy())

        return hook

    def register_hooks(self):
        """Register backward hooks on model parameters."""
        self.remove_hooks()

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            if not self._should_capture(name):
                continue

            handle = param.register_post_accumulate_grad_hook(
                self._make_hook(name, param)
            )
            self.handles.append(handle)

        print(f"[GradientCapture] Registered {len(self.handles)} gradient hooks")

    def remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def flush(self):
        """Flush pending gradient statistics to the server."""
        if neural_probe is None:
            return

        self.backward_count += 1

        if self.backward_count % self.flush_every == 0:
            neural_probe.flush_gradient_batch()

    def set_training_step(self, step: int):
        """Update the training step counter."""
        if neural_probe is not None:
            neural_probe.set_training_step(step)


def create_attention_gradient_hook(layer_id: int, head_id: int = 0):
    """
    Create a hook specifically for attention weight gradients.

    Args:
        layer_id: Layer index
        head_id: Attention head index (for multi-head attention)

    Returns:
        Hook function for attention weights
    """

    def hook(grad):
        if grad is None or neural_probe is None:
            return

        grad_cpu = grad.detach().cpu().contiguous().float()

        stats = {
            "layer_id": layer_id,
            "head_id": head_id,
            "mean": float(grad_cpu.mean()),
            "std": float(grad_cpu.std()),
            "max": float(grad_cpu.max()),
            "min": float(grad_cpu.min()),
        }
        print(f"[Attention Gradient] Layer {layer_id} Head {head_id}: {stats}")

    return hook


class GradientMonitor:
    """
    High-level gradient monitoring with anomaly detection.
    """

    def __init__(
        self,
        model: nn.Module,
        explosion_threshold: float = 100.0,
        vanishing_threshold: float = 1e-8,
    ):
        self.model = model
        self.explosion_threshold = explosion_threshold
        self.vanishing_threshold = vanishing_threshold
        self.history: Dict[int, List[float]] = {}

    def check_gradients(self) -> Dict[str, List[int]]:
        """
        Check for gradient anomalies.

        Returns:
            Dictionary with 'exploded' and 'vanished' layer IDs
        """
        exploded = []
        vanished = []

        layer_pattern = re.compile(r"layers\.(\d+)")

        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue

            grad_norm = param.grad.norm().item()

            match = layer_pattern.search(name)
            layer_id = int(match.group(1)) if match else hash(name) % 10000

            if layer_id not in self.history:
                self.history[layer_id] = []
            self.history[layer_id].append(grad_norm)

            if grad_norm > self.explosion_threshold:
                exploded.append(layer_id)
            elif grad_norm < self.vanishing_threshold:
                vanished.append(layer_id)

        return {"exploded": exploded, "vanished": vanished}

    def get_statistics(self) -> Dict[int, Dict[str, float]]:
        """Get gradient statistics for all layers."""
        stats = {}

        layer_pattern = re.compile(r"layers\.(\d+)")

        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue

            match = layer_pattern.search(name)
            layer_id = int(match.group(1)) if match else hash(name) % 10000

            grad = param.grad.detach().cpu().float()

            stats[layer_id] = {
                "mean": float(grad.mean()),
                "std": float(grad.std()),
                "min": float(grad.min()),
                "max": float(grad.max()),
                "norm": float(grad.norm()),
            }

        return stats
