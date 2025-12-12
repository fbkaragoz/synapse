from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np

try:
    import neural_probe
except ImportError as e:
    raise SystemExit(
        "Could not import neural_probe. Build/install the backend extension first.\n"
        "Tip: from repo root, run: `pip install -e backend_extension --no-build-isolation`"
    ) from e


@dataclass(frozen=True)
class Llama8BLike:
    n_layers: int = 32
    d_model: int = 4096
    d_ff: int = 11008


def _skewed_noise(rng: np.random.Generator, n: int) -> np.ndarray:
    # Skew towards small values so a mid threshold doesn't light everything up.
    x = rng.random(n, dtype=np.float32)
    return x * x * x


def _wave(n: int, t: float, phase: float) -> np.ndarray:
    idx = np.arange(n, dtype=np.float32)
    return (0.12 * (np.sin(idx * 0.02 + t * 1.7 + phase) + 1.0)).astype(np.float32)


def make_activation(rng: np.random.Generator, n: int, t: float, layer_phase: float) -> np.ndarray:
    x = _skewed_noise(rng, n)
    x += _wave(n, t, layer_phase)
    np.clip(x, 0.0, 1.0, out=x)
    return x.astype(np.float32, copy=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Neural-Flow: Llama-8B-like activation simulator")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--fps", type=float, default=10.0, help="Packets cadence (simulation ticks/sec)")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layers", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=4096)
    parser.add_argument("--d-ff", type=int, default=11008)
    parser.add_argument(
        "--per-block",
        choices=["residual", "attn+mlp", "attn+mlp+residual"],
        default="attn+mlp",
        help="How many activation 'surfaces' to emit per transformer block",
    )
    args = parser.parse_args()

    arch = Llama8BLike(n_layers=args.layers, d_model=args.d_model, d_ff=args.d_ff)
    rng = np.random.default_rng(args.seed)

    print(f"[sim] starting server ws://{args.host}:{args.port}")
    neural_probe.start_server(args.host, args.port)
    neural_probe.set_threshold(float(args.threshold))

    tick_s = 1.0 / max(0.1, float(args.fps))
    start = time.perf_counter()
    step = 0

    # Layer id scheme (sorted numeric => stable Z-stack):
    # per block: emit 2 or 3 "planes" with different sizes
    # - attn_out: d_model
    # - mlp_out : d_ff
    # - residual: d_model
    def layer_ids(block: int):
        base = block * 3
        if args.per_block == "residual":
            return [(base + 2, arch.d_model, block * 0.37 + 2.0)]
        if args.per_block == "attn+mlp":
            return [
                (base + 0, arch.d_model, block * 0.37 + 0.0),
                (base + 1, arch.d_ff, block * 0.37 + 1.0),
            ]
        return [
            (base + 0, arch.d_model, block * 0.37 + 0.0),
            (base + 1, arch.d_ff, block * 0.37 + 1.0),
            (base + 2, arch.d_model, block * 0.37 + 2.0),
        ]

    print("[sim] streaming activations (Ctrl+C to stop)")
    try:
        while True:
            t = time.perf_counter() - start

            for block in range(arch.n_layers):
                for layer_id, n, phase in layer_ids(block):
                    act = make_activation(rng, n, t, phase)
                    neural_probe.log_activation(layer_id, act)

            step += 1
            if step % 10 == 0:
                print(f"[sim] step={step} layers={arch.n_layers} per_block={args.per_block} threshold={args.threshold}")

            time.sleep(tick_s)
    except KeyboardInterrupt:
        pass
    finally:
        # NOTE: stop_server may block depending on uWebSockets shutdown wiring;
        # leaving it out is acceptable for a simulator.
        try:
            neural_probe.stop_server()
        except Exception:
            pass


if __name__ == "__main__":
    main()
