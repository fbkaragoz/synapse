import sys
import time

import numpy as np

# ensure we can import the installed module or local build
try:
    import neural_probe
except ImportError:
    print("Could not import neural_probe. Make sure it is installed.")
    sys.exit(1)

def main():
    print("Starting server...")
    neural_probe.start_server("localhost", 9000)
    
    # Set threshold to see sparse packets
    neural_probe.set_threshold(0.5)

    print("Simulating a single-layer loop (Ctrl+C to stop)...")
    print("For a multi-layer Llama-like stack, run: python backend_extension/python/simulate_llama8b.py")
    try:
        step = 0
        while True:
            # Simulate a layer activation
            # Random scaling to trigger threshold sometimes
            # Oscillate intensity to show dynamic particles
            intensity = 1.0 + np.sin(step * 0.1) * 0.5 
            data = np.random.rand(1024).astype(np.float32) * intensity
            
            # Log it
            neural_probe.log_activation(0, data)
            
            if step % 100 == 0:
                print(f"Training step {step} (Intensity: {intensity:.2f})")
            
            step += 1
            time.sleep(0.1) # Simulate computation
    except KeyboardInterrupt:
        pass
        
    print("Stopping server...")
    neural_probe.stop_server()
    print("[Python] Done.")

if __name__ == "__main__":
    main()
