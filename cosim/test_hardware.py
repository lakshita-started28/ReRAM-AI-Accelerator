#!/usr/bin/env python3
"""
Comprehensive Hardware Test - Run this FIRST
Tests all components before full co-simulation
"""

import sys
import numpy as np
from pathlib import Path

print("="*70)
print("COMPREHENSIVE HARDWARE TEST")
print("="*70)

# Test 1: Library exists
print("\n[1/6] Checking hardware library...")
lib_path = Path('cosim/libreram_hw.so')
if lib_path.exists():
    print(f"      ✓ Found: {lib_path}")
else:
    print(f"      ✗ NOT FOUND: {lib_path}")
    print("      ACTION: Run 'cd cosim && ./build.sh'")
    sys.exit(1)

# Test 2: Import Python wrapper
print("\n[2/6] Testing Python wrapper import...")
sys.path.insert(0, 'cosim')
try:
    from reram_hw import ReRAMHardwareAccelerator, SimpleCrossbarModel, SCALE_FACTOR
    print(f"      ✓ Import successful")
    print(f"      ✓ SCALE_FACTOR = {SCALE_FACTOR}")
except Exception as e:
    print(f"      ✗ Import failed: {e}")
    sys.exit(1)

# Test 3: Initialize hardware
print("\n[3/6] Initializing hardware...")
try:
    crossbar = SimpleCrossbarModel()
    mock_weights = np.random.randn(256, 784).astype(np.float32) * 0.1
    mock_bias = np.random.randn(256).astype(np.float32) * 0.1
    crossbar.set_weights(mock_weights, mock_bias)
    
    hw = ReRAMHardwareAccelerator(crossbar_model=crossbar)
    print("      ✓ Hardware initialized successfully")
except Exception as e:
    print(f"      ✗ Initialization failed: {e}")
    sys.exit(1)

# Test 4: Single inference
print("\n[4/6] Running single inference test...")
try:
    test_input = np.random.randint(0, 256, size=784, dtype=np.uint8)
    output = hw.run(test_input)
    cycles = hw.get_cycle_count()
    
    print(f"      ✓ Inference complete")
    print(f"        Output shape: {output.shape}")
    print(f"        Output range: [{output.min()}, {output.max()}]")
    print(f"        Cycles: {cycles}")
    print(f"        Time: {cycles * 10 / 1e6:.3f}ms")
    
    # Validate output
    if output.shape != (256,):
        print(f"      ✗ Wrong output shape: {output.shape}")
        sys.exit(1)
    
except Exception as e:
    print(f"      ✗ Inference failed: {e}")
    sys.exit(1)

# Test 5: Multiple inference (stress test)
print("\n[5/6] Running 10 inference stress test...")
try:
    cycles_list = []
    for i in range(10):
        test_input = np.random.randint(0, 256, size=784, dtype=np.uint8)
        output = hw.run(test_input)
        cycles = hw.get_cycle_count()
        cycles_list.append(cycles)
    
    avg_cycles = np.mean(cycles_list)
    std_cycles = np.std(cycles_list)
    
    print(f"      ✓ All 10 inferences passed")
    print(f"        Avg cycles: {int(avg_cycles)} ± {int(std_cycles)}")
    print(f"        Avg time: {avg_cycles * 10 / 1e6:.3f}ms")
    
except Exception as e:
    print(f"      ✗ Stress test failed: {e}")
    sys.exit(1)

# Test 6: Check model weights
print("\n[6/6] Checking trained model...")
model_path = Path('model/model_weights.pth')
if model_path.exists():
    print(f"      ✓ Found: {model_path}")
    import torch
    try:
        weights = torch.load(model_path)
        print(f"      ✓ Weights loaded successfully")
        print(f"        Keys: {list(weights.keys())[:3]}...")
    except Exception as e:
        print(f"      ⚠ Weights file exists but can't load: {e}")
else:
    print(f"      ⚠ NOT FOUND: {model_path}")
    print("      ACTION: Run 'python model/quick_train.py'")
    print("      NOTE: Will use random weights (lower accuracy)")

# Summary
print("\n" + "="*70)
print("✅ ALL HARDWARE TESTS PASSED")
print("="*70)
print("\nYou're ready to run:")
print("  1. Quick test:  python cosim/run_cosim.py --samples 10")
print("  2. Full test:   python cosim/run_cosim.py --samples 1000")
print("  3. Benchmarks:  python cosim/run_phase56.py --samples 1000")
print()