#!/usr/bin/env python3
"""
Python interface to Verilator-compiled hardware
FIXED: Matches your normalization strategy exactly
"""

import ctypes
import numpy as np
from pathlib import Path

# CRITICAL: Must match quick_train.py and run_cosim.py
NORM_MEAN = 0.1307
NORM_STD = 0.3081
SCALE_FACTOR = 650.0  # Your chosen scale factor

class ReRAMHardwareAccelerator:
    """Python wrapper for Verilog hardware accelerator"""
    
    def __init__(self, library_path='cosim/libreram_hw.so', crossbar_model=None):
        self.lib_path = Path(library_path)
        
        if not self.lib_path.exists():
            raise FileNotFoundError(
                f"Hardware library not found: {library_path}\n"
                f"Run 'cd cosim && ./build.sh' to build it"
            )
        
        # Load library
        self.lib = ctypes.CDLL(str(self.lib_path))
        
        # Define function signatures
        self.lib.hardware_create.restype = ctypes.c_void_p
        self.lib.hardware_destroy.argtypes = [ctypes.c_void_p]
        
        self.lib.hardware_set_callback.argtypes = [
            ctypes.c_void_p,
            ctypes.CFUNCTYPE(ctypes.c_int16, ctypes.c_uint16, ctypes.POINTER(ctypes.c_uint8))
        ]
        
        self.lib.hardware_run_inference.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_int16)
        ]
        self.lib.hardware_run_inference.restype = None
        
        self.lib.hardware_get_cycles.argtypes = [ctypes.c_void_p]
        self.lib.hardware_get_cycles.restype = ctypes.c_uint64
        
        # Create hardware instance
        self.hw = self.lib.hardware_create()
        
        # Store crossbar model
        self.crossbar = crossbar_model
        
        # Create callback function
        self.callback_func = ctypes.CFUNCTYPE(
            ctypes.c_int16,
            ctypes.c_uint16,
            ctypes.POINTER(ctypes.c_uint8)
        )(self._crossbar_callback)
        
        # Register callback
        self.lib.hardware_set_callback(self.hw, self.callback_func)
        
        print("✓ Hardware accelerator initialized")
    
    def _crossbar_callback(self, neuron_idx, pixels_ptr):
        """
        Callback from C++ - compute one neuron
        CRITICAL: Must match your training normalization exactly
        """
        if self.crossbar is None:
            return int(sum(pixels_ptr[:10]))
        
        # 1. Convert uint8 pointer to numpy (raw pixels 0-255)
        pixels_raw = np.ctypeslib.as_array(pixels_ptr, shape=(784,))
        
        # 2. CRITICAL: Normalize EXACTLY like training
        # Training uses: transforms.Normalize((NORM_MEAN,), (NORM_STD,))
        # Formula: (x - mean) / std
        pixels_float = pixels_raw.astype(np.float64) / 255.0  # [0,1]
        pixels_normalized = (pixels_float - NORM_MEAN) / NORM_STD
        
        # 3. Compute neuron (returns float pre-activation)
        result_float = self.crossbar.compute_neuron(neuron_idx, pixels_normalized)
        
        # 4. Scale to int16 range for hardware transmission
        result_scaled = int(np.clip(result_float * SCALE_FACTOR, -32768, 32767))
        
        return result_scaled
    
    def run(self, input_pixels):
        """
        Run inference on hardware
        
        Args:
            input_pixels: numpy array (784,) of uint8 [0-255]
        
        Returns:
            output_neurons: numpy array (256,) of int16 (scaled by SCALE_FACTOR)
        """
        assert input_pixels.shape == (784,), f"Expected (784,), got {input_pixels.shape}"
        assert input_pixels.dtype == np.uint8, f"Expected uint8, got {input_pixels.dtype}"
        
        # Prepare buffers
        input_buf = input_pixels.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        output_buf = (ctypes.c_int16 * 256)()
        
        # Run hardware
        self.lib.hardware_run_inference(self.hw, input_buf, output_buf)
        
        # Convert to numpy
        output_neurons = np.array(output_buf, dtype=np.int16)
        
        return output_neurons
    
    def get_cycle_count(self):
        """Get number of clock cycles used"""
        return self.lib.hardware_get_cycles(self.hw)
    
    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'hw') and self.hw:
            self.lib.hardware_destroy(self.hw)


class SimpleCrossbarModel:
    """ReRAM crossbar - exact software emulation"""
    
    def __init__(self):
        self.weights = None
        self.bias = None
        self.scale_factor = SCALE_FACTOR # <--- ADD THIS LINE
    
    def set_weights(self, weights, bias):
        """Load trained weights"""
        self.weights = weights.astype(np.float64)
        self.bias = bias.astype(np.float64)
        
        print(f"✓ Crossbar loaded:")
        print(f"    Weights: [{self.weights.min():.3f}, {self.weights.max():.3f}]")
        print(f"    Bias: [{self.bias.min():.3f}, {self.bias.max():.3f}]")
    
    def compute_neuron(self, neuron_idx, pixels_normalized):
        """
        Compute one neuron output
        
        Args:
            neuron_idx: 0-255
            pixels_normalized: (784,) float64 normalized array
        
        Returns:
            float64 pre-activation output (before BatchNorm, before ReLU)
        """
        # Simple linear: y = W @ x + b
        output = np.dot(self.weights[neuron_idx], pixels_normalized) + self.bias[neuron_idx]
        return output


if __name__ == "__main__":
    # Test the hardware interface
    print("\n" + "="*60)
    print("Testing Hardware Interface")
    print("="*60)
    
    # Create mock crossbar
    crossbar = SimpleCrossbarModel()
    mock_weights = np.random.randn(256, 784).astype(np.float32) * 0.1
    mock_bias = np.random.randn(256).astype(np.float32) * 0.1
    crossbar.set_weights(mock_weights, mock_bias)
    
    # Initialize hardware
    hw = ReRAMHardwareAccelerator(crossbar_model=crossbar)
    
    # Create test input
    test_input = np.random.randint(0, 256, size=784, dtype=np.uint8)
    
    print("\nRunning test inference...")
    output = hw.run(test_input)
    cycles = hw.get_cycle_count()
    
    print(f"✓ Inference complete")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min()}, {output.max()}]")
    print(f"  Clock cycles: {cycles}")
    print(f"  Time @ 100MHz: {cycles * 10 / 1e6:.2f}ms")