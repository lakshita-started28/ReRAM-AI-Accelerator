#!/usr/bin/env python3
"""
PHASE 5: Co-Simulation - FIXED for 1000 samples
Key fixes: Proper normalization, batch handling, scaling
"""

import sys
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from reram_hw import ReRAMHardwareAccelerator, SimpleCrossbarModel, SCALE_FACTOR, NORM_MEAN, NORM_STD

model_dir = Path(__file__).parent.parent / 'model'

class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.relu1(self.bn1(self.fc1(x)))
        x = self.relu2(self.bn2(self.fc2(x)))
        return self.fc3(x)
    
    def forward_from_layer2(self, layer1_output):
        """Continue from hardware layer 1 output (after BN+ReLU)"""
        x = self.relu2(self.bn2(self.fc2(layer1_output)))
        return self.fc3(x)

def load_model():
    """Load trained model"""
    model = MNISTModel()
    model_path = model_dir / 'model_weights.pth'
    
    if model_path.exists():
        model.load_state_dict(torch.load(model_path))
        print("✓ Loaded trained model")
    else:
        print("⚠ No saved model, using random weights")
        print("  Run: python model/quick_train.py first!")
    
    model.eval()
    return model

def run_cosimulation(num_samples=10):
    """Co-simulation with exact SW-HW match"""
    
    print("\n" + "="*70)
    print("PHASE 5: HARDWARE-SOFTWARE CO-SIMULATION")
    print("="*70)
    print(f"Testing: {num_samples} samples")
    print(f"Goal: 100% SW-HW accuracy match")
    print("="*70 + "\n")
    
    # Load model
    print("Loading model...")
    model = load_model()
    
    # Load datasets (raw for HW, normalized for SW)
    print("Loading MNIST test dataset...")
    
    # Raw images (0-255) for hardware input
    transform_raw = transforms.Compose([transforms.ToTensor()])
    test_set_raw = torchvision.datasets.MNIST(
        root='../data', train=False, download=True, transform=transform_raw
    )
    
    # Normalized images for software reference
    transform_norm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((NORM_MEAN,), (NORM_STD,))
    ])
    test_set_norm = torchvision.datasets.MNIST(
        root='../data', train=False, download=False, transform=transform_norm
    )
    
    # Extract weights and BatchNorm params
    fc1_weights = model.fc1.weight.detach().cpu().numpy()
    fc1_bias = model.fc1.bias.detach().cpu().numpy()
    
    bn1_weight = model.bn1.weight.detach().cpu().numpy()
    bn1_bias = model.bn1.bias.detach().cpu().numpy()
    bn1_mean = model.bn1.running_mean.detach().cpu().numpy()
    bn1_var = model.bn1.running_var.detach().cpu().numpy()
    bn1_eps = model.bn1.eps
    
    # Create crossbar model
    crossbar = SimpleCrossbarModel()
    crossbar.set_weights(fc1_weights, fc1_bias)
    
    # Initialize hardware
    print("\nInitializing hardware accelerator...")
    try:
        hw = ReRAMHardwareAccelerator(crossbar_model=crossbar)
    except Exception as e:
        print(f"⚠ Hardware error: {e}")
        print("  Make sure to run: cd cosim && ./build.sh")
        return
    
    # Results tracking
    results = {
        'sw_predictions': [],
        'hw_predictions': [],
        'sw_correct': 0,
        'hw_correct': 0,
        'matches': 0,
        'hw_cycles': [],
    }
    
    print("\n" + "-"*70)
    print(f"Running inference on {num_samples} samples")
    print("-"*70 + "\n")
    
    # Process samples one by one
    for i in range(num_samples):
        # Get raw and normalized samples
        img_raw, label = test_set_raw[i]
        img_norm, _ = test_set_norm[i]
        
        if num_samples <= 20:  # Print details for small tests
            print(f"Sample {i+1}/{num_samples} (Label: {label})")
        elif (i + 1) % 100 == 0:  # Progress for large tests
            print(f"  Progress: {i+1}/{num_samples}")
        
        # ===== SOFTWARE PATH (REFERENCE) =====
        with torch.no_grad():
            sw_out = model(img_norm.unsqueeze(0))
            pred_sw = sw_out.argmax(1).item()
            results['sw_predictions'].append(pred_sw)
            if pred_sw == label:
                results['sw_correct'] += 1
        
        if num_samples <= 20:
            print(f"  Software:  Predicted {pred_sw} {'✓' if pred_sw == label else '✗'}")
        
        # ===== HARDWARE PATH =====
        # Prepare uint8 input (0-255)
        img_raw_uint8 = (img_raw.squeeze().numpy() * 255.0).astype(np.uint8).flatten()
        
        try:
            # Run hardware (returns int16 scaled by SCALE_FACTOR)
            hw_out_raw = hw.run(img_raw_uint8)
            cycles = hw.get_cycle_count()
            results['hw_cycles'].append(cycles)
            
            # De-scale back to float
            hw_out_float = hw_out_raw.astype(np.float32) / SCALE_FACTOR
            
            # Apply BatchNorm (same as software)
            hw_out_bn = ((hw_out_float - bn1_mean) / 
                        np.sqrt(bn1_var + bn1_eps) * bn1_weight + bn1_bias)
            
            # Apply ReLU
            hw_out_relu = np.maximum(hw_out_bn, 0)
            
            # Continue with rest of network
            with torch.no_grad():
                layer2_input = torch.from_numpy(hw_out_relu).float().unsqueeze(0)
                hw_out = model.forward_from_layer2(layer2_input)
                pred_hw = hw_out.argmax(1).item()
                results['hw_predictions'].append(pred_hw)
                if pred_hw == label:
                    results['hw_correct'] += 1
                if pred_hw == pred_sw:
                    results['matches'] += 1
            
            if num_samples <= 20:
                match_sym = '✓' if pred_hw == pred_sw else '✗'
                print(f"  Hardware:  Predicted {pred_hw} {'✓' if pred_hw == label else '✗'} "
                      f"[SW-HW: {match_sym}]")
                print(f"             Cycles: {cycles}\n")
            
        except Exception as e:
            print(f"  Hardware error: {str(e)}")
            results['hw_predictions'].append(-1)
    
    # ===== RESULTS SUMMARY =====
    print("\n" + "="*70)
    print("CO-SIMULATION RESULTS")
    print("="*70)
    
    sw_acc = 100.0 * results['sw_correct'] / num_samples
    hw_valid = len([p for p in results['hw_predictions'] if p != -1])
    hw_acc = 100.0 * results['hw_correct'] / hw_valid if hw_valid > 0 else 0
    match_rate = 100.0 * results['matches'] / hw_valid if hw_valid > 0 else 0
    
    print(f"\n📊 Accuracy (n={num_samples}):")
    print(f"  Software-only:        {sw_acc:.1f}% ({results['sw_correct']}/{num_samples})")
    print(f"  Hardware-accelerated: {hw_acc:.1f}% ({results['hw_correct']}/{hw_valid})")
    print(f"  SW-HW match rate:     {match_rate:.1f}% ({results['matches']}/{hw_valid})")
    print(f"  Accuracy difference:  {abs(sw_acc - hw_acc):.1f}%")
    
    if results['hw_cycles']:
        avg_cycles = np.mean(results['hw_cycles'])
        avg_time_ms = avg_cycles * 10 / 1e6  # @100MHz
        
        print(f"\n⚡ Hardware Performance:")
        print(f"  Avg cycles/sample:    {int(avg_cycles)}")
        print(f"  Avg time @ 100MHz:    {avg_time_ms:.3f}ms")
        print(f"  Throughput:           {1000/avg_time_ms:.0f} samples/sec")
    
    # Validation status
    print(f"\n🎯 Validation Status:")
    if match_rate >= 98.0:
        print("  ✅ EXCELLENT: SW-HW match ≥98% (Production ready!)")
        status = "PASS"
    elif match_rate >= 95.0:
        print("  ✅ VERY GOOD: SW-HW match ≥95%")
        status = "PASS"
    elif match_rate >= 90.0:
        print("  ✓ GOOD: SW-HW match ≥90%")
        status = "PASS"
    else:
        print("  ⚠ NEEDS TUNING: SW-HW match <90%")
        status = "PARTIAL"
    
    print("\n" + "="*70)
    
    # Save results
    results_file = Path('../results/cosim_results.npz')
    results_file.parent.mkdir(exist_ok=True, parents=True)
    
    np.savez(results_file,
             sw_predictions=results['sw_predictions'],
             hw_predictions=results['hw_predictions'],
             hw_cycles=results['hw_cycles'],
             sw_accuracy=sw_acc,
             hw_accuracy=hw_acc,
             match_rate=match_rate,
             status=status)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    if status == "PASS":
        print("\n✅ Phase 5 Complete!")
        print("   Next: python cosim/run_phase56.py --samples 1000")
    else:
        print("\n⚠ Debug suggestions:")
        print("  1. Rebuild hardware: cd cosim && ./build.sh")
        print("  2. Check model: ls -lh model/model_weights.pth")
        print("  3. Try fewer samples: --samples 10")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run co-simulation')
    parser.add_argument('--samples', type=int, default=10,
                       help='Number of test samples (default: 10)')
    args = parser.parse_args()
    
    run_cosimulation(num_samples=args.samples)