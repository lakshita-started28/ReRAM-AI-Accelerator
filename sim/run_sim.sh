#!/bin/bash

echo "========================================="
echo "ReRAM Controller Simulation"
echo "========================================="

# Clean previous simulation
rm -f sim/*.vcd sim/*.vvp

# Compile Verilog files with Icarus Verilog
echo "Compiling Verilog files..."
iverilog -o sim/reram_sim.vvp \
    rtl/reram_controller.v \
    rtl/dac_10bit.v \
    rtl/adc_12bit.v \
    testbench/tb_reram_controller.v

if [ $? -ne 0 ]; then
    echo "✗ Compilation failed!"
    exit 1
fi

echo "✓ Compilation successful"

# Run simulation
echo "Running simulation..."
vvp sim/reram_sim.vvp

if [ $? -ne 0 ]; then
    echo "✗ Simulation failed!"
    exit 1
fi

echo ""
echo "✓ Simulation complete"
echo "Waveform saved to: sim/reram_controller.vcd"
echo ""
echo "To view waveform:"
echo "  gtkwave sim/reram_controller.vcd"