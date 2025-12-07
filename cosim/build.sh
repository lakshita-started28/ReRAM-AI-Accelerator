#!/bin/bash
# Build Verilator co-simulation library

set -e

echo "======================================"
echo "Building ReRAM Hardware Co-Simulation"
echo "======================================"

# Clean previous build
rm -rf obj_dir
rm -f libreram_hw.so

echo "Step 1: Running Verilator..."
verilator --cc \
    ../rtl/reram_controller_verilator.v \
    --top-module reram_controller_verilator \
    -Mdir obj_dir \
    -O3 --x-assign fast --x-initial fast \
    -CFLAGS "-fPIC" \
    -LDFLAGS "-shared"

# Add wrapper to Verilator build system
cp verilator_wrapper.cpp obj_dir/

# Build object files (but DO NOT link)
make -C obj_dir -f Vreram_controller_verilator.mk VM_TRACE=0

echo "Step 2: Linking shared library..."
g++ -shared -fPIC \
    -Wl,--export-dynamic \
    obj_dir/Vreram_controller_verilator__ALL.o \
    verilator_wrapper.cpp \
    /usr/share/verilator/include/verilated.cpp \
    -I/usr/share/verilator/include \
    -Iobj_dir \
    -o libreram_hw.so

if [ $? -ne 0 ]; then
    echo "Error: C++ linking failed"
    exit 1
fi


echo ""
echo "✓ Build successful!"
echo "  Output: libreram_hw.so"
echo ""
