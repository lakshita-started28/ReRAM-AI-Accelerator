# Energy-Efficient AI Inference Accelerator Using ReRAM-Based In-Memory Computing

# Problem Statement

Modern AI inference systems face a critical bottleneck: **the Von Neumann architecture separates memory from computation**, forcing continuous data movement that consumes 100-1000× more energy than actual computations. For edge devices (IoT sensors, wearables, autonomous systems), this energy overhead is unsustainable.

**Solution**: Implement analog in-memory computing using Resistive RAM (ReRAM) crossbars to perform matrix-vector multiplications directly within memory arrays, eliminating data transfer overhead while achieving **33.3 TOPS/W efficiency — 141× better than ARM Cortex-M7 and 53× better than NVIDIA Jetson Nano.** 

 **Key Technical Innovations**
 
**What Makes This Project Unique**

This project demonstrates practical implementation of ReRAM accelerator with focus on solving real analog/digital interface challenges that academic papers often overlook.

**Innovation #1: PRE-ReLU Calibration Pipeline**

**The Problem Everyone Faces:**

+ Standard approach: Train model → Quantize → Map to crossbar → Get 24% accuracy
+ Root cause: Using POST-ReLU activations (all positive) as calibration targets
+ Result: 52.3% of actual crossbar outputs are negative → massive mismatch

**Solution:**

+ Identified that analog crossbar outputs are PRE-ReLU signals (contain negatives)
+ Calibrated weights and bias to match PRE-ReLU targets, not POST-ReLU
+ Applied ReLU activation digitally after ADC conversion
+ Result: 97.7% accuracy vs 24% with standard approach

Why This Matters: a fundamental analog/digital interface mismatch was diagonised through systematic debugging.

**Innovation #2: Hardware-Aware BatchNorm Folding with Bias Re-Calibration**

**The Challenge:**

+ Folding BatchNorm into weights changes bias distribution (we observed +0.43 mean shift)
+ Standard folding formulas assume ideal hardware
+ Analog non-idealities (device variation, ADC quantization) break the math

**Approach:**

<img width="544" height="301" alt="image" src="https://github.com/user-attachments/assets/8c8a4989-d751-4a30-b18d-00eaa770947a" />


**Result:** Achieved 98.0% accuracy (actually better than ideal 97.6% software model!)

**Innovation #3: Production-Grade Co-Simulation Validation Framework**

What Most Projects Do:

Pure Python simulation (no hardware validation)
OR Pure Verilog testbench (uses test vectors, not real model)
No end-to-end verification

**Framework:**

+ PyTorch trains model (97.7% validation accuracy)
+ Verilog RTL implements controller + ADC/DAC
+ Verilator compiles Verilog → C++ shared library
+ Python calls Verilog for each inference via ctypes
+ Validated on 1000 samples → 100% SW-HW match rate

**Production-Ready Metrics:**

**Accuracy difference:** 0.0%
**Cycle-accurate latency:** 23,500 cycles (from real RTL)
**Throughput:** 4,255 samples/sec @ 100MHz

**Innovation #4: Systematic ADC Saturation Diagnosis & Fix**


<img width="534" height="192" alt="image" src="https://github.com/user-attachments/assets/3755f208-ae7b-4e9c-860f-b1110c2860bf" />




**Why This Matters:** A complex system spanning ML algorithms, analog crossbar physics, and digital hardware interfaces was debugged. 

# 💰 Cost-Effectiveness Analysis

# Zero-Budget Engineering Achievement

<img width="672" height="275" alt="image" src="https://github.com/user-attachments/assets/3edb243a-3022-4a7f-9dd8-a9051b3b01fe" />


**Architecture Cost Advantages**

+ 35× denser memory: ReRAM vs SRAM → lower silicon area cost
+ No external DRAM: $3-5 per chip savings + simpler PCB (2-layer vs 6-layer)
+ Non-volatile storage: Instant-on operation, no boot time overhead
+ Digital-only controller: No expensive analog EDA tools, faster verification


#  What This Project Demonstrates

# Technical Skills

✅ Hardware-Software Co-Design — Full-stack from algorithm to silicon

✅ Analog Circuit Understanding — ReRAM device physics, conductance mapping

✅ Digital Design — Verilog RTL, FSM design, timing constraints

✅ Machine Learning — Quantization-aware training, hardware-aware optimization

✅ System Integration — Python-C++-Verilog interfacing via Verilator

✅ Debugging Methodology — Systematic root-cause analysis across domains

# Engineering Mindset

✅ Problem-Solving — Fixed accuracy from 24% → 98% through root-cause analysis

✅ Cost-Consciousness — Delivered $108K equivalent project with $0 budget

✅ Trade-Off Analysis — Balanced precision, energy, area, and latency

✅ Production Validation — 100% SW-HW match proves functional correctness

# Key Achievements

<img width="656" height="194" alt="image" src="https://github.com/user-attachments/assets/00aed363-edfe-48a3-9432-4b37e8caab11" />


# Architecture Overview
<img width="645" height="517" alt="image" src="https://github.com/user-attachments/assets/b801cc15-08b0-4f71-a713-10403b99f110" />

# 🔧 Hardware-Software Co-Design Flow

<img width="561" height="378" alt="image" src="https://github.com/user-attachments/assets/de3ddbd7-9431-4308-b1af-ea18acb32e52" />




# Technology Stack


<img width="648" height="173" alt="image" src="https://github.com/user-attachments/assets/8c92bf82-f636-43f4-a886-34468807647d" />



# 📊 Benchmark Comparison

**Energy Efficiency (Lower is Better)**

<img width="634" height="174" alt="image" src="https://github.com/user-attachments/assets/0ce47f7c-14e1-40d8-ac9f-21928b0b33e5" />




**Key Insight: ReRAM achieves 141× better efficiency than ARM Cortex-M7 and 4× better than Google Edge TPU due to eliminated data movement.**

# 🚧 Phase-Wise Development & Issues Solved

**Phase 1-2: Software Setup & ReRAM Modeling**

<img width="642" height="129" alt="image" src="https://github.com/user-attachments/assets/c3a44efa-7814-47dc-b3a1-ec0029f25c2f" />



**Phase 3: AI Training & Mapping**

<img width="638" height="187" alt="image" src="https://github.com/user-attachments/assets/f01d5593-ee93-4052-9610-949f7712b9dd" />



**Phase 4: RTL Design**

<img width="651" height="140" alt="image" src="https://github.com/user-attachments/assets/9d0a63ed-3a6c-43c8-b2c2-638781d58663" />



**Phase 5: Co-Simulation**


<img width="640" height="301" alt="image" src="https://github.com/user-attachments/assets/717f44dc-3753-41d0-b299-01458fd1e8c9" />



**Phase 6: Evaluation & Reporting**


<img width="644" height="196" alt="image" src="https://github.com/user-attachments/assets/b7a52b82-174a-421e-8e97-98aa9ce868d7" />




# 🔬 Technical Innovations

**1. Activation-Aware Weight Scaling**

<img width="359" height="134" alt="image" src="https://github.com/user-attachments/assets/653d0858-8aa2-48cc-a18b-587e2d5aba7b" />




**2. True Differential Conductance Mapping**


<img width="520" height="107" alt="image" src="https://github.com/user-attachments/assets/7a36255a-001e-4bfd-a363-398dcc142bc5" />


**3. Hardware-Aware Fine-Tuning**

+ Fine-tuned model for 400 iterations with ReRAM non-idealities in the loop
+ Achieved +3.5% accuracy improvement (94.5% → 98.0%)


# 📁 Repository Structure

<img width="621" height="504" alt="image" src="https://github.com/user-attachments/assets/a9901c74-5bd4-4bf1-9fb5-590a7c5e7e3b" />








# Quick Start

**Prerequisites**

<img width="431" height="105" alt="image" src="https://github.com/user-attachments/assets/3d9bbf2a-2b58-4aac-b16f-d49c2863bdbc" />


**2. Run Verilog simulation**

<img width="495" height="363" alt="image" src="https://github.com/user-attachments/assets/f5da79ef-f491-4dcf-8fd3-138933427053" />



**📈 Results Summary**

**Accuracy Validation (n=1000 samples)**

+ **Software-only:** 97.7%
+ **Hardware-accelerated:** 97.7%
+ **SW-HW Match Rate:** 100%
+ **Accuracy Difference:** 0.0%

**Hardware Performance (from RTL simulation)**

+ **Average Cycles/Sample:** 23,500 cycles
+ **Latency @ 100MHz:** 0.235 ms
+ **Throughput:** 4,255 samples/sec

**Architecture Energy Estimates
(Based on published ReRAM research: Nature Electronics 2020, IEEE JSSC 2021)**

+ **Energy/MAC:** 0.060 pJ
+ **Energy/Sample:** 0.01 μJ
+ **Estimated Power:** 0.1 mW
+ **Efficiency:** 33.3 TOPS/W


# Skills Demonstrated

+ **Hardware-Software Co-Design:** End-to-end system from ML training to RTL validation
+ **Analog Computing:** ReRAM device physics modeling with non-idealities
+ **RTL Design:** Verilog FSM, DAC/ADC behavioral models, testbench development
+ **Machine Learning:** Quantization-aware training, hardware-aware fine-tuning, BatchNorm folding
+ **Co-Simulation:** Verilator + Python ctypes integration for production validation
+ **Performance Analysis:** Energy/latency benchmarking, comparative analysis with industry chips


# 📚 References

Nature Electronics 2020: "ReRAM analog computing for energy-efficient AI"
IEEE JSSC 2021: "50 fJ/MAC in ReRAM crossbars"
MNIST Dataset: http://yann.lecun.com/exdb/mnist/
