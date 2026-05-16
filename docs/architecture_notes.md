# Architecture Notes: Latency-Optimized Heterogeneous AI Accelerator

## 1. System Overview
This project integrates custom HLS kernels for Softmax and GELU operations alongside a Xilinx DPU on the ZCU104 MPSoC. The goal is to provide a complete, high-performance pipeline for Vision Transformer (ViT) models.

## 2. Hardware Block Diagram
*(Instructions: Add your Vivado Block Design screenshot here as `architecture_diagram.png`)*

### Key Components:
- **Xilinx DPU (B4096):** Handles Conv/Linear layers.
- **vit_softmax:** Custom HLS kernel with AXI-Stream interface.
- **vit_gelu:** Custom HLS kernel with AXI-Stream interface.
- **AXI DMA:** Orchestrates data movement between PS (Processing System) and PL (Programmable Logic).

## 3. Data Pipeline
The data follows this path:
1. **Host Memory:** Input attention scores (Q8.8 fixed-point).
2. **AXI DMA (MM2S):** Streams data to the PL.
3. **Softmax Kernel:** Computes normalization.
4. **GELU Kernel:** Computes non-linear activation.
5. **AXI DMA (S2MM):** Streams results back to Host Memory.

## 4. Latency Analysis
The hardware pipeline is optimized for constant-time complexity O(1). 
- **Compute Time:** < 1ms (estimated based on clock cycles).
- **Driver Overhead:** ~50ms (Python `mmap` and `/dev/mem` interactions).

## 5. Software Driver
The driver is implemented in Python, utilizing `mmap` for direct register and memory access, ensuring low-latency control without the need for complex kernel drivers.
