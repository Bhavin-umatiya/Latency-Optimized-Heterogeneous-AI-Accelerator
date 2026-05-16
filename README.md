# Latency-Optimized Heterogeneous AI Accelerator for Vision Transformers

Final year project — 7th semester
Platform: Xilinx ZCU104 MPSoC

## Architecture

Heterogeneous design combining:
- Dual-core Xilinx DPU (DPUCZDX8G_ISA1_B4096) at 300MHz = 2.4 TOPS
- Custom HLS Softmax kernel (Vitis HLS C++)
- Custom HLS GELU kernel (Vitis HLS C++)
- AXI-Stream DMA interconnect

## Why heterogeneous?

The Xilinx DPU handles linear matrix operations efficiently.
Vision Transformers require Softmax, LayerNorm, and GELU which
the DPU cannot execute. Custom HLS kernels handle these operations,
creating a truly heterogeneous pipeline.

## Benchmark Results

| Metric | ARM Cortex-A53 | FPGA HLS Kernel |
|--------|---------------|-----------------|
| Softmax 5-dim | 84.8 us | <1 ms kernel |
| Softmax 768-dim | 154.7 us | <1 ms kernel |
| Output correctness | sum=1.0 | sum=1.0 |
| DPU throughput | N/A | 2.4 TOPS INT8 |
| DPU cores | N/A | 2 x B4096 @ 300MHz |

Note: End-to-end FPGA latency includes ~50ms DMA setup overhead
from Python /dev/mem driver. Kernel compute time is under 1ms,
confirmed by timing invariance across 5-dim and 768-dim vectors.

## Hardware Setup

- Board: Xilinx ZCU104 MPSoC
- Tool: Vivado 2022.2 + Vitis HLS + Vitis AI 3.0
- OS: PetaLinux 2022.2
- Boot: Hybrid strategy (2022.2 base image + custom bitstream)

## Key Challenge Solved

Silent boot hang after BL31 caused by Vitis 2025.1 FSBL/PMUFW
incompatibility with PetaLinux 2022.2 kernel. Solved using hybrid
bootgen strategy with 2022.2 ELF files and custom Vivado bitstream.

## Project Structure

hardware/vivado/     - Vivado block design files
hardware/hls/        - HLS kernel source code (C++)
software/            - Python test and benchmark scripts
results/             - Benchmark output screenshots
docs/                - Architecture diagrams and report

## Results

![Hardware Success - Green Lights](./results/hardware_success.jpg)

xdputil query confirms:
- 2 DPU cores active at 0x80000000 and 0x80001000
- Architecture: DPUCZDX8G_ISA1_B4096
- Frequency: 300 MHz
- Vitis AI runtime 3.0 fully operational

## Author

Bhavin Umatiya
B.Tech Electronics/VLSI — 7th Semester
