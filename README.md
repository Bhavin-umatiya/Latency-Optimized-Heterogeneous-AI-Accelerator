# Latency-Optimized Heterogeneous AI Accelerator for Vision Transformers

Xilinx ZCU104 MPSoC

## 🚀 Key Visuals
| Architecture | Performance |
| :---: | :---: |
| ![Vivado Block Design](./results/bd_architecture.png) | ![Performance Comparison](./results/latency_comparison_graph.png) |
| *Hardware Interconnect* | *ARM vs FPGA Latency* |

---

## 🧠 Architecture Overview
This is a heterogeneous design combining high-performance deep learning cores with custom mathematical accelerators:
- **Xilinx DPU:** Dual-core (B4096) @ 300MHz for Linear/Conv layers.
- **Custom HLS Softmax:** C++ optimized kernel for attention normalization.
- **Custom HLS GELU:** C++ optimized kernel for transformer activations.
- **AXI-Stream Pipeline:** Zero-copy DMA data movement between PS and PL.

## ❓ Why Heterogeneous? (The "Non-GEMM" Bottleneck)

### ⚠️ The Problem: CPU-FPGA Memory Bottleneck
Standard accelerators like the Xilinx DPU are optimized for GEMM (matrix multiply) but lack native hardware support for Softmax (requires exp() and division) and GELU activations. In standard deployments, this causes a costly memory ping-pong bottleneck:
1. DPU computes a linear layer → result in DDR memory.
2. ARM CPU reads from DDR, computes Softmax in software (measured: **154.69 µs** for 768-dim vector).
3. ARM writes result back to DDR → DPU reads it again for the next layer.

This memory round-trip keeps the DPU idle and creates a memory wall that severely limits end-to-end system throughput.

### 🚀 Our Solution: Heterogeneous PL Pipeline
Custom Vitis HLS kernels integrated directly in the AXI4-Stream pipeline eliminate the ARM round-trip for non-linear operations. From timing invariance across 5-dim and 768-dim inputs (both measuring ~50 ms total with identical DMA overhead), we estimate the actual kernel compute time is **under 1.0 ms** on the FPGA fabric at 300 MHz.

This heterogeneous approach is architecturally inspired by dedicated non-linear units in industry accelerators such as the specialized **Transformer Engine** in NVIDIA's Hopper/Blackwell GPUs and the activation units in Google's TPUs.

## 📊 Benchmark Results — Measured on ZCU104 Hardware

### 5-Element Test Vector `[1.0, 2.0, 3.0, 4.0, 5.0]`
| Metric | ARM Cortex-A53 | FPGA Subsystem (ZCU104) |
|--------|---------------|-----------------|
| Software Compute Latency | **84.8 µs** | — |
| FPGA Kernel Compute (Estimated) | N/A | **< 1.0 ms** (from timing invariance) |
| End-to-End System Latency (Measured) | 84.8 µs | **~50.1 ms** (Python `/dev/mem` wrapper) |
| Output Correctness | sum = 1.0 ✅ | sum = 1.0 ✅ |
| Top Class | cls 4 | cls 4 ✅ |

### 768-Dimension ViT Attention Vector
| Metric | ARM Cortex-A53 | FPGA Subsystem (ZCU104) |
|--------|---------------|-----------------|
| Software Compute Latency | **154.69 µs** | — |
| FPGA Kernel Compute (Estimated) | N/A | **< 1.0 ms** (from timing invariance) |
| End-to-End System Latency (Measured) | 154.69 µs | **~50.1 ms** (Python `/dev/mem` wrapper) |
| Output Correctness | sum = 1.0 ✅ | sum = 1.0 ✅ |
| Top-3 Indices (ARM) | [209, 478, 179] | — |
| DPU Throughput | N/A | 2.4 TOPS |

> **System Latency Analysis:**
> * **Why ~50.1 ms?** The end-to-end system latency is entirely dominated by the high-level Python `mmap`/`/dev/mem` register write overhead and the conservative `time.sleep(0.05)` OS synchronization delay in the test wrapper.
> * **Why < 1.0 ms kernel compute?** Since the measured system latency remains virtually identical (~50.1 ms) when scaling from a tiny 5-element vector to a massive 768-element vector, the physical kernel compute time is invariant to the vector size at this scale and is estimated to be well under 1.0 ms at 300 MHz.

## 🛠️ Hardware Setup & Key Challenges
- **Platform:** Xilinx ZCU104 (Zynq UltraScale+ MPSoC)
- **OS:** PetaLinux 2022.2
- **HLS Tool:** Vitis HLS 2025.1
- **Block Design:** Vivado 2022.2 (reproduced via `hardware/vivado/vit_accelerator_bd.tcl`)
- **The "Silent Boot" Fix:** Resolved a critical FSBL/PMUFW incompatibility between Vitis 2025.1 and PetaLinux 2022.2 by implementing a hybrid `bootgen` strategy using `manual.bif`.

## 📦 Prerequisites
```bash
# On the ZCU104 board (PetaLinux 2022.2)
pip3 install numpy

# On host (for block design rebuild)
# Vivado 2022.2 + Vitis HLS 2025.1
```

## 🚀 How to Run
1. **Load the bitstream:**
   ```bash
   cd /run/media/mmcblk0p1
   fpgautil -b system.bit
   # Expected: "BIN FILE loaded through FPGA manager successfully"
   ```
2. **Verify hardware is alive:**
   ```bash
   devmem 0xA0000000   # DMA   -> expect 0x00010002 (Idle)
   devmem 0xB0000000   # GELU  -> expect 0x00000004 (ap_idle)
   devmem 0xB0010000   # Softmax -> expect 0x00000004 (ap_idle)
   ```
3. **Transfer benchmark scripts:**
   ```bash
   scp software/benchmarks/*.py root@<board_ip>:/run/media/mmcblk0p1/
   ```
4. **Run 5-element benchmark:**
   ```bash
   python3 /run/media/mmcblk0p1/final_test.py
   ```
5. **Run 768-dim ViT benchmark:**
   ```bash
   python3 /run/media/mmcblk0p1/bench_768.py
   ```

## 📁 Project Structure
- `hardware/hls/`: Optimized HLS source code (C++).
- `hardware/vivado/`: Block design TCL and manual.bif.
- `software/benchmarks/`: Python drivers and test suite.
- `results/`: Performance logs and hardware proof.
- `docs/`: Architecture reports and diagrams.

## ✅ Results Proof
![Hardware Success - Green Lights](./results/hardware_success.jpg)

## 👤 Author
**Bhavin Umatiya**
B.Tech Electronics & communication

[![GitHub](https://img.shields.io/badge/GitHub-Bhavin--umatiya-181717?logo=github)](https://github.com/Bhavin-umatiya)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?logo=linkedin)](https://www.linkedin.com/in/bhavin-umatiya/)
