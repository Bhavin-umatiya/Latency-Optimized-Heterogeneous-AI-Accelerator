import mmap
import struct
import time
import numpy as np

# ── Hardware Address Map ────────────────────────────────────────────────────
DMA_BASE  = 0xA0000000
GELU_BASE = 0xB0000000
SOFT_BASE = 0xB0010000
RAM_IN    = 0x1B800000
RAM_OUT   = 0x1B801000

# ── AXI DMA Register Offsets (Xilinx PG021) ────────────────────────────────
DMA_MM2S_CTRL   = 0x00   # MM2S Control
DMA_MM2S_SR     = 0x04   # MM2S Status (bit1 = Idle)
DMA_MM2S_ADDR   = 0x18   # MM2S Source address
DMA_MM2S_LEN    = 0x28   # MM2S Transfer length
DMA_S2MM_CTRL   = 0x30   # S2MM Control
DMA_S2MM_SR     = 0x34   # S2MM Status (bit1 = Idle)
DMA_S2MM_ADDR   = 0x48   # S2MM Destination address
DMA_S2MM_LEN    = 0x58   # S2MM Transfer length

# ── Kernel Register Offsets (Vitis HLS s_axilite) ──────────────────────────
KERN_CTRL  = 0x00  # bit0=start, bit1=done, bit2=idle, bit3=ready
KERN_SR    = 0x00  # same register, read-back

TIMEOUT_US = 500_000  # 500 ms poll timeout

# ── Low-level helpers ───────────────────────────────────────────────────────
def write_reg(mem, offset, value):
    mem.seek(offset)
    mem.write(struct.pack('I', value))

def read_reg(mem, offset):
    mem.seek(offset)
    return struct.unpack('I', mem.read(4))[0]

def wait_dma_idle(mem_dma, sr_offset, label="DMA"):
    """Poll status register until bit1 (Idle) is set or timeout."""
    t0 = time.perf_counter()
    while True:
        sr = read_reg(mem_dma, sr_offset)
        if sr & 0x2:  # bit1 = Idle
            return sr
        if (time.perf_counter() - t0) * 1e6 > TIMEOUT_US:
            raise TimeoutError(f"{label} timeout! SR=0x{sr:08X}")

# ── Main ────────────────────────────────────────────────────────────────────
print("=" * 55)
print("  HETEROGENEOUS ViT ACCELERATOR BENCHMARK")
print("=" * 55)

# [1] ARM baseline
test_vec = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
N_ITER = 10_000
t0 = time.perf_counter()
for _ in range(N_ITER):
    e = np.exp(test_vec - test_vec.max())
    arm_out = e / e.sum()
t1 = time.perf_counter()
arm_us = ((t1 - t0) / N_ITER) * 1e6

print("\n[1] ARM Cortex-A53 Softmax baseline (5-element)")
print("    Output : " + str(np.round(arm_out, 4)))
print("    Sum    : " + str(round(float(arm_out.sum()), 6)))
print("    Latency: " + str(round(arm_us, 2)) + " us")

# [2] FPGA HLS Kernels
print("\n[2] FPGA HLS Kernels (5-element, with DMA polling)")

with open("/dev/mem", "r+b") as f:
    mem_dma  = mmap.mmap(f.fileno(), 0x10000, offset=DMA_BASE)
    mem_gelu = mmap.mmap(f.fileno(), 0x10000, offset=GELU_BASE)
    mem_soft = mmap.mmap(f.fileno(), 0x10000, offset=SOFT_BASE)
    mem_in   = mmap.mmap(f.fileno(), 0x1000,  offset=RAM_IN)
    mem_out  = mmap.mmap(f.fileno(), 0x1000,  offset=RAM_OUT)

    # Reset DMA channels
    write_reg(mem_dma, DMA_MM2S_CTRL, 0x4)   # reset
    write_reg(mem_dma, DMA_S2MM_CTRL, 0x4)   # reset
    wait_dma_idle(mem_dma, DMA_MM2S_SR, "MM2S reset")
    wait_dma_idle(mem_dma, DMA_S2MM_SR, "S2MM reset")

    # Write input (5 x Q8.8 fixed-point values)
    mem_in.seek(0)
    for v in [256, 512, 768, 1024, 1280]:   # 1.0, 2.0, 3.0, 4.0, 5.0 in Q8.8
        mem_in.write(struct.pack('H', v))

    # Start HLS kernels (one-shot, bit0)
    write_reg(mem_soft, KERN_CTRL, 0x01)
    write_reg(mem_gelu, KERN_CTRL, 0x01)

    fpga_start = time.perf_counter()

    # Setup S2MM (receive) first
    write_reg(mem_dma, DMA_S2MM_CTRL, 0x1)
    write_reg(mem_dma, DMA_S2MM_ADDR, RAM_OUT)
    write_reg(mem_dma, DMA_S2MM_LEN, 10)     # 5 values × 2 bytes

    # Setup MM2S (send)
    write_reg(mem_dma, DMA_MM2S_CTRL, 0x1)
    write_reg(mem_dma, DMA_MM2S_ADDR, RAM_IN)
    write_reg(mem_dma, DMA_MM2S_LEN, 10)     # 5 values × 2 bytes

    # ── Poll until both channels are idle (no sleep!) ──────────────────────
    mm2s_sr = wait_dma_idle(mem_dma, DMA_MM2S_SR, "MM2S")
    s2mm_sr = wait_dma_idle(mem_dma, DMA_S2MM_SR, "S2MM")

    fpga_end = time.perf_counter()
    fpga_us  = (fpga_end - fpga_start) * 1e6

    soft_sr = read_reg(mem_soft, KERN_SR)
    gelu_sr = read_reg(mem_gelu, KERN_SR)

    print("    Softmax SR  : " + hex(soft_sr))
    print("    GELU SR     : " + hex(gelu_sr))
    print("    DMA MM2S SR : " + hex(mm2s_sr))
    print("    DMA S2MM SR : " + hex(s2mm_sr))

    # Read back outputs
    print("\n    Raw outputs (Q8.8):")
    raw = []
    for i in range(5):
        mem_out.seek(i * 2)
        v = struct.unpack('H', mem_out.read(2))[0]
        raw.append(v)
        print("    Val " + str(i) + ": " + str(v) + "  -> " + str(round(v / 256.0, 4)))

    total = sum(raw) if sum(raw) > 0 else 1
    norm  = [round(v / total, 4) for v in raw]
    print("\n    Normalized   : " + str(norm))
    print("    Sum (->1.0?) : " + str(round(sum(norm), 6)))
    print("    Predicted cls: " + str(norm.index(max(norm))))

print("\n" + "=" * 55)
print("  BENCHMARK RESULTS")
print("=" * 55)
print("  ARM  Softmax : " + str(round(arm_us,       2)) + " us")
print("  FPGA (polled): " + str(round(fpga_us,      2)) + " us")
print("  ARM  Latency : " + str(round(arm_us/1000,  3)) + " ms")
print("  FPGA Latency : " + str(round(fpga_us/1000, 3)) + " ms")
if fpga_us < arm_us:
    print("  SPEEDUP      : " + str(round(arm_us / fpga_us, 2)) + "x faster on FPGA")
else:
    print("  NOTE: DMA setup overhead dominates small 5-element vector")
    print("  Run bench_768.py for real ViT 768-dim results")
print("=" * 55)
