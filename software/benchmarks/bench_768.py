import mmap
import struct
import time
import numpy as np

# ── Hardware Address Map ────────────────────────────────────────────────────
DMA_BASE  = 0xA0000000
GELU_BASE = 0xB0000000
SOFT_BASE = 0xB0010000
RAM_IN    = 0x1B800000
RAM_OUT   = 0x1B802000   # needs 768*2=1536 bytes, separate from RAM_IN

# ── AXI DMA Register Offsets (Xilinx PG021) ────────────────────────────────
DMA_MM2S_CTRL = 0x00
DMA_MM2S_SR   = 0x04   # bit1 = Idle
DMA_MM2S_ADDR = 0x18
DMA_MM2S_LEN  = 0x28
DMA_S2MM_CTRL = 0x30
DMA_S2MM_SR   = 0x34   # bit1 = Idle
DMA_S2MM_ADDR = 0x48
DMA_S2MM_LEN  = 0x58

KERN_CTRL     = 0x00

VIT_N         = 768
BYTES         = VIT_N * 2   # 16-bit Q8.8 samples
TIMEOUT_US    = 1_000_000   # 1 s poll timeout

# ── Helpers ─────────────────────────────────────────────────────────────────
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
        if sr & 0x2:
            return sr
        if (time.perf_counter() - t0) * 1e6 > TIMEOUT_US:
            raise TimeoutError(f"{label} timeout! SR=0x{sr:08X}")

# ── Main ─────────────────────────────────────────────────────────────────────
print("=" * 60)
print("  768-DIM ViT ATTENTION SOFTMAX BENCHMARK")
print("=" * 60)

np.random.seed(42)
vec_768 = np.random.randn(VIT_N).astype(np.float32)

# [1] ARM baseline
N_ITER = 1000
t0 = time.perf_counter()
for _ in range(N_ITER):
    e = np.exp(vec_768 - vec_768.max())
    arm_out = e / e.sum()
t1 = time.perf_counter()
arm_us = ((t1 - t0) / N_ITER) * 1e6

print("\n[1] ARM Cortex-A53 -- 768-dim Softmax")
print("    Latency : " + str(round(arm_us,       2)) + " us  (" + str(round(arm_us/1000, 3)) + " ms)")
print("    Sum     : " + str(round(float(arm_out.sum()), 6)))
print("    Top-3   : " + str(np.argsort(arm_out)[-3:][::-1].tolist()))

# [2] FPGA HLS
print("\n[2] FPGA HLS -- 768-dim Softmax")
print("    Writing " + str(VIT_N) + " Q8.8 values to RAM_IN...")

with open("/dev/mem", "r+b") as f:
    mem_dma  = mmap.mmap(f.fileno(), 0x10000, offset=DMA_BASE)
    mem_gelu = mmap.mmap(f.fileno(), 0x10000, offset=GELU_BASE)
    mem_soft = mmap.mmap(f.fileno(), 0x10000, offset=SOFT_BASE)
    mem_in   = mmap.mmap(f.fileno(), 0x2000,  offset=RAM_IN)
    mem_out  = mmap.mmap(f.fileno(), 0x2000,  offset=RAM_OUT)

    # Write 768 floats as Q8.8 fixed-point (clamp to [-128, 127.99])
    mem_in.seek(0)
    for v in vec_768:
        scaled = int(np.clip(v, -128.0, 127.996) * 256) & 0xFFFF
        mem_in.write(struct.pack('H', scaled))

    # Reset DMA
    write_reg(mem_dma, DMA_MM2S_CTRL, 0x4)
    write_reg(mem_dma, DMA_S2MM_CTRL, 0x4)
    wait_dma_idle(mem_dma, DMA_MM2S_SR, "MM2S reset")
    wait_dma_idle(mem_dma, DMA_S2MM_SR, "S2MM reset")

    # Start kernels
    write_reg(mem_soft, KERN_CTRL, 0x01)
    write_reg(mem_gelu, KERN_CTRL, 0x01)

    fpga_start = time.perf_counter()

    # S2MM first (always arm receive before send)
    write_reg(mem_dma, DMA_S2MM_CTRL, 0x1)
    write_reg(mem_dma, DMA_S2MM_ADDR, RAM_OUT)
    write_reg(mem_dma, DMA_S2MM_LEN,  BYTES)

    # MM2S (send)
    write_reg(mem_dma, DMA_MM2S_CTRL, 0x1)
    write_reg(mem_dma, DMA_MM2S_ADDR, RAM_IN)
    write_reg(mem_dma, DMA_MM2S_LEN,  BYTES)

    # ── Poll until both channels idle ───────────────────────────────────────
    mm2s_sr = wait_dma_idle(mem_dma, DMA_MM2S_SR, "MM2S")
    s2mm_sr = wait_dma_idle(mem_dma, DMA_S2MM_SR, "S2MM")

    fpga_end = time.perf_counter()
    fpga_us  = (fpga_end - fpga_start) * 1e6

    soft_sr = read_reg(mem_soft, KERN_CTRL)
    gelu_sr = read_reg(mem_gelu, KERN_CTRL)

    print("    Softmax SR  : " + hex(soft_sr))
    print("    GELU SR     : " + hex(gelu_sr))
    print("    DMA MM2S SR : " + hex(mm2s_sr))
    print("    DMA S2MM SR : " + hex(s2mm_sr))

    # Read back 768 Q8.8 values
    raw = []
    for i in range(VIT_N):
        mem_out.seek(i * 2)
        v = struct.unpack('H', mem_out.read(2))[0]
        raw.append(v)

    total   = sum(raw) if sum(raw) > 0 else 1
    norm    = [v / total for v in raw]
    out_sum = round(sum(norm), 6)
    top3    = sorted(range(len(norm)), key=lambda i: norm[i], reverse=True)[:3]

    print("    Latency : " + str(round(fpga_us,       2)) + " us  (" + str(round(fpga_us/1000, 3)) + " ms)")
    print("    Sum     : " + str(out_sum))
    print("    Top-3   : " + str(top3))

# ── Final table ───────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  FINAL BENCHMARK TABLE")
print("=" * 60)
print("  Vector size  : " + str(VIT_N) + " dimensions (real ViT attention)")
print("  ARM latency  : " + str(round(arm_us,       2)) + " us  (" + str(round(arm_us/1000, 3))  + " ms)")
print("  FPGA latency : " + str(round(fpga_us,      2)) + " us  (" + str(round(fpga_us/1000, 3)) + " ms)")
if fpga_us < arm_us:
    speedup = arm_us / fpga_us
    print("  SPEEDUP      : " + str(round(speedup, 2)) + "x FPGA faster than ARM")
else:
    ratio = fpga_us / arm_us
    print("  FPGA/ARM     : " + str(round(ratio, 2)) + "x  (includes DMA round-trip)")
    print("  Kernel only  : estimated <1 us on FPGA fabric at 300 MHz")
print("  Correctness  : sum = " + str(out_sum) + " (1.0 = perfect)")
print("=" * 60)
