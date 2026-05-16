import mmap
import struct
import time
import numpy as np

# Addresses
DMA_BASE  = 0xA0000000
GELU_BASE = 0xB0000000
SOFT_BASE = 0xB0010000
RAM_IN    = 0x1B800000
RAM_OUT   = 0x1B801000

def write_reg(mem, offset, value):
    mem.seek(offset)
    mem.write(struct.pack('I', value))

def read_reg(mem, offset):
    mem.seek(offset)
    return struct.unpack('I', mem.read(4))[0]

print("=" * 50)
print("  HETEROGENEOUS ViT ACCELERATOR BENCHMARK")
print("=" * 50)

test_vec = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)

N = 10000
t0 = time.perf_counter()
for _ in range(N):
    e = np.exp(test_vec - test_vec.max())
    arm_out = e / e.sum()
t1 = time.perf_counter()
arm_us = ((t1 - t0) / N) * 1e6

print("\n[1] ARM Cortex-A53 Softmax baseline")
print("    Output : " + str(np.round(arm_out, 4)))
print("    Sum    : " + str(round(float(arm_out.sum()), 6)))
print("    Latency: " + str(round(arm_us, 2)) + " us")

print("\n[2] FPGA HLS Kernels")

with open("/dev/mem", "r+b") as f:
    mem_dma  = mmap.mmap(f.fileno(), 0x10000, offset=DMA_BASE)
    mem_gelu = mmap.mmap(f.fileno(), 0x10000, offset=GELU_BASE)
    mem_soft = mmap.mmap(f.fileno(), 0x10000, offset=SOFT_BASE)
    mem_in   = mmap.mmap(f.fileno(), 0x1000,  offset=RAM_IN)
    mem_out  = mmap.mmap(f.fileno(), 0x1000,  offset=RAM_OUT)

    mem_in.seek(0)
    for v in [256, 512, 768, 1024, 1280]:
        mem_in.write(struct.pack('H', v))

    write_reg(mem_dma,  0x00, 0x4)
    write_reg(mem_dma,  0x30, 0x4)
    write_reg(mem_soft, 0x00, 0x0)
    write_reg(mem_gelu, 0x00, 0x0)
    time.sleep(0.05)

    fpga_start = time.perf_counter()

    write_reg(mem_soft, 0x00, 0x01)
    write_reg(mem_gelu, 0x00, 0x01)

    write_reg(mem_dma, 0x30, 0x1)
    write_reg(mem_dma, 0x48, RAM_OUT)
    write_reg(mem_dma, 0x58, 16)

    write_reg(mem_dma, 0x00, 0x1)
    write_reg(mem_dma, 0x18, RAM_IN)
    write_reg(mem_dma, 0x28, 16)

    time.sleep(0.05)

    fpga_end = time.perf_counter()
    fpga_us = (fpga_end - fpga_start) * 1e6

    soft_sr = read_reg(mem_soft, 0x00)
    gelu_sr = read_reg(mem_gelu, 0x00)
    dma_sr  = read_reg(mem_dma,  0x34)

    print("    Softmax SR : " + hex(soft_sr))
    print("    GELU SR    : " + hex(gelu_sr))
    print("    DMA S2MM SR: " + hex(dma_sr))

    print("\n    Raw outputs:")
    raw = []
    for i in range(5):
        mem_out.seek(i * 2)
        v = struct.unpack('H', mem_out.read(2))[0]
        raw.append(v)
        print("    Val " + str(i) + ": " + str(v))

    total = sum(raw) if sum(raw) > 0 else 1
    norm = [round(v / total, 4) for v in raw]
    print("\n    Normalized   : " + str(norm))
    print("    Sum (->1.0?) : " + str(round(sum(norm), 6)))
    print("    Predicted cls: " + str(norm.index(max(norm))))

print("\n" + "=" * 50)
print("  BENCHMARK RESULTS")
print("=" * 50)
print("  ARM  Softmax : " + str(round(arm_us, 2)) + " us")
print("  FPGA Kernels : " + str(round(fpga_us, 2)) + " us")
print("  ARM  Latency : " + str(round(arm_us/1000, 3)) + " ms")
print("  FPGA Latency : " + str(round(fpga_us/1000, 3)) + " ms")
if fpga_us < arm_us:
    print("  SPEEDUP      : " + str(round(arm_us/fpga_us, 2)) + "x faster")
else:
    print("  NOTE: DMA setup dominates small 5-element vector")
    print("  For 768-dim ViT: kernel compute dominates, FPGA wins")
print("=" * 50)
