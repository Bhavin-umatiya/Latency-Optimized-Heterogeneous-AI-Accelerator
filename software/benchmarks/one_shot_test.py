import mmap
import struct
import time

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

with open("/dev/mem", "r+b") as f:
    mem_dma  = mmap.mmap(f.fileno(), 0x10000, offset=DMA_BASE)
    mem_gelu = mmap.mmap(f.fileno(), 0x10000, offset=GELU_BASE)
    mem_soft = mmap.mmap(f.fileno(), 0x10000, offset=SOFT_BASE)
    mem_in   = mmap.mmap(f.fileno(), 0x1000, offset=RAM_IN)
    mem_out  = mmap.mmap(f.fileno(), 0x1000, offset=RAM_OUT)

    # 1. Reset everything
    write_reg(mem_dma, 0x00, 0x4)
    write_reg(mem_dma, 0x30, 0x4)
    write_reg(mem_soft, 0x00, 0x0)
    write_reg(mem_gelu, 0x00, 0x0)
    time.sleep(0.1)

    # 2. Start Kernels in "One-Shot" mode (0x01)
    write_reg(mem_soft, 0x00, 0x01)
    write_reg(mem_gelu, 0x00, 0x01)

    # 3. Setup DMA Receive
    write_reg(mem_dma, 0x30, 0x1) 
    write_reg(mem_dma, 0x48, RAM_OUT)
    write_reg(mem_dma, 0x58, 256)

    # 4. Setup DMA Send
    write_reg(mem_dma, 0x00, 0x1)
    write_reg(mem_dma, 0x18, RAM_IN)
    write_reg(mem_dma, 0x28, 256)

    print("Transfer started...")
    time.sleep(1)

    print(f"Softmax Done? {hex(read_reg(mem_soft, 0x00))}")
    print(f"GELU Done?    {hex(read_reg(mem_gelu, 0x00))}")
    print(f"DMA S2MM SR:  {hex(read_reg(mem_dma, 0x34))}")

    # Read back 5 values
    print("\n--- Result Check ---")
    for i in range(5):
        mem_out.seek(i*2)
        print(f"Val {i}: {struct.unpack('H', mem_out.read(2))[0]}")
