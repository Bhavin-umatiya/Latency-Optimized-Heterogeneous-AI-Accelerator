import mmap
import struct
import time

# Addresses
DMA_BASE  = 0xA0000000
GELU_BASE = 0xB0000000
SOFT_BASE = 0xB0010000
RAM_IN    = 0x1B800000  # Safe Input Buffer
RAM_OUT   = 0x1B801000  # Safe Output Buffer

def write_reg(mem, offset, value):
    mem.seek(offset)
    mem.write(struct.pack('I', value))

def read_reg(mem, offset):
    mem.seek(offset)
    return struct.unpack('I', mem.read(4))[0]

with open("/dev/mem", "r+b") as f:
    # Map Registers
    mem_dma  = mmap.mmap(f.fileno(), 0x10000, offset=DMA_BASE)
    mem_gelu = mmap.mmap(f.fileno(), 0x10000, offset=GELU_BASE)
    mem_soft = mmap.mmap(f.fileno(), 0x10000, offset=SOFT_BASE)
    # Map RAM Data Buffers
    mem_in   = mmap.mmap(f.fileno(), 0x1000, offset=RAM_IN)
    mem_out  = mmap.mmap(f.fileno(), 0x1000, offset=RAM_OUT)

    print("--- Preparing Data ---")
    # Put 128 numbers (0, 1, 2... 127) into RAM_IN (2 bytes each)
    for i in range(128):
        mem_in.seek(i*2)
        mem_in.write(struct.pack('H', i))  # 'H' is 16-bit unsigned

    print("--- Starting Kernels ---")
    write_reg(mem_soft, 0x00, 0x81)  # Start Softmax (ap_start | ap_continue)
    write_reg(mem_gelu, 0x00, 0x81)  # Start GELU   (ap_start | ap_continue)

    print("--- Starting DMA Transfer ---")
    # Reset DMA
    write_reg(mem_dma, 0x00, 0x4)
    write_reg(mem_dma, 0x30, 0x4)
    time.sleep(0.1)

    # Setup Receive (S2MM)
    write_reg(mem_dma, 0x30, 0x1)    # Run
    write_reg(mem_dma, 0x48, RAM_OUT) # Destination
    write_reg(mem_dma, 0x58, 256)     # Length: 128 elements * 2 bytes

    # Setup Send (MM2S)
    write_reg(mem_dma, 0x00, 0x1)    # Run
    write_reg(mem_dma, 0x18, RAM_IN)  # Source
    write_reg(mem_dma, 0x28, 256)     # Length: 128 elements * 2 bytes

    print("--- Waiting for Completion ---")
    timeout = 10
    while timeout > 0:
        dma_status = read_reg(mem_dma, 0x34)  # Check S2MM Status
        if dma_status & 0x1000:               # Idle bit
            break
        time.sleep(0.1)
        timeout -= 0.1

    if timeout <= 0:
        print("Error: DMA Timeout! (Is TLAST working?)")
    else:
        print("Success! Data received.")
        print("\n--- FIRST 5 OUTPUT VALUES ---")
        for i in range(5):
            mem_out.seek(i*2)
            val = struct.unpack('H', mem_out.read(2))[0]
            print("Index " + str(i) + ": " + str(val))

    print("\n--- PROJECT COMPLETE! ---")
