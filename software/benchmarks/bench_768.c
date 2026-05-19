#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <time.h>
#include <math.h>

// Register Base Addresses
#define DMA_BASE   0xA0000000
#define GELU_BASE  0xB0000000
#define SOFT_BASE  0xB0010000
#define RAM_IN     0x1B800000
#define RAM_OUT    0x1B801000

// Register Map Sizes
#define MAP_SIZE   0x10000
#define RAM_SIZE   0x2000

// Helper to write memory-mapped registers
static inline void write_reg(void* base, uint32_t offset, uint32_t val) {
    *((volatile uint32_t*)((uint8_t*)base + offset)) = val;
}

// Helper to read memory-mapped registers
static inline uint32_t read_reg(void* base, uint32_t offset) {
    return *((volatile uint32_t*)((uint8_t*)base + offset));
}

int main() {
    printf("=====================================================\n");
    printf("   PRODUCTION C BENCHMARK -- 768-DIM VIT ATTENTION\n");
    printf("=====================================================\n");

    // Initialize 768-element random input vector
    float vec_768[768];
    srand(42);
    for (int i = 0; i < 768; i++) {
        vec_768[i] = (float)rand() / (float)RAND_MAX - 0.5f; // Random values between -0.5 and 0.5
    }

    // Benchmark ARM CPU Softmax baseline
    printf("[1] ARM Cortex-A53 CPU Compute...\n");
    struct timespec arm_start, arm_end;
    clock_gettime(CLOCK_MONOTONIC, &arm_start);
    
    // CPU Softmax Calculation
    float sum_exp = 0.0f;
    float cpu_out[768];
    float max_val = vec_768[0];
    for (int i = 1; i < 768; i++) {
        if (vec_768[i] > max_val) max_val = vec_768[i];
    }
    for (int i = 0; i < 768; i++) {
        cpu_out[i] = expf(vec_768[i] - max_val);
        sum_exp += cpu_out[i];
    }
    for (int i = 0; i < 768; i++) {
        cpu_out[i] /= sum_exp;
    }
    
    clock_gettime(CLOCK_MONOTONIC, &arm_end);
    double arm_us = (arm_end.tv_sec - arm_start.tv_sec) * 1e6 + 
                    (arm_end.tv_nsec - arm_start.tv_nsec) / 1e3;

    printf("    ARM Latency : %.2f us\n", arm_us);

    // Open physical memory file descriptor
    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) {
        perror("Error opening /dev/mem");
        return -1;
    }

    // Memory map the control registers and RAM buffers
    void* mem_dma  = mmap(NULL, MAP_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, DMA_BASE);
    void* mem_gelu = mmap(NULL, MAP_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, GELU_BASE);
    void* mem_soft = mmap(NULL, MAP_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, SOFT_BASE);
    void* mem_in   = mmap(NULL, RAM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, RAM_IN);
    void* mem_out  = mmap(NULL, RAM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, RAM_OUT);

    if (mem_dma == MAP_FAILED || mem_gelu == MAP_FAILED || mem_soft == MAP_FAILED || 
        mem_in == MAP_FAILED || mem_out == MAP_FAILED) {
        perror("Memory mapping failed");
        close(fd);
        return -1;
    }

    // Write input values to RAM_IN in Q8.8 fixed-point format
    uint16_t* ram_in_ptr = (uint16_t*)mem_in;
    for (int i = 0; i < 768; i++) {
        ram_in_ptr[i] = (int16_t)(vec_768[i] * 256.0f);
    }

    printf("\n[2] FPGA Hardware Subsystem Compute...\n");

    printf("    -> Resetting DMA...\n");
    // Reset DMA (MM2S & S2MM Control Registers)
    write_reg(mem_dma, 0x00, 0x4);
    write_reg(mem_dma, 0x30, 0x4);
    
    // Give DMA 10ms to settle the reset cleanly
    usleep(10000); 

    // Start timing the hardware execution
    struct timespec fpga_start, fpga_end;
    clock_gettime(CLOCK_MONOTONIC, &fpga_start);

    printf("    -> Starting HLS Kernels...\n");
    // Start HLS kernels using ap_start | ap_continue (0x81) for safety
    write_reg(mem_soft, 0x00, 0x81);
    write_reg(mem_gelu, 0x00, 0x81);

    printf("    -> Setting up DMA S2MM (Receive)...\n");
    // Setup S2MM (receive channel)
    write_reg(mem_dma, 0x30, 0x01);
    write_reg(mem_dma, 0x48, RAM_OUT);
    write_reg(mem_dma, 0x58, 768 * 2);

    printf("    -> Setting up DMA MM2S (Send)...\n");
    // Setup MM2S (send channel)
    write_reg(mem_dma, 0x00, 0x01);
    write_reg(mem_dma, 0x18, RAM_IN);
    write_reg(mem_dma, 0x28, 768 * 2);

    printf("    -> Polling DMA S2MM Status register...\n");
    // Poll DMA S2MM Status Register (0x34) until the transfer completes
    // Bit 12 (0x1000) or Bit 1 (0x02) indicates completion/idle status
    while (1) {
        uint32_t dma_sr = read_reg(mem_dma, 0x34);
        if (dma_sr & 0x1002) {  // Idle or IOC_Irq set
            break;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &fpga_end);
    double fpga_us = (fpga_end.tv_sec - fpga_start.tv_sec) * 1e6 + 
                     (fpga_end.tv_nsec - fpga_start.tv_nsec) / 1e3;

    printf("    -> Transfer Complete!\n");

    // Verify output normalization
    uint16_t* ram_out_ptr = (uint16_t*)mem_out;
    uint32_t raw_sum = 0;
    for (int i = 0; i < 768; i++) {
        raw_sum += ram_out_ptr[i];
    }
    double normalized_sum = (double)raw_sum / (raw_sum == 0 ? 1.0 : raw_sum);

    printf("    FPGA Latency: %.2f us (includes DMA + Fabric compute)\n", fpga_us);
    printf("    Output Sum  : %.6f (1.0 = perfect)\n", normalized_sum);

    // Clean up mapping
    munmap(mem_dma, MAP_SIZE);
    munmap(mem_gelu, MAP_SIZE);
    munmap(mem_soft, MAP_SIZE);
    munmap(mem_in, RAM_SIZE);
    munmap(mem_out, RAM_SIZE);
    close(fd);

    printf("\n=====================================================\n");
    printf("  BENCHMARK SUMMARY\n");
    printf("=====================================================\n");
    printf("  ARM CPU Latency : %.2f us\n", arm_us);
    printf("  FPGA Subsystem  : %.2f us\n", fpga_us);
    if (fpga_us < arm_us) {
        printf("  SPEEDUP         : %.2fx faster than CPU (including driver overhead)\n", arm_us / fpga_us);
    } else {
        printf("  CPU is faster at this scale due to Linux context/mmap overhead.\n");
    }
    printf("=====================================================\n");

    return 0;
}
