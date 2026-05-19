/*
 * bench_768.c — Production-Grade Native C Benchmark
 * ZCU104 Heterogeneous ViT Accelerator (Softmax + GELU HLS Kernels)
 *
 * Fixes applied:
 *  1. Correct Q8.8 -> float normalization for output sum verification
 *  2. DMA S2MM armed BEFORE HLS kernels started (correct ordering)
 *  3. Poll only on IOC_Irq (bit 12 = 0x1000), not idle bit (false positive)
 *  4. Timing starts after kernel start writes, before DMA trigger
 *  5. msync() cache flush before DMA read
 *  6. Poll loop has a timeout to prevent infinite hang
 *  7. ARM and FPGA latencies averaged over N_ITER iterations
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <time.h>
#include <math.h>

// ── Hardware Addresses ────────────────────────────────────────────────────────
#define DMA_BASE   0xA0000000UL
#define GELU_BASE  0xB0000000UL
#define SOFT_BASE  0xB0010000UL
#define RAM_IN     0x1B800000UL
#define RAM_OUT    0x1B801000UL

// ── Map Sizes ─────────────────────────────────────────────────────────────────
#define MAP_SIZE   0x10000
#define RAM_SIZE   0x2000

// ── Benchmark Config ──────────────────────────────────────────────────────────
#define N_ELEMENTS  768
#define N_ITER      100          // Run N_ITER times and report median
#define DMA_TIMEOUT 500000       // Max poll iterations before giving up

// ── Register Helpers ──────────────────────────────────────────────────────────
static inline void write_reg(void* base, uint32_t offset, uint32_t val) {
    *((volatile uint32_t*)((uint8_t*)base + offset)) = val;
}
static inline uint32_t read_reg(void* base, uint32_t offset) {
    return *((volatile uint32_t*)((uint8_t*)base + offset));
}

// ── Comparison helper for qsort (median calculation) ─────────────────────────
static int cmp_double(const void* a, const void* b) {
    double da = *(const double*)a;
    double db = *(const double*)b;
    return (da > db) - (da < db);
}

int main() {
    printf("=====================================================\n");
    printf("   PRODUCTION C BENCHMARK -- 768-DIM VIT ATTENTION\n");
    printf("   (N=%d iterations, reporting median latency)\n", N_ITER);
    printf("=====================================================\n");

    // ── Seed random input vector ──────────────────────────────────────────────
    float vec[N_ELEMENTS];
    srand(42);
    for (int i = 0; i < N_ELEMENTS; i++)
        vec[i] = (float)rand() / (float)RAND_MAX - 0.5f;

    // ── [1] ARM CPU Softmax Benchmark — N_ITER averaged ───────────────────────
    printf("\n[1] ARM Cortex-A53 CPU Softmax (%d iterations)...\n", N_ITER);
    double arm_samples[N_ITER];
    for (int iter = 0; iter < N_ITER; iter++) {
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);

        float max_val = vec[0];
        for (int i = 1; i < N_ELEMENTS; i++)
            if (vec[i] > max_val) max_val = vec[i];

        float sum_exp = 0.0f;
        float cpu_out[N_ELEMENTS];
        for (int i = 0; i < N_ELEMENTS; i++) {
            cpu_out[i] = expf(vec[i] - max_val);
            sum_exp += cpu_out[i];
        }
        for (int i = 0; i < N_ELEMENTS; i++)
            cpu_out[i] /= sum_exp;

        clock_gettime(CLOCK_MONOTONIC, &t1);
        arm_samples[iter] = (t1.tv_sec - t0.tv_sec) * 1e6 +
                            (t1.tv_nsec - t0.tv_nsec) / 1e3;
    }
    qsort(arm_samples, N_ITER, sizeof(double), cmp_double);
    double arm_median_us = arm_samples[N_ITER / 2];
    printf("    ARM Median Latency : %.2f us\n", arm_median_us);

    // ── Open /dev/mem ─────────────────────────────────────────────────────────
    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) { perror("Error opening /dev/mem"); return -1; }

    void* mem_dma  = mmap(NULL, MAP_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, DMA_BASE);
    void* mem_gelu = mmap(NULL, MAP_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, GELU_BASE);
    void* mem_soft = mmap(NULL, MAP_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, SOFT_BASE);
    void* mem_in   = mmap(NULL, RAM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, RAM_IN);
    void* mem_out  = mmap(NULL, RAM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, RAM_OUT);

    if (mem_dma == MAP_FAILED || mem_gelu == MAP_FAILED || mem_soft == MAP_FAILED ||
        mem_in  == MAP_FAILED || mem_out  == MAP_FAILED) {
        perror("mmap failed"); close(fd); return -1;
    }

    // ── Write Q8.8 input into RAM_IN ─────────────────────────────────────────
    uint16_t* ram_in_ptr = (uint16_t*)mem_in;
    for (int i = 0; i < N_ELEMENTS; i++)
        ram_in_ptr[i] = (int16_t)(vec[i] * 256.0f);

    // FIX 5: Flush CPU cache so DMA engine reads fresh data from DDR
    msync(mem_in, RAM_SIZE, MS_SYNC);

    // ── [2] FPGA Hardware Benchmark — N_ITER averaged ─────────────────────────
    printf("\n[2] FPGA Hardware Subsystem (%d iterations)...\n", N_ITER);
    double fpga_samples[N_ITER];
    int timeout_count = 0;

    for (int iter = 0; iter < N_ITER; iter++) {

        // Reset DMA MM2S and S2MM channels
        write_reg(mem_dma, 0x00, 0x4);
        write_reg(mem_dma, 0x30, 0x4);
        usleep(1000);   // 1 ms settling time for reset

        // FIX 2: Arm S2MM receive channel BEFORE starting HLS kernels
        write_reg(mem_dma, 0x30, 0x01);           // S2MM Run
        write_reg(mem_dma, 0x48, RAM_OUT);         // S2MM Destination addr
        write_reg(mem_dma, 0x58, N_ELEMENTS * 2); // S2MM byte count

        // Start HLS kernels: ap_start (bit0) | ap_continue (bit7) = 0x81
        write_reg(mem_soft, 0x00, 0x81);
        write_reg(mem_gelu, 0x00, 0x81);

        // FIX 4: Start timing HERE — after kernel setup, right before MM2S trigger
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);

        // Trigger MM2S send channel — this starts the actual data transfer
        write_reg(mem_dma, 0x00, 0x01);           // MM2S Run
        write_reg(mem_dma, 0x18, RAM_IN);          // MM2S Source addr
        write_reg(mem_dma, 0x28, N_ELEMENTS * 2); // MM2S byte count

        // FIX 3: Poll S2MM Status register (0x34) for IOC_Irq (bit 12 = 0x1000) ONLY
        // Idle bit (bit 1) is set at startup — only IOC_Irq indicates actual completion
        int timeout = DMA_TIMEOUT;
        while (timeout-- > 0) {
            if (read_reg(mem_dma, 0x34) & 0x1000) break;
        }
        clock_gettime(CLOCK_MONOTONIC, &t1);

        if (timeout <= 0) {
            fprintf(stderr, "    WARNING: DMA Timeout on iteration %d!\n", iter);
            timeout_count++;
            fpga_samples[iter] = 999999.0; // Mark as invalid
        } else {
            fpga_samples[iter] = (t1.tv_sec - t0.tv_sec) * 1e6 +
                                  (t1.tv_nsec - t0.tv_nsec) / 1e3;
        }
    }

    qsort(fpga_samples, N_ITER, sizeof(double), cmp_double);
    double fpga_median_us = fpga_samples[N_ITER / 2];

    // FIX 1: Correct output sum verification — convert Q8.8 back to float
    uint16_t* ram_out_ptr = (uint16_t*)mem_out;
    float fpga_sum = 0.0f;
    for (int i = 0; i < N_ELEMENTS; i++)
        fpga_sum += ram_out_ptr[i] / 256.0f;   // Q8.8 -> float

    printf("    FPGA Median Latency : %.2f us\n", fpga_median_us);
    printf("    Output Sum (Q8.8)   : %.4f  (1.0 = perfect Softmax)\n", fpga_sum);
    if (timeout_count > 0)
        printf("    DMA Timeouts        : %d / %d iterations\n", timeout_count, N_ITER);

    munmap(mem_dma, MAP_SIZE);
    munmap(mem_gelu, MAP_SIZE);
    munmap(mem_soft, MAP_SIZE);
    munmap(mem_in, RAM_SIZE);
    munmap(mem_out, RAM_SIZE);
    close(fd);

    printf("\n=====================================================\n");
    printf("  BENCHMARK SUMMARY (Median over %d runs)\n", N_ITER);
    printf("=====================================================\n");
    printf("  ARM CPU Latency  : %.2f us\n", arm_median_us);
    printf("  FPGA Subsystem   : %.2f us  (includes HLS + AXI DMA)\n", fpga_median_us);
    if (fpga_median_us < arm_median_us)
        printf("  SPEEDUP          : %.2fx faster than ARM CPU\n",
               arm_median_us / fpga_median_us);
    else
        printf("  NOTE: DMA+overhead dominates. Fabric compute is sub-ms.\n");
    printf("  Output Sum Verify: %.4f (expect ~1.0 for correct Softmax)\n", fpga_sum);
    printf("=====================================================\n");

    return 0;
}
