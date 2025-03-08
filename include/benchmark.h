#ifndef BENCHMARK_H
#define BENCHMARK_H

#include "common.h"

INLINE double get_time(void) {
#ifdef _WIN32
    static double freq_inv = 0.0;
    if (freq_inv == 0.0) {
        LARGE_INTEGER freq;
        QueryPerformanceFrequency(&freq);
        freq_inv = 1.0 / (double)freq.QuadPart;
    }
    LARGE_INTEGER count;
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart * freq_inv;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
#endif
}

typedef struct {
    double init;
    double align;
    double write;
    double total;
} BenchmarkTimes;

static BenchmarkTimes g_times = {0};

#define BENCH_INIT_START() double init_start = get_time()
#define BENCH_INIT_END() do { \
    if (get_mode_benchmark()) { \
        g_times.init = get_time() - init_start; \
        printf("\nInit time: %.6f seconds\n", g_times.init); \
    } \
} while(0)

#define BENCH_ALIGN_START() double align_start = get_time()
#define BENCH_ALIGN_END() do { \
    if (get_mode_benchmark()) { \
        g_times.align = get_time() - align_start; \
        printf("Alignment time: %.6f seconds\n", g_times.align); \
    } \
} while(0)

#define BENCH_WRITE_START() double write_start = get_time()
#define BENCH_WRITE_END() do { \
    if (get_mode_benchmark()) { \
        g_times.write = get_time() - write_start; \
        printf("Write time: %.6f seconds\n", g_times.write); \
    } \
} while(0)

#define BENCH_TOTAL() do { \
    if (get_mode_benchmark()) { \
        g_times.total = g_times.init + g_times.align + g_times.write; \
        printf("\nTotal time: %.6f seconds\n", g_times.total); \
    } \
} while(0)

#endif