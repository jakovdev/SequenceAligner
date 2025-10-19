#pragma once
#ifndef UTIL_BENCHMARK_H
#define UTIL_BENCHMARK_H

#include <stddef.h>

extern void bench_io_start(void);
extern void bench_align_start(void);
extern void bench_filter_start(void);

extern void bench_io_end(void);
extern void bench_align_end(void);
extern void bench_filter_end(void);

extern void bench_io_print(void);
extern void bench_align_print(void);
extern void bench_filter_print(size_t filtered);

extern void bench_total_print(size_t alignments);

#endif // UTIL_BENCHMARK_H
