#pragma once
#ifndef UTIL_BENCHMARK_H
#define UTIL_BENCHMARK_H

#include <stddef.h>

void bench_io_start(void);
void bench_align_start(void);
void bench_filter_start(void);

void bench_io_end(void);
void bench_align_end(void);
void bench_filter_end(void);

void bench_io_print(void);
void bench_align_print(void);
void bench_filter_print(size_t filtered);

void bench_total_print(size_t alignments);

#endif // UTIL_BENCHMARK_H
