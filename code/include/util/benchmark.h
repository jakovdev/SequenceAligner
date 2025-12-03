#pragma once
#ifndef UTIL_BENCHMARK_H
#define UTIL_BENCHMARK_H

#include "system/types.h"

void bench_io_start(void);
void bench_align_start(void);
void bench_filter_start(void);

void bench_io_end(void);
void bench_align_end(void);
void bench_filter_end(void);

void bench_io_print(void);
void bench_align_print(void);
void bench_filter_print(void);

void bench_total_print(u64 alignments);

#endif /* UTIL_BENCHMARK_H */
