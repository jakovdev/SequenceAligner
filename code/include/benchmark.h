#pragma once
#ifndef BENCHMARK_H
#define BENCHMARK_H

#include "biotypes.h"

extern void bench_io_start(void);
extern void bench_align_start(void);

extern void bench_io_end(void);
extern void bench_align_end(void);

extern void bench_print_align(void);
extern void bench_print_io(void);
extern void bench_print_total(alignment_size_t alignments);

#endif