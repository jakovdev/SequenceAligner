#pragma once
#ifndef ARGS_H
#define ARGS_H

#include "stdbool.h"

#include "biotypes.h"

extern void args_init(int argc, char* argv[]);
extern void args_print_config(void);

extern const char* args_input(void);
extern const char* args_output(void);
extern int args_gap_penalty(void);
extern int args_gap_open(void);
extern int args_gap_extend(void);
extern unsigned long args_thread_num(void);
extern AlignmentMethod args_align_method(void);
extern SequenceType args_sequence_type(void);
extern int args_scoring_matrix(void);
extern unsigned int args_compression(void);
extern float args_filter(void);
extern bool args_mode_multithread(void);
extern bool args_mode_benchmark(void);
extern bool args_mode_write(void);

#ifdef USE_CUDA
extern bool args_mode_cuda(void);
#endif

#endif // ARGS_H