#pragma once
#ifndef CORE_APP_ARGS_H
#define CORE_APP_ARGS_H

#include "core/bio/types.h"

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
extern bool args_mode_benchmark(void);
extern bool args_mode_write(void);
extern bool args_mode_cuda(void);

#endif // CORE_APP_ARGS_H