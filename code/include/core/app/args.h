#pragma once
#ifndef CORE_APP_ARGS_H
#define CORE_APP_ARGS_H

#include "core/bio/types.h"

void args_init(int argc, char *argv[]);
void args_print_config(void);

const char *args_input(void);
const char *args_output(void);
int args_gap_penalty(void);
int args_gap_open(void);
int args_gap_extend(void);
unsigned long args_thread_num(void);
AlignmentMethod args_align_method(void);
SequenceType args_sequence_type(void);
int args_scoring_matrix(void);
unsigned int args_compression(void);
float args_filter(void);
bool args_mode_benchmark(void);
bool args_mode_write(void);
bool args_mode_cuda(void);
bool args_force(void);

#endif // CORE_APP_ARGS_H
