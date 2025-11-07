#pragma once
#ifndef CORE_APP_ARGS_H
#define CORE_APP_ARGS_H

#include "core/bio/types.h"
#include "system/types.h"

void args_init(int argc, char *argv[]);
void args_print_config(void);

const char *args_input(void);
const char *args_output(void);
s32 args_gap_penalty(void);
s32 args_gap_open(void);
s32 args_gap_extend(void);
int args_thread_num(void);
enum AlignmentMethod args_align_method(void);
enum SequenceType args_sequence_type(void);
int args_scoring_matrix(void);
u8 args_compression(void);
double args_filter(void);
bool args_mode_benchmark(void);
bool args_mode_write(void);
bool args_mode_cuda(void);
bool args_force(void);

#endif // CORE_APP_ARGS_H
