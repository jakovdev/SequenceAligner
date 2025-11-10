#pragma once
#ifndef APP_ARGS_H
#define APP_ARGS_H

#include "bio/types.h"
#include "system/types.h"

void args_init(int argc, char *argv[]);
void args_print_config(void);

const char *args_input(void);
const char *args_output(void);
s32 args_gap_pen(void);
s32 args_gap_open(void);
s32 args_gap_ext(void);
int args_thread_num(void);
enum AlignmentMethod args_align_method(void);
enum SequenceType args_sequence_type(void);
int args_sub_matrix(void);
u8 args_compression(void);
double args_filter(void);
bool args_mode_benchmark(void);
bool args_mode_write(void);
bool args_mode_cuda(void);
bool args_force(void);

#endif // APP_ARGS_H
