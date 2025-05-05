#ifndef ARGS_H
#define ARGS_H

#include "stdbool.h"

extern void args_init(int argc, char* argv[]);
extern void args_print_config(void);

extern const char* args_path_input(void);
extern const char* args_path_output(void);
extern int args_gap_penalty(void);
extern int args_gap_start(void);
extern int args_gap_extend(void);
extern int args_thread_num(void);
extern int args_align_method(void);
extern int args_sequence_type(void);
extern int args_scoring_matrix(void);
extern int args_compression_level(void);
extern float args_filter_threshold(void);
extern bool args_mode_multithread(void);
extern bool args_mode_benchmark(void);
extern bool args_mode_filter(void);
extern bool args_mode_write(void);

#endif // ARGS_H