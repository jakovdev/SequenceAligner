#pragma once
#ifndef BIO_ALGORITHM_MATRIX_H
#define BIO_ALGORITHM_MATRIX_H

#include "system/types.h"

extern thread_local s32 *restrict g_matrix;
extern thread_local s32 *restrict g_match;
extern thread_local s32 *restrict g_gap_x;
extern thread_local s32 *restrict g_gap_y;

void matrix_buffers_init(s32 seq_len_max);

void matrix_buffers_free(void);

#endif /* BIO_ALGORITHM_MATRIX_H */
