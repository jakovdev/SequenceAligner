#pragma once
#ifndef BIO_ALGORITHM_MATRIX_H
#define BIO_ALGORITHM_MATRIX_H

#include "system/compiler.h"
#include "system/types.h"

extern thread_local s32 *g_restrict MATRIX;
extern thread_local s32 *g_restrict MATCH;
extern thread_local s32 *g_restrict GAP_X;
extern thread_local s32 *g_restrict GAP_Y;

void matrix_buffers_init(void);

void matrix_buffers_free(void);

#endif /* BIO_ALGORITHM_MATRIX_H */
