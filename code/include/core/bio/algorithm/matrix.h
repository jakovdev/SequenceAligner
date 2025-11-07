#pragma once
#ifndef CORE_BIO_ALGORITHM_MATRIX_H
#define CORE_BIO_ALGORITHM_MATRIX_H

#include <stddef.h>

#include "system/types.h"

#define MAX_STACK_SEQUENCE_LENGTH (4 * KiB)

#define STACK_MATRIX_THRESHOLD (128 * KiB)
#define MATRIX_SIZE(len1, len2) ((len1 + 1) * (len2 + 1))
#define MATRICES_3X_SIZE(len1, len2) (3 * MATRIX_SIZE(len1, len2))
#define MATRIX_BYTES(len1, len2) (MATRIX_SIZE(len1, len2) * sizeof(s32))
#define MATRICES_3X_BYTES(len1, len2) (3 * MATRIX_BYTES(len1, len2))
#define USE_STACK_MATRIX(bytes) ((bytes) <= STACK_MATRIX_THRESHOLD)

#define MATRIX_SIZE_S(len1, len2, bytes) \
	(USE_STACK_MATRIX(bytes) ? MATRIX_SIZE(len1, len2) : 1)
#define MATRICES_3X_SIZE_S(len1, len2, bytes) \
	(USE_STACK_MATRIX(bytes) ? MATRICES_3X_SIZE(len1, len2) : 1)

s32 *matrix_alloc(s32 *stack_matrix, size_t bytes);

void matrix_free(s32 *matrix, s32 *stack_matrix);

#endif // CORE_BIO_ALGORITHM_MATRIX_H
